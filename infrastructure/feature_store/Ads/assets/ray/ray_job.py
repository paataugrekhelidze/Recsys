import ray
import ray.data.datasource.partitioning as partitioning
import polars as pl
from datetime import datetime, timedelta
import os
from ray.data.datasource.partitioning import PathPartitionFilter

# Initialize Ray (connects to the cluster defined in rayCluster.yaml)
ray.init(
    address='auto'
)

format_string = '%Y-%m-%d'

window_days = int(os.getenv('WINDOW_DAYS', 1))
# start_date = os.getenv('START_DATE', datetime.now().strftime(format_string))
end_date = os.getenv('END_DATE', datetime.now().strftime(format_string))

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield (start_date + timedelta(n)).strftime(format_string)

# expand days_to_read to include the window lookback period
# e.g. if start_date=2026-03-20 and window_days=5, then last_5_clicks at 2026-03-20 will need data from 2026-03-15
end_date = datetime.strptime(end_date, format_string)
start_date = end_date - timedelta(days=window_days)
dates_to_read = list(daterange(start_date, end_date))

def compute_rolling_windows(user_df, days: int, end_date: datetime):
    """Computes the last 5 unique campaigns clicked/converted in a window."""

    if not isinstance(user_df, pl.DataFrame):
        user_df = pl.from_arrow(user_df)
    
    # Fail fast if the upstream data contract is broken
    assert user_df.schema["event_timestamp"] in (pl.Datetime, pl.Datetime("us")), \
        "Data pipeline error: event_timestamp is not a Datetime object"

    user_df = user_df.sort('event_timestamp')
    last_5_clicks = []
    last_5_conversions = []
    timestamps = user_df['event_timestamp']

    # For each (uid, timestamp)
    for idx in range(user_df.height):
        current_time = timestamps[idx]
        # Get uid data within the appropriate range
        cutoff_time = current_time - pl.duration(days=days)
        mask = (timestamps >= cutoff_time) & (timestamps < current_time)
        window_df = user_df.filter(mask)
        # Last 5 unique campaigns for clicks
        click_campaigns = window_df.filter(pl.col('click') == 1)['campaign'].unique()[-5:]
        last_5_clicks.append(",".join([str(x) for x in click_campaigns]))
        # Last 5 unique campaigns for conversions
        conv_campaigns = window_df.filter(pl.col('conversion') == 1)['campaign'].unique()[-5:]
        last_5_conversions.append(",".join([str(x) for x in conv_campaigns]))

    user_df = user_df.with_columns([
        pl.Series('last_5_clicks', last_5_clicks),
        pl.Series('last_5_conversions', last_5_conversions)
    ])


    # only return date range that was submitted
    # ETL outputs jobs for a single day
    return user_df.filter(
        (pl.col('event_timestamp') >= end_date) & (pl.col('event_timestamp') < end_date + timedelta(days=1))
    ).to_arrow()
    
    # alternative implementation, although it takes the same amount of time
    # window_str = f"{days}d"
    # def get_state(df, filter_col, new_name):
    #     return (
    #         df.filter(pl.col(filter_col) == 1)
    #         .rolling(
    #             index_column="event_timestamp", 
    #             period=window_str,
    #             closed="right"
    #         )
    #         .agg(
    #             # Get unique campaigns, maintain chronological order, take last 5
    #             pl.col("campaign").unique(maintain_order=True).tail(5).alias(new_name)
    #         )
    #     )

    # click_state = get_state(user_df, "click", "last_5_clicks")
    # conv_state = get_state(user_df, "conversion", "last_5_conversions")
    
    # # Join States back to Main DataFrame (Point-in-Time Join)
    # # strategy='backward' ensures we only see data from BEFORE the current timestamp
    # user_df = user_df.join_asof(
    #     click_state, 
    #     on="event_timestamp", 
    #     # by="uid", 
    #     strategy="backward"
    # ).join_asof(
    #     conv_state, 
    #     on="event_timestamp", 
    #     # by="uid", # batch is already grouped by id, this is redundant
    #     strategy="backward"
    # )

    # return (
    #     user_df.with_columns([
    #         pl.col("last_5_clicks").fill_null(pl.lit([], dtype=pl.List(pl.Utf8))),
    #         pl.col("last_5_conversions").fill_null(pl.lit([], dtype=pl.List(pl.Utf8)))
    #     ])
    #     .filter(pl.col('event_timestamp').is_between(start_date, end_date))
    #     .to_arrow()
    # )

date_filter = PathPartitionFilter.of(
    style="hive",
    filter_fn=lambda d: f"{d['year']}-{int(d['month']):02d}-{int(d['day']):02d}" in dates_to_read
)

# Read the partitioned raw data from S3
# only reads specified partitions
# e.g.
# s3://<MY-BUCKET>/criteo/unprocessed/year=2026/month=03/day=05/data.parquet
# s3://<MY-BUCKET>/criteo/unprocessed/year=2026/month=03/day=06/data.parquet
schema = partitioning.Partitioning('hive', field_names=['year', 'month', 'day'])
# ray distributes stateless read tasks across workers
raw_ds = ray.data.read_parquet(
    's3://<MY-BUCKET>/criteo/unprocessed/', 
    partitioning=schema,
    # more efficient than filter because it prevents Ray from even opening files that don't match the date list.
    partition_filter=date_filter ,
    columns=['event_timestamp', 'uid', 'campaign', 'click', 'conversion'] + [f'cat{i}' for i in range(1, 10)]
)

print(f"Files being processed: {raw_ds.input_files()}")

# perform sequential window aggregation per group (uid)
# Using groupBy, Ray reshuffles the data so that all rows with the same uid end up in the same block.
# map_groups receives a data block that contains all values of the same uid (no uid appears in more than one block)
# lambda function (compute_rolling_windows) is applied once per block (so, 16 times in parallel)
# https://docs.ray.io/en/latest/data/data-internals.html#hash-shuffling
# NOTE: OOM is not handled well in this situation, when a data block is deserialized (not spillable to disk) for a lambda function in map_groups it may not fill in memory
transformed_ds = raw_ds.groupby(
    'uid', 
    #num_partitions=16 # hash(uid) % num_partitions - a block id that all uid data will be assigned to.
    ).map_groups(
    lambda df: compute_rolling_windows(
        df, 
        days=window_days,
        # start_date = start_date,
        end_date = end_date
    ), 
    batch_format='pyarrow'
)

# # Write the transformed dataset back to S3
# transformed_ds.write_parquet(
#     f"s3://<MY-BUCKET>/feast/criteo/transformed/features/year={end_date.year}/month={end_date.month:02d}/day={end_date.day:02d}/",
# )

# Ray's native write_parquet creates huge files for some reason...

# # instead load the arrow tables in polars and then call write_parquet (often 10x reduction in file size)
# # load arrow tables and save one block at a time to prevent OOM
# for ix, table_ref in enumerate(transformed_ds.to_arrow_refs()):
#     pl_df = pl.from_arrow(ray.get(table_ref))
#     pl_df.write_parquet(f"s3://<MY-BUCKET>/feast/criteo/transformed/features/year={end_date.year}/month={end_date.month:02d}/day={end_date.day:02d}/table-{ix:02d}.parquet", use_pyarrow=True)

# NOTE: may cause OOM, but one file per date partition keeps Redshift spectrum more efficient.
dfs = [pl.from_arrow(ray.get(table_ref)) for table_ref in transformed_ds.to_arrow_refs()]
combined_df = pl.concat(dfs)
combined_df.write_parquet(f"s3://<MY-BUCKET>/feast/criteo/transformed/features/year={end_date.year}/month={end_date.month:02d}/day={end_date.day:02d}/combined.parquet", use_pyarrow=True)

print('Feature transformation complete!')
ray.shutdown()