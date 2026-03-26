import ray
import ray.data.datasource.partitioning as partitioning
import polars as pl
from datetime import datetime, timedelta
import os
from ray.data.datasource.partitioning import PathPartitionFilter
import joblib

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


class WindowAggregator:
    def __init__(self, window_days, end_date):
        import joblib
        # Loaded once per worker, not serialized over the network
        self.V = joblib.load("ordinal_encoders.joblib") 
        self.window_days = window_days
        self.end_date = end_date
        self.columns_to_encode = [
            'uid', 'campaign', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9'
        ]

        # Categorize columns by their cardinality
        self.low_card_cols = ['campaign', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat8', 'cat9']
        self.high_card_cols = ['cat7']

        # workaround because directly applying scikit to polars datafram is really slow 
        self.join_maps = {}
        self.replace_maps = {}
        for col in self.columns_to_encode:
            encoder = self.V[col]
            # Handle both Scikit-Learn OrdinalEncoder and LabelEncoder types
            classes = encoder.categories_[0] if hasattr(encoder, 'categories_') else encoder.classes_

            if col in self.high_card_cols:
                # Create a 2-column DataFrame: [Original Value, Encoded Value]
                self.join_maps[col] = pl.DataFrame({
                    col: classes,
                    f"{col}_encoded": [int(i) for i in range(len(classes))]
                })
            else:
                self.replace_maps[col] = {val: int(i) for i, val in enumerate(classes)}

    def __call__(self, user_df):
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

        # Since grouped by uid, all rows in this block have the same ID.
        current_uid = user_df['uid'][0]
        encoded_uid = self.replace_maps['uid'].get(current_uid, self.V['uid'].unknown_value)
        user_df = user_df.with_columns(pl.lit(encoded_uid).alias('uid'))

        # transform campaign column only to avoid many cols for last_5_click and last_5_conversion columns
        
        # slow scikit-learn trasform
        # user_df = user_df.with_columns([
        #     pl.Series("campaign", self.V["campaign"].transform(user_df["campaign"].to_numpy().reshape(-1, 1)).flatten())
        # ])

        # # faster, left join
        # user_df = user_df.join(self.mappings["campaign"], on="campaign", how="left")
        # # fill Nulls with the default unknown value, drop extra column
        # user_df = user_df.with_columns(
        #     pl.col("campaign_encoded").fill_null(self.V["campaign"].unknown_value).alias("campaign")
        # ).drop("campaign_encoded")

        # fastes on low cardinality, Replace 'campaign' (675 values) instantly
        user_df = user_df.with_columns(
            pl.col("campaign").replace(self.replace_maps["campaign"], default=self.V["campaign"].unknown_value)
        )

        # For each (uid, timestamp)
        for idx in range(user_df.height):
            current_time = timestamps[idx]
            # Get uid data within the appropriate range
            cutoff_time = current_time - pl.duration(days=self.window_days)
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
        user_df = user_df.filter(
            (pl.col('event_timestamp') >= self.end_date) & (pl.col('event_timestamp') < self.end_date + timedelta(days=1))
        )
        
        # transform other columns on smaller data
        
        # High Cardinality Join (cat7)
        user_df = user_df.join(self.join_maps["cat7"], on="cat7", how="left")

        # Low Cardinality Batch Replace (cat1-6, 8-9)
        replace_exprs = [
            pl.col(col).replace(self.replace_maps[col], default=self.V[col].unknown_value)
            for col in self.low_card_cols if col != 'campaign'
        ]

        user_df = user_df.with_columns(
            replace_exprs + [
                pl.col("cat7_encoded").fill_null(self.V["cat7"].unknown_value).alias("cat7")
            ]
        ).drop("cat7_encoded")


        return user_df.to_arrow()
    

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
    WindowAggregator,
    fn_constructor_kwargs={
        "window_days": window_days,
        "end_date": end_date
    }, 
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