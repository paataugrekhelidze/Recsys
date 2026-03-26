from datetime import timedelta
from feast import Entity, FeatureService, FeatureView, Field, RedshiftSource
from feast.types import Int64, String

user = Entity(name="user", join_keys=["uid"])

# Points to the output of the Ray ETL job
# transformed_source = FileSource(
#     name="criteo_transformed",
#     path="s3://<MY-BUCKET>/feast/criteo/transformed/features/",
#     timestamp_field="event_timestamp",
#     file_format=ParquetFormat(),
# )
# criteo_data_source = RedshiftSource(
#     name="criteo_processed",
#     query="SELECT * FROM criteo.features",
#     # The event timestamp is used for point-in-time joins and for ensuring only
#     # features within the TTL are returned
#     timestamp_field="event_timestamp"
# )
criteo_data_source = RedshiftSource(
    name="criteo_processed",
    schema="public",
    table="temp_feast_entities",
    # The event timestamp is used for point-in-time joins and for ensuring only
    # features within the TTL are returned
    timestamp_field="event_timestamp"
)

impression_transformed_view = FeatureView(
    name="impression_transformed_view",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="campaign", dtype=Int64),
        Field(name="cat1", dtype=Int64),
        Field(name="cat2", dtype=Int64),
        Field(name="cat3", dtype=Int64),
        Field(name="cat4", dtype=Int64),
        Field(name="cat5", dtype=Int64),
        Field(name="cat6", dtype=Int64),
        Field(name="cat7", dtype=Int64),
        Field(name="cat8", dtype=Int64),
        Field(name="cat9", dtype=Int64),
        Field(name="last_5_clicks", dtype=String),
        Field(name="last_5_conversions", dtype=String),
    ],
    online=True,
    source=criteo_data_source
)

user_activity = FeatureService(
    name="user_activity", features=[impression_transformed_view]
)