# This is an example feature definition file

from datetime import timedelta

import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    PushSource,
    RedshiftSource,
    RequestSource,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64

# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
user = Entity(name="user", join_keys=["uid"])
# ad = Entity(name="ad", join_keys=["campaign"])

# Defines a data source from which feature values can be retrieved. Sources are queried when building training
# datasets or materializing features into an online store.
criteo_data_source = RedshiftSource(
    # The Redshift table where features can be found
    # table="criteo.unprocessed",
    name="criteo_unprocessed",
    query="SELECT * FROM criteo.unprocessed",
    # The event timestamp is used for point-in-time joins and for ensuring only
    # features within the TTL are returned
    timestamp_field="event_timestamp"
    # The (optional) created timestamp is used to ensure there are no duplicate
    # feature rows in the offline store or when building training datasets
    # created_timestamp_column="created",
    # Database to redshift source.
    # database="feast",
)
# criteo_data_source = RedshiftSource(
#     name="criteo_unprocessed",
#     query="""
#         SELECT 
#             uid, campaign, conversion, conversion_timestamp, conversion_id,
#             attribution, click, click_pos, click_nb, cost, cpo, time_since_last_click,
#             cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9,
#             CAST(event_timestamp AS TIMESTAMP) AS event_timestamp
#         FROM criteo.unprocessed
#     """,
#     timestamp_field="event_timestamp"
# )


impresion_fv = FeatureView(
    # The unique name of this feature view. Two feature views in a single
    # project cannot have the same name
    name="impression_unprocessed_view",
    entities=[user],
    ttl=timedelta(days=365), # user impressions per session can be seconds apart, keep ttl small
    # The list of features defined below act as a schema to both define features
    # for both materialization of features into a store, and are used as references
    # during retrieval for building a training dataset or serving features
    schema=[
        Field(name="uid", dtype=Int64),
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
    ],
    online=True,
    source=criteo_data_source,
    # Tags are user defined key/value pairs that are attached to each
    # feature view
    # tags={"team": "driver_performance"},
)

# ad_fv = FeatureView(
#     # The unique name of this feature view. Two feature views in a single
#     # project cannot have the same name
#     name="ad_unprocessed_view",
#     entities=[ad],
#     ttl=timedelta(days=1), # ad impressions per session can be seconds apart, keep ttl small
#     # ["campaign", "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9"]
#     schema=[
#         Field(name="campaign", dtype=Int64),
#         Field(name="cat1", dtype=Int64),
#         Field(name="cat2", dtype=Int64),
#         Field(name="cat3", dtype=Int64),
#         Field(name="cat4", dtype=Int64),
#         Field(name="cat5", dtype=Int64),
#         Field(name="cat6", dtype=Int64),
#         Field(name="cat7", dtype=Int64),
#         Field(name="cat8", dtype=Int64),
#         Field(name="cat9", dtype=Int64),
#     ],
#     online=True,
#     source=criteo_data_source,
#     # Tags are user defined key/value pairs that are attached to each
#     # feature view
#     # tags={"team": "driver_performance"},
# )



# Define a request data source which encodes features / information only
# available at request time (e.g. part of the user initiated HTTP request)
# input_request = RequestSource(
#     name="vals_to_add",
#     schema=[
#         Field(name="val_to_add", dtype=Int64),
#         Field(name="val_to_add_2", dtype=Int64),
#     ],
# )


# Define an on demand feature view which can generate new features based on
# existing feature views and RequestSource features
# @on_demand_feature_view(
#     sources=[driver_stats_fv, input_request],
#     schema=[
#         Field(name="conv_rate_plus_val1", dtype=Float64),
#         Field(name="conv_rate_plus_val2", dtype=Float64),
#     ],
# )
# def transformed_conv_rate(inputs: pd.DataFrame) -> pd.DataFrame:
#     df = pd.DataFrame()
#     df["conv_rate_plus_val1"] = inputs["conv_rate"] + inputs["val_to_add"]
#     df["conv_rate_plus_val2"] = inputs["conv_rate"] + inputs["val_to_add_2"]
#     return df


# This groups features into a model version
# user = FeatureService(
#     name="driver_activity_v1",
#     features=[
#         driver_stats_fv[["conv_rate"]],  # Sub-selects a feature from a feature view
#         transformed_conv_rate,  # Selects all features from the feature view
#     ],
# )
user_activity = FeatureService(
    name="user_activity", features=[impresion_fv]
)

# # Defines a way to push data (to be available offline, online or both) into Feast.
# driver_stats_push_source = PushSource(
#     name="driver_stats_push_source",
#     batch_source=criteo_data_source,
# )

# # Defines a slightly modified version of the feature view from above, where the source
# # has been changed to the push source. This allows fresh features to be directly pushed
# # to the online store for this feature view.
# driver_stats_fresh_fv = FeatureView(
#     name="driver_hourly_stats_fresh",
#     entities=[driver],
#     ttl=timedelta(days=1),
#     schema=[
#         Field(name="conv_rate", dtype=Float32),
#         Field(name="acc_rate", dtype=Float32),
#         Field(name="avg_daily_trips", dtype=Int64),
#     ],
#     online=True,
#     source=driver_stats_push_source,  # Changed from above
#     tags={"team": "driver_performance"},
# )


# # Define an on demand feature view which can generate new features based on
# # existing feature views and RequestSource features
# @on_demand_feature_view(
#     sources=[driver_stats_fresh_fv, input_request],  # relies on fresh version of FV
#     schema=[
#         Field(name="conv_rate_plus_val1", dtype=Float64),
#         Field(name="conv_rate_plus_val2", dtype=Float64),
#     ],
# )
# def transformed_conv_rate_fresh(inputs: pd.DataFrame) -> pd.DataFrame:
#     df = pd.DataFrame()
#     df["conv_rate_plus_val1"] = inputs["conv_rate"] + inputs["val_to_add"]
#     df["conv_rate_plus_val2"] = inputs["conv_rate"] + inputs["val_to_add_2"]
#     return df


# driver_activity_v3 = FeatureService(
#     name="driver_activity_v3",
#     features=[driver_stats_fresh_fv, transformed_conv_rate_fresh],
# )





# transformations needed
# categorical data -> numeric values (create a map table for efficient joins instead of apply python OrdinalEncoder, unknown values will be dealt with using hashing, null value will have a special index)
# historical n window features (e.g last an campaigns clicked or converted)