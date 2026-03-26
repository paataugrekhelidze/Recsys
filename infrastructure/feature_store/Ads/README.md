## Feature Store with Feast
Goal of the repository is to introduce an interface that connects raw data from previous ads related exercises to training and serving pipelines. Feast is a feature store that offers many capabilities: feature repository for versioning and collaboration, consistency between training and inference features, schema validation, point in time correctness (prevents data leakage), plugins to efficiently generate training data (spark or ray engine), a method to "materialize" latest feature values to an online store (e.g. DynamoDB) for a quick retrieval during inference, and more. 

#### Pipeline
1. Data source coming from s3 (e.g. parquet files created in preprocess.ipynb) - phase 1
2. ETL job (ideally created via scheduler) using ray distributed computing. Raw data in s3 is used to perform sequential window aggregation and mapping of ordinalEncoder. All transformations should ideally be pushed to offline etl jobs to improve online performance. Transformed data is written to another path in s3.
3. Redshift spectrum (via Glue crawler) reads data directly from s3. Feast uses redshift as offline_store to perform point-in-time joins to generate training data. - phase 1
4. Airflow schedule (train): starts training data generation process (using get_historical_features() in Feast) -> starts model training process (ray.Train DDP or FSDP) -> uploads updated model parameters to mlflow (model repository, save tha pointer to the featureService to keep the features consistent with the model) - phase 2
5. Airflow schedule (serve): CI job that loads updated parameters online with zero downtime (ray serve) - phase 2
6. Airflow schedule (online features): Feast materialize/materialize-incremental will generate latest feature values and push data to online-store (DynamoDB) for quick retrieval. Compute engine (Ray) can be used to generate onDemandFeature transformations (avoid if possible) - phase 3
7. Pushing streaming data (kafka + Flink) to offline (training) and online store (cache for inference) as they arrive (e.g. clicks and conversions) - phase 3
8. Prediction service that combines data from online store (streaming and batch) and on demand (at request-time) on a batch of data and returns scores across multiple tasks (CTR, CVR) - phase 2-3

#### Configure Glue data catalog
Auto-detects partitions and registers the schema. Redshift spectrum needs the schema from Glue data catalog to read data from s3
```bash
# Create a Glue database
aws glue create-database --database-input '{"Name": "criteo_db"}' --region us-west-2

# create role and permission to access s3 for glue crawler
# Create the role for redshift spectrum
aws iam create-role \
  --role-name GlueRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "glue.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy \
  --role-name GlueRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

aws iam attach-role-policy \
  --role-name GlueRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
GLUE_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/GlueRole"

# Create a crawler to scan your S3 parquet partition structure
# InheritFromTable — new partitions inherit their schema from the parent table definition. If the table schema changes, all partitions update to match.
aws glue create-crawler \
  --name criteo-crawler \
  --role $GLUE_ROLE_ARN \
  --database-name criteo_db \
  --region us-west-2 \
  --targets '{"S3Targets": [{"Path": "s3://<MY-BUCKET>/feast/criteo/transformed/features/"}]}' \
  --configuration '{"Version":1.0,"CrawlerOutput":{"Partitions":{"AddOrUpdateBehavior":"InheritFromTable"}}}'

# should partition by day (31)
aws glue start-crawler --name criteo-crawler --region us-west-2

# # Delete Stale Glue table
# aws glue delete-table --database-name criteo_db --name unprocessed --region us-west-2
```

#### Configure Redshift (using aws cli directly)
```bash
# it is assumed that files to s3 have already been uploaded and the Glue data catalog has been created

# Create the role for redshift spectrum
aws iam create-role \
  --role-name RedshiftSpectrumRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "redshift.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach S3 read access (scope to your bucket in production)
aws iam attach-role-policy \
  --role-name RedshiftSpectrumRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Attach Glue access (Spectrum uses Glue as the metastore)
aws iam attach-role-policy \
  --role-name RedshiftSpectrumRole \
  --policy-arn arn:aws:iam::aws:policy/AWSGlueConsoleFullAccess

# need to write temporary files in s3_staging_location
aws iam attach-role-policy \
  --role-name RedshiftSpectrumRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REDSHIFT_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/RedshiftSpectrumRole"

aws redshift-serverless create-namespace \
  --namespace-name feast-namespace \
  --admin-username admin \
  --admin-user-password Feast123 \
  --db-name dev \
  --region us-west-2 \
  --iam-roles $REDSHIFT_ROLE_ARN \
  --default-iam-role-arn $REDSHIFT_ROLE_ARN

aws redshift-serverless create-workgroup \
  --workgroup-name feast-workgroup \
  --namespace-name feast-namespace \
  --region us-west-2 \
  --base-capacity 8

# get amazon resource name for the workgroup
WORKGROUP_ARN=$(aws redshift-serverless get-workgroup \
  --workgroup-name feast-workgroup \
  --region us-west-2 \
  --query 'workgroup.workgroupArn' \
  --output text)

# Cap at 20 RPU-hours per day (~$7.20/day max at $0.36/RPU-hour)
# sorry I'm cheap
aws redshift-serverless create-usage-limit \
  --resource-arn $WORKGROUP_ARN \
  --usage-type serverless-compute \
  --amount 20 \
  --period daily \
  --breach-action deactivate \
  --region us-west-2

```

#### Create mapping from redshift (serverless) to external catalog (Glue)
```bash


ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REDSHIFT_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/RedshiftSpectrumRole"

aws redshift-data execute-statement \
    --workgroup-name feast-workgroup \
    --region us-west-2 \
    --database dev \
    --sql "CREATE EXTERNAL SCHEMA IF NOT EXISTS criteo FROM DATA CATALOG DATABASE 'criteo_db' \
    IAM_ROLE '${REDSHIFT_ROLE_ARN}' \
    REGION 'us-west-2';"

# need to create a local table otherwise feast get_historical_features does not work with spectrum
aws redshift-data execute-statement \
    --workgroup-name feast-workgroup \
    --region us-west-2 \
    --database dev \
    --sql "DROP TABLE IF EXISTS public.temp_feast_entities; \
           CREATE TABLE public.temp_feast_entities AS \
           SELECT * \
           FROM criteo.features \
           WHERE (year='2026' AND month='03' AND day='20');"

# drop if needed
# aws redshift-data execute-statement \
#     --workgroup-name feast-workgroup \
#     --region us-west-2 \
#     --database dev \
#     --sql "DROP SCHEMA IF EXISTS criteo;"

# make sure that schema was successfully created
aws redshift-data describe-statement --id <STATEMENT-ID> --region us-west-2

# execute query statement
aws redshift-data execute-statement \
    --workgroup-name feast-workgroup \
    --region us-west-2 \
    --database dev \
    --sql "SELECT * from criteo.features LIMIT 1;"


# print the query result
aws redshift-data get-statement-result --id <STATEMENT-ID> --region us-west-2
```


#### Configure Feast
```bash
# install Feast
uv add 'feast[aws]==0.61'

# create a feature repository, which will contain the state and definition of the feature store
# further manual modifications in feature_store.yaml
# inputs (my example)
# region: us-west-2
# cluster ID: feast-workgroup
# Database Name: feast
# Username: admin
# s3_staging_location: s3://<BUCKET-NAME>/feast/criteo/staging
# IAM: arn:aws:iam::<ACCOUNT_ID>:role/RedshiftSpectrumRole
feast init -t aws

# apply config, creates registry.pb, centralized definitions for features views, entities, data sources...
# make sure to port-forward ray client to localhost 10001 for ray batch engine to connect to the ray cluster, see assets/ray/README.md
feast apply

# push data from offline store (redshift) to online store (DynamoDB)
feast materialize 2026-03-20T00:00:00 2026-03-25T00:00:00
```

#### Links
1. [What Are Feature Stores: The Backbone of Scalable ML Systems](https://medium.com/write-a-catalyst/what-are-feature-stores-the-backbone-of-scalable-ml-systems-4fd9bf13080f)
2. [Feast Documentation](https://docs.feast.dev)