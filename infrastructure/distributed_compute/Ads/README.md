### Distributed Computing with Ray

ML models, such as Wide & Deep, DLRM(v1, v2, and now v3) and DCN(v1, v2, ...) were designed train on massive user-behavior datasets and serve them efficiently. Due to the system constraints, the concepts and the model architectures are relatively simple, but training and serving at a scale requires a distributed system, which is the focus this project.


#### Prereq commands
bash`
# create s3 bucket
aws s3api create-bucket \
    --bucket ml-paugre \
    --region us-west-2 \
    --create-bucket-configuration LocationConstraint=us-west-2

# upload local parquet files (preprocessed)
aws s3 cp train/train.parquet s3://ml-paugre/criteo/train.parquet
aws s3 cp val/val.parquet s3://ml-paugre/criteo/val.parquet

# upload teacher model weights
aws s3 cp last_checkpoint.pth s3://ml-paugre/ads/teacher/model.pth

# Ray creates IAM role for the head node. However, our function requires the worker node to have s3 permissions and the default role for the head node also needs to be modified to allow passRole to the worker nodes, lets custom roles and attach to both type of nodes
# Concepts summarized:
# Policy: List of AWS API permissions (S3 access).
# Role: The "Persona" that owns those policies.
# Trust Policy: Specifying services that can assume this role.
# Instance Profile: The container that delivers the Persona to the EC2 instance.

# Worker Role
# 1. create policy - s3 access
aws iam create-policy \
  --policy-name RayClusterS3ScopedPolicy \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Sid": "ReadTrainingData",
        "Effect": "Allow",
        "Action": [
          "s3:GetObject"
        ],
        "Resource": [
          "arn:aws:s3:::ml-paugre/criteo/*",
          "arn:aws:s3:::ml-paugre/ads/teacher/*"
        ]
      },
      {
        "Sid": "WriteCheckpoints",
        "Effect": "Allow",
        "Action": [
          "s3:PutObject",
          "s3:GetObject",
          "s3:DeleteObject",
          "s3:AbortMultipartUpload",
          "s3:ListMultipartUploadParts"
        ],
        "Resource": [
          "arn:aws:s3:::ml-paugre/ads/*"
        ]
      },
      {
        "Sid": "ListBucket",
        "Effect": "Allow",
        "Action": "s3:ListBucket",
        "Resource": "arn:aws:s3:::ml-paugre"
      }
    ]
  }'

# 2. create role - inside state that ec2 instances (e.g worker nodes) can assume this role
aws iam create-role \
    --role-name RayWorkerS3Role \
    --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": { "Service": "ec2.amazonaws.com" },
        "Action": "sts:AssumeRole"
      }
    ]
  }'

# 3. attach the s3 policy to the role
YOUR_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
aws iam attach-role-policy \
    --role-name RayWorkerS3Role \
    --policy-arn arn:aws:iam::$YOUR_ACCOUNT_ID:policy/RayClusterS3ScopedPolicy

# 4. create instance profile and link the role
aws iam create-instance-profile --instance-profile-name RayWorkerS3Profile
aws iam add-role-to-instance-profile \
    --instance-profile-name RayWorkerS3Profile \
    --role-name RayWorkerS3Role

# 5. attach the following instance profile to the worker node in cluster.yaml
echo "arn:aws:iam::$YOUR_ACCOUNT_ID:instance-profile/RayWorkerS3Profile"

# Head Role
# 1. Create the head node role with EC2 trust
aws iam create-role \
  --role-name RayHeadRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": { "Service": "ec2.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }]
  }'

# 2. Attach Ray's required EC2 autoscaler permissions
aws iam attach-role-policy \
  --role-name RayHeadRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# 3. Grant PassRole for the worker role
YOUR_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
aws iam put-role-policy \
  --role-name RayHeadRole \
  --policy-name AllowPassWorkerRole \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::'"$YOUR_ACCOUNT_ID"':role/RayWorkerS3Role"
    }]
  }'

# 4. Create the instance profile and attach the role
aws iam create-instance-profile --instance-profile-name RayHeadProfile
aws iam add-role-to-instance-profile \
  --instance-profile-name RayHeadProfile \
  --role-name RayHeadRole

# 5. attach the following instance profile to the head node in cluster.yaml
echo "arn:aws:iam::$YOUR_ACCOUNT_ID:instance-profile/RayHeadProfile"

`



bash`
# ray commands
ray up cluster.yaml
ray dashboard cluster.yaml --port=8265
ray job submit \
  --working-dir . \
  --address http://localhost:8265 \
  --no-wait \
  --runtime-env-json '{
    "env_vars": {
      "RAY_RUNTIME_ENV_HOOK": "ray._private.runtime_env.uv_runtime_env_hook.hook",
      "RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S": "800"
    },
    "excludes": [".venv", "__pycache__"]
  }' \
  -- uv run train.py
ray down cluster.yaml

# useful ray commands:

# To retrieve the IP address of the cluster head:
ray get-head-ip cluster.yaml

# To port-forward the cluster's Ray Dashboard to the local machine:
ray dashboard cluster.yaml --port=<dashboard-port>

# To submit a job to the cluster, port-forward the Ray Dashboard in another terminal and run:
ray job submit --address http://localhost:<dashboard-port> --working-dir . -- python my_script.py

# To ssh into the head node
ray attach cluster.yaml

# To monitor autoscaling (great for debugging):
ray exec cluster.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'

# Query the logs of the job:
ray job logs raysubmit_BvwAiNwjmwsWHnjM
# Query the status of the job:
ray job status raysubmit_BvwAiNwjmwsWHnjM
# Request the job to be stopped:
ray job stop raysubmit_BvwAiNwjmwsWHnjM
`

**Links**
1. [Configure Ray Cluster on AWS](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html)