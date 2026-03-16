# Distributed Computing with Ray

ML models, such as Wide & Deep, DLRM(v1, v2, and now v3) and DCN(v1, v2, ...) were designed train on massive user-behavior datasets and serve them efficiently. Due to the system constraints, the concepts and the model architectures are relatively simple, but training and serving at a scale requires a distributed system, which is the focus this project.

## Configure a Ray Cluster on AWS

```bash
# create s3 bucket
aws s3api create-bucket \
    --bucket <BUCKET-NAME> \
    --region us-west-2 \
    --create-bucket-configuration LocationConstraint=us-west-2

# upload local parquet files (preprocessed)
aws s3 cp train/train.parquet s3:/<BUCKET-NAME>/criteo/train.parquet
aws s3 cp val/val.parquet s3:/<BUCKET-NAME>/criteo/val.parquet

# upload teacher model weights
aws s3 cp last_checkpoint.pth s3:/<BUCKET-NAME>/ads/teacher/model.pth

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
          "arn:aws:s3:::<BUCKET-NAME>/criteo/*",
          "arn:aws:s3:::<BUCKET-NAME>/ads/teacher/*"
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
          "arn:aws:s3:::<BUCKET-NAME>/ads/*"
        ]
      },
      {
        "Sid": "ListBucket",
        "Effect": "Allow",
        "Action": "s3:ListBucket",
        "Resource": "arn:aws:s3:::<BUCKET-NAME>"
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

```

#### Post Deployment
```bash
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
```
## Configure Ray on Kubernetes

While training and evaluation happens offline, model inference is online, and requires infrastructure that better supports production-level system orchestration and scaling, so in this section I explore KubeRay. Goal of this section is maximize utilization and performance of the serve instances to serve sparse MMoE ranker.

Ray.Serve notable features:
- max_queued_requests Maximum number of requests to this deployment that will be queued at each caller. Once the limit is reached, subsequent requests will raise a BackPressureError. Great to ensure the system stays stable. 
- max_ongoing_requests: Maximum number of queries that are sent to a replica of this deployment without receiving a response. For batched requests it will be good to at least make it the size of the batch.
- @serve.batch max_batch_size: requests are batched and executed asynchronously once the limit, max_batch_size, is reached. Greate to increase throughput and prevent unnecessary inference overheads.
- @serve.batch batch_wait_timeout_s: requests are batched and executed asynchronously once the limit, batch_wait_timeout_s, is reached. Set a timeout limit to meet SLA requirements.
- @serve.batch max_concurrent_batches: the maximum number of batches that can be executed concurrently.
- serve.multiplexed max_num_models_per_replica: the maximum number of models to be loaded and cache on each replica. Great for the expert model caching. Ideally it should work with batched requests, where multiple requests for a specific expert will be routed to replicas that contain the cached expert. Once the max_num_models_per_replica is reached, if the request is for an expert that is currently not loaded, then the least recently used (LRU) expert will be removed from memory. Goal is to prevent too many of such operations.

#### Identifying Optimization Goals
Each node can contain multiple replicas, but how much memory would be required per replica?
M_total ≈ (N_models × M_weights) + (N_batches × M_peak(B_max)) + M_system

- N_models: The max_num_models_per_replica in the @serve.multiplexed config.
- M_weights: Memory footprint of one model's weights.
- N_batches: max_concurrent_batches (default is 1).
- M_peak(B_max): Peak activation memory for a forward pass with max_batch_size.
- M_system : Overheads for Ray, CUDA context, and libraries.
​
Given the memory constraints of a node:
1. maximize max_num_models_per_replica for higher cache hits
2. find optimal max_batch_size and max_concurrent_batchrs for the highest throughput, given the SLA time constrain set by batch_wait_timeout_s and resource constraint of the node


Decision between CPU vs GPU will be decided based on whether the model execution spends more time on transfer(pick CPU) or compute(pick GPU)

#### Development Steps
1. Build serve config file by iteratively developing and testing the serve application on ray cluster
2. Configure Ray on Kubernetes
3. Deploy Ray Serve and the custom application(config file) on kuberntes cluster using KubeRay

#### Ray Serve Commands - VM-based
No Kubernetes involved — Ray is the orchestrator
```bash
ray up cluster-serve-vm.yaml

# serve the application locally
serve run serve_mmoe:app

# serve the application remotely
serve run \
  --address ray://54.218.239.33:10001 \
  --working-dir . \
  --runtime-env-json '{
    "pip": ["torch", "boto3", "datasets"],
    "env_vars": {
      "RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S": "800",
      "ADS_CHECKPOINT_PATH": "s3:/<BUCKET-NAME>/ads/finalRanker/ray_train_run-2026-03-14_06-36-25/checkpoint_2026-03-14_08-21-45.762228/model.pt",
      "ADS_EXPERT_WEIGHTS_S3_URI": "s3:/<BUCKET-NAME>/ads/finalRanker/split",
      "AWS_REGION": "us-west-2"
    },
    "excludes": [".venv", "__pycache__"]
  }' serve_mmoe:app

# fowrard port 8000 to local machine to make api calls to localhost:8000
ray attach cluster-serve-vm.yaml -p 8000
```

#### Migrating to Kubernetes

The hirarchy of the resources is different in K8s-based deployment, each physical node can hace multiple ray pods (equivalent to a ray Node in VM-based), and each pod can run multiple replicas:

K8s Node (physical machine)
├── worker pod A  →  Ray node A  →  budget: {cpu: 8, memory: 16Gi, "shared-resource": 1}
│   ├── Ranker replica 1  (actor process, consumes cpu:1, "shared-resource": 0.2)
│   ├── Ranker replica 2  (actor process, consumes cpu:1, "shared-resource": 0.2)
│   ├── Ranker replica 3  (actor process, consumes cpu:1, "shared-resource": 0.2)
│   ├── Ranker replica 4  (actor process, consumes cpu:1, "shared-resource": 0.2)
│   └── Ranker replica 5  (actor process, consumes cpu:1, "shared-resource": 0.2)
│       → "shared-resource" budget exhausted (5 × 0.2 = 1.0), CPU exhausted too
└── worker pod B  →  Ray node B  →  budget: {cpu: 4, ...}
    └── ... more replicas

```bash
# generate config file describing ray application configurations
serve build serve_mmoe:app -o serve-config.yaml

# create a managed kubernetes cluster (EKS)
brew install aws/tap/eksctl
brew install helm

# create EKS cluster
# creates kubeconfig at ~/.kube/config
eksctl create cluster \
  --name ray-serve-cluster \
  --region us-west-2 \
  --without-nodegroup

# Install KubeRay operator
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator


# add IAM OIDC provider to the cluster (required to add policies)
eksctl utils associate-iam-oidc-provider \
  --region us-west-2 \
  --cluster ray-serve-cluster \
  --approve

# Create a K8s service account linked to a policy
eksctl create iamserviceaccount \
  --name ray-s3-sa \
  --namespace default \
  --cluster ray-serve-cluster \
  --region us-west-2 \
  --attach-policy-arn arn:aws:iam::<ACCOUNT-ID>:policy/RayClusterS3ScopedPolicy \
  --approve

# node group for system pods
eksctl create nodegroup \
  --cluster ray-serve-cluster \
  --name ray-system-ng \
  --node-type t3.small \
  --region us-west-2 \
  --nodes 1 --nodes-min 1 --nodes-max 1 \
  --node-labels role=system

# add permissions to enable cluster autoscaler
# https://docs.aws.amazon.com/eks/latest/best-practices/cas.html
# Create policy with full permissions per AWS best practices
aws iam create-policy \
  --policy-name ClusterAutoscalerPolicy \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "autoscaling:DescribeAutoScalingGroups",
          "autoscaling:DescribeAutoScalingInstances",
          "autoscaling:DescribeLaunchConfigurations",
          "autoscaling:DescribeScalingActivities",
          "autoscaling:DescribeTags",
          "ec2:DescribeImages",
          "ec2:DescribeInstanceTypes",
          "ec2:DescribeLaunchTemplateVersions",
          "ec2:GetInstanceTypesFromInstanceRequirements",
          "eks:DescribeNodegroup"
        ],
        "Resource": ["*"]
      },
      {
        "Effect": "Allow",
        "Action": [
          "autoscaling:SetDesiredCapacity",
          "autoscaling:TerminateInstanceInAutoScalingGroup"
        ],
        "Resource": ["*"]
      }
    ]
  }'

# link the policy to a service account
eksctl create iamserviceaccount \
  --name cluster-autoscaler \
  --namespace kube-system \
  --cluster ray-serve-cluster \
  --region us-west-2 \
  --attach-policy-arn arn:aws:iam::<ACCOUNT-ID>:policy/ClusterAutoscalerPolicy \
  --approve \
  --override-existing-serviceaccounts


# Install cluster (node) autoscaler. Ray autoscaler (pod) is enabled separately in rayService.yaml.
helm repo add autoscaler https://kubernetes.github.io/autoscaler
helm install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system \
  --set autoDiscovery.clusterName=ray-serve-cluster \
  --set awsRegion=us-west-2 \
  --set rbac.serviceAccount.create=false \
  --set rbac.serviceAccount.name=cluster-autoscaler

# Head node group — tainted so only the head pod tolerates it
eksctl create nodegroup \
  --cluster ray-serve-cluster \
  --name ray-head-ng \
  --node-type m5.large \
  --region us-west-2 \
  --nodes 1 --nodes-min 1 --nodes-max 1 \
  --node-labels role=ray-head

# workers have no toleration
kubectl taint nodes -l role=ray-head role=ray-head:NoSchedule

# Expert worker node group
eksctl create nodegroup \
  --cluster ray-serve-cluster \
  --name ray-expert-ng \
  --node-type m5.xlarge \
  --region us-west-2 \
  --nodes 1 --nodes-min 1 --nodes-max 4 \
  --node-labels role=ray-expert-worker

# Shared worker node group
eksctl create nodegroup \
  --cluster ray-serve-cluster \
  --name ray-shared-ng \
  --node-type m5.xlarge \
  --region us-west-2 \
  --nodes 1 --nodes-min 1 --nodes-max 2 \
  --node-labels role=ray-shared-worker

# working dir needs to be included in the application, which could be added through s3
zip -r serve_code.zip serve_mmoe.py Ads.py tricks
# alternatively, create a custom ray image and copy the same files
aws s3 cp serve_code.zip s3:/<BUCKET-NAME>/ads/serve_code/serve_code.zip # will be linked to the application inside rayService.yaml

# finally, deploy the application resources
kubectl apply -f rayService.yaml

# port forwarding to localhost
kubectl get svc # get name of the rayService
kubectl port-forward svc/ads-ranker-xxxx-head-svc 8265:8265 # for dashboard
kubectl port-forward svc/ads-ranker-serve-svc 8000:8000 # for the scoring service

# track custom serve logs
kubectl exec ads-ranker-xxxx-head-xxxx -c ray-head -- bash -c 'tail -f /tmp/ray/session_latest/logs/serve/*'
kubectl exec ads-ranker-xxxx-expert-workers-worker-xxxx -c ray-worker -- bash -c 'tail -f /tmp/ray/session_latest/logs/serve/*'
kubectl exec ads-ranker-xxxx-shared-workers-worker-xxxx -c ray-worker -- bash -c 'tail -f /tmp/ray/session_latest/logs/serve/*'
 
# tear down the resources
kubectl delete -f rayService.yaml

# Delete IAM service accounts (IRSA roles created by eksctl)
eksctl delete iamserviceaccount --name ray-s3-sa --namespace default --cluster ray-serve-cluster --region us-west-2
eksctl delete iamserviceaccount --name cluster-autoscaler --namespace kube-system --cluster ray-serve-cluster --region us-west-2

# Delete manually created IAM policies
aws iam delete-policy --policy-arn arn:aws:iam::<ACCOUNT-ID>:policy/ClusterAutoscalerPolicy
aws iam delete-policy --policy-arn arn:aws:iam::<ACCOUNT-ID>:policy/RayClusterS3ScopedPolicy

eksctl delete cluster \
  --name ray-serve-cluster \
  --region us-west-2

```

#### Additional Notes
- The Ray autoscaler adjusts the number of Ray nodes in a Ray cluster. On Kubernetes, each Ray node is run as a Kubernetes Pod. Thus in the context of Kubernetes, the Ray autoscaler scales Ray Pod quantities. In this sense, the Ray autoscaler plays a role similar to that of the Kubernetes Horizontal Pod Autoscaler (HPA), but The HPA determines scale based on physical usage metrics like CPU and memory. By contrast, the Ray autoscaler uses the logical resources expressed in task and actor annotations.
- Each Ray cluster is managed by its own Ray autoscaler process, running as a sidecar container in the Ray head Pod
- The Ray Autoscaler and the Kubernetes Cluster Autoscaler complement each other. After the Ray autoscaler decides to create a Ray Pod, the Kubernetes Cluster Autoscaler can provision a Kubernetes node so that the Pod can be placed
- **It is recommended to configure your RayCluster so that only one Ray Pod fits per Kubernetes node.**


#### Links
1. [Configure Ray Cluster on AWS](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html)
2. [Ray Serve Development Worlflow](https://docs.ray.io/en/latest/serve/advanced-guides/dev-workflow.html#serve-dev-workflow)