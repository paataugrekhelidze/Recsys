#### Configure KubeRay for Feast's Ray Compute Engine

```bash
# create a managed kubernetes cluster (EKS)
brew install aws/tap/eksctl
brew install helm

# create EKS cluster
# creates kubeconfig at ~/.kube/config
eksctl create cluster \
  --name feast-ray-cluster \
  --region us-west-2 \
  --without-nodegroup

# Install KubeRay operator
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator


# add IAM OIDC provider to the cluster (required to add policies)
eksctl utils associate-iam-oidc-provider \
  --region us-west-2 \
  --cluster feast-ray-cluster \
  --approve

# create policy to access s3 resources
aws iam create-policy \
  --policy-name FeastRayClusterS3ScopedPolicy \
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
          "arn:aws:s3:::<MY-BUCKET>/criteo/*"
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
          "arn:aws:s3:::<MY-BUCKET>/feast/*"
        ]
      },
      {
        "Sid": "ListBucket",
        "Effect": "Allow",
        "Action": "s3:ListBucket",
        "Resource": "arn:aws:s3:::<MY-BUCKET>"
      }
    ]
  }'

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create a K8s service account linked to a policy
eksctl create iamserviceaccount \
  --name ray-s3-sa \
  --namespace default \
  --cluster feast-ray-cluster \
  --region us-west-2 \
  --attach-policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/FeastRayClusterS3ScopedPolicy \
  --approve

# node group for system pods
eksctl create nodegroup \
  --cluster feast-ray-cluster \
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
  --cluster feast-ray-cluster \
  --region us-west-2 \
  --attach-policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/ClusterAutoscalerPolicy \
  --approve \
  --override-existing-serviceaccounts

# Install cluster (node) autoscaler. Ray autoscaler (pod) is enabled separately in rayService.yaml.
helm repo add autoscaler https://kubernetes.github.io/autoscaler
helm install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system \
  --set autoDiscovery.clusterName=feast-ray-cluster \
  --set awsRegion=us-west-2 \
  --set rbac.serviceAccount.create=false \
  --set rbac.serviceAccount.name=cluster-autoscaler

# Head node group — tainted so only the head pod tolerates it
eksctl create nodegroup \
  --cluster feast-ray-cluster \
  --name ray-head-ng \
  --node-type m5.large \
  --region us-west-2 \
  --nodes 1 --nodes-min 1 --nodes-max 1 \
  --node-labels role=ray-head

# workers have no toleration
kubectl taint nodes -l role=ray-head role=ray-head:NoSchedule

# worker node group
eksctl create nodegroup \
  --cluster feast-ray-cluster \
  --name ray-ng \
  --node-type m5.xlarge \
  --region us-west-2 \
  --nodes 2 --nodes-min 2 --nodes-max 4 \
  --node-labels role=ray-worker


# Deploy Ray cluster
kubectl apply -f rayCluster.yaml


# Forward ray cluster address to localhost
kubectl port-forward svc/ray-compute-engine-head-svc 10001:10001
# Access Ray dashboard locally
kubectl port-forward svc/ray-compute-engine-head-svc 8265:8265

# port-forwarding for port 8265 is required to expose the Ray Dashboard
# ETL job for a single day, scheduler should submit new jobs daily (or more granular)
ray job submit \
--address http://localhost:8265 \
--working-dir . \
--runtime-env-json '{
    "pip": ["polars", "joblib", "scikit-learn"],
    "env_vars": {
      "WINDOW_DAYS": "2",
      "END_DATE": "2026-03-20"
    },
    "excludes": ["rayCluster.yaml", "README.md"]
  }' \
-- python ray_job.py
```