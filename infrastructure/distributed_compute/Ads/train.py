from functools import partial

from Ads import AdsDataset, EarlyRanker, EmbeddingLayer, Solver, FinalRankerMMoE, FullRanker, EmbeddingLayer, DLRMTower
import torch
from ray.train.torch import TorchTrainer, prepare_model
from ray.train import FailureConfig, ScalingConfig, get_dataset_shard, RunConfig, get_checkpoint
from ray.data import read_parquet
from ray import init, shutdown
from s3torchconnector import S3Checkpoint
import os

init(address='auto')

# similar DLRM towers for both student and teacher architecture
projection_layer = 128
emb_layers = {
    "uid": int(2e6),
    "campaign": 401, # +1 for null at index 400
    "cat1": 10,
    "cat2": 70,
    "cat3": 1829,
    "cat4": 21,
    "cat5": 51,
    "cat6": 30,
    "cat7": 57196,
    "cat8": 11,
    "cat9": 30,
}

sparse_features = { 
    "user" : ["uid", "last_n_click_campaigns_1D", "last_n_conversion_campaigns_1D"],
    "ad"   : ["campaign", "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9"]
}


# ########### define student model ############ START
def get_student():
    # DLRM towers for features (users, ads) -> shared-bottom DNN -> multiple task heads for a multi-task Architecture

    bottom_mlp_layers = [256, 64]

    E = EmbeddingLayer(emb_layers=emb_layers, emb_dim=bottom_mlp_layers[-1])

    # define user and ad tower params
    base_params = {
        "bottom_mlp_layers" : bottom_mlp_layers,
        "projection_layer" : projection_layer,
        "embs" : E.embs,
        "dense_num" : 0
    }
    # copy to get independent dicts
    u_params  = {**base_params, "sparse_num": len(sparse_features["user"])}
    ad_params = {**base_params, "sparse_num": len(sparse_features["ad"])}

    shared_dims = [512, 256]

    # initialize a model
    # CTR, CVR, CTR distill, CVR distill
    task_dims = [[128, 1], [128, 1], [128, 1], [128, 1]]

    model = EarlyRanker(
        shared_dims = shared_dims,
        task_dims = task_dims,
        u_params = u_params,
        ad_params = ad_params
    )

    return model

# ########### define student model ############ END

# ########### define teacher model ############ START

def get_teacher():

    # for "simplicity", Stack DLRMTower with the mmoe ranker. This could potentially be used to share cached values across early-stage and final rankers.
    # e.g. user and ad towers are shared between the rankers, but the final ranker also includes additional context-based (streaming features) tower.

    bottom_mlp_layers = [512, 256, 64]

    E = EmbeddingLayer(emb_layers=emb_layers, emb_dim=bottom_mlp_layers[-1])

    # define user and ad tower params
    base_params = {
        "bottom_mlp_layers" : bottom_mlp_layers,
        "projection_layer" : projection_layer,
        "embs" : E.embs,
        "dense_num" : 0
    }
    # copy to get independent dicts
    u_params  = {**base_params, "sparse_num": len(sparse_features["user"])}
    ad_params = {**base_params, "sparse_num": len(sparse_features["ad"])}

    uTower = DLRMTower(**u_params)
    adTower = DLRMTower(**ad_params)


    mmoe_layers = 2
    expert_num = 4
    expert_dims = [256]
    # CTR, CVR
    final_task_dims = [[256, 128, 64, 1], [256, 128, 64, 1]]
    
    mmoe = FinalRankerMMoE(input_size=projection_layer*2, # input coming from user and ad towers
                                layers=mmoe_layers,
                                expert_num=expert_num,
                                expert_dims=expert_dims,
                                task_dims=final_task_dims,
                                top_k=4)

    model = FullRanker(
        towers={"user": uTower, "ad": adTower},
        final_ranker=mmoe
    )

    # state_dict = torch.load(f"s3://{bucket_path}", map_location=torch.device(device))
    # model.load_state_dict(state_dict)

    # load last checkpoint for a teacher model
    # checkpoint_dir = "./checkpoints/fullRanker"
    # checkpoint = torch.load(os.path.join(checkpoint_dir, f"last_checkpoint.pth"))
    # full_model.load_state_dict(checkpoint["model_state_dict"])


    return model

# ########### define teacher model ############ END

# ########### define optimizer ############ START
def get_optimizer(model, emb_lr, dense_lr):
    emb_params = []
    dense_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embs" in name:          # covers towers.user.embs.* and towers.ad.embs.*
            emb_params.append(p)
        else:
            dense_params.append(p)

    optimizer = torch.optim.Adagrad(
        [
            {"params": emb_params,   "lr": emb_lr},
            {"params": dense_params, "lr": dense_lr, "weight_decay": 1e-6},
        ],
        # starting value of the squared gradient accumulator in Adagrad.
        # if initial_accumulator_value=0 -> first step lr / sqrt(0 + ε) → very large update
        # Sparse embeddings (lr=0.05) — most embedding rows are never seen in the first few batches, so G=0 would cause huge updates on first encounter. 1e-8 softens this.
        # Dense MLPs (lr=1e-3) — less critical since dense params are updated every batch and G grows quickly anyway
        # initial_accumulator_value=1e-8,
    )
    return optimizer

# ########### define optimizer ############ END

# ########## Define Ray Task ########### Start
# train_func only runs on worker nodes
# this means that student and teacher models are loaded on worker nodes
def train_func(config):
    collate_with_args = partial(AdsDataset._collate_batch, user_v=config["user_v"], campaign_v=config["campaign_v"])

    # define model architectures
    early_model = get_student()
    full_model = get_teacher()

    # Ray prepares models for DDP or FSDP
    # MUST UNDERSTAND BETTER ....
    early_model = prepare_model(early_model)

    # load pretrained teacher weights from S3
    device = next(early_model.parameters()).device # model was loaded using ray, get which device was used to load the model
    teacher_checkpoint = S3Checkpoint(region=config["region"])
    s3_path = config["teacher_state_dict_s3"]
    with teacher_checkpoint.reader(f"s3://{s3_path}") as reader:
        state_dict = torch.load(reader,  map_location=device)

    # state_dict = torch.load(f"s3://{config["teacher_state_dict_s3"]}")
    # full_model stays as-is, just move to device
    full_model.load_state_dict(state_dict["model_state_dict"])
    full_model = full_model.to(device)
    
    # Get the specific shard for this worker
    train_data_shard = get_dataset_shard("train")


    # define optimizer for the model
    optimizer = get_optimizer(early_model, emb_lr=config["emb_lr"], dense_lr=config["dense_lr"])

    # in case if the worker node is reclaimed (spot instances go down often)
    # Access the latest reported checkpoint to resume from if one exists
    # make the process resilient from unexpected worker node failures
    student_checkpoint = get_checkpoint()
    start_epoch = 0
    
    if student_checkpoint:
        with student_checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))

            # prepare_model wraps the model structure and adds extra "module" prefix
            # if this is the case, load weights into model.module instead
            model_to_load = early_model.module if hasattr(early_model, "module") else early_model
            model_to_load.load_state_dict(checkpoint_dict["model_state_dict"])

            optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
            start_epoch = checkpoint_dict["epoch"] + 1




    solver = Solver(
        model = early_model,
        data = train_data_shard,
        optimizer = optimizer,
        device = device,
        epochs = config["epochs"],
        batch_size=config["batch_size"],
        collate_fn = collate_with_args,
        #checkpoint_dir = checkpoint_dir,
        distillation_weights = [3, 3],
        teacher = full_model,
        distillation_tasks = [2, 3],
        start_epoch=start_epoch
    )

    solver.train()

# ########## Define Ray Task ########### END

# basic idea between Ray.Data (read_parquet) and Ray.Train (TorchTrainer) tasks
# 1. read tasks execute on the read-labeled nodes
# 2. they produce Ray dataset blocks there
# 3. trainer workers (possibly on different nodes) ask for their shard batches
# 4. Ray transfers the needed blocks over the network / object store to those trainer workers

data_path = "s3://<BUCKET-NAME>/criteo/train.parquet"

# Create a Ray Dataset from S3 Parquet files
# starts tasks
train_ds = read_parquet(data_path,
                        # label that must match with the node tags to be scheduled
                        # use it to prevent scheduling on the head node
                        # ray_remote_args will pass this to the remote tasks
                        ray_remote_args={
                            "label_selector": {
                                "worker-label" : "True" # schedule read_parquet task only on worker nodes
                            },
                            #"resources": {"worker_node": 1} # do not use the worker_node here, worker_node is to limit one trainer per worker, we want to allow multiple read tasks running per worker
                        },
                        # This forces the reader to split the large files into smaller blocks
                        # Try increasing this number if you still get memory warnings
                        # by default it was creating 2 large blocks
                        override_num_blocks=20
                        )
num_workers = 2
use_gpu = False
config = { "epochs": 10,
          "batch_size": 1024,
          "emb_lr": 0.05, 
          "dense_lr": 1e-3, 
          "user_v": emb_layers["uid"], 
          "campaign_v": emb_layers["campaign"],
          "teacher_state_dict_s3": "<BUCKET-NAME>/ads/teacher/model.pth",
          "region": "us-west-2"
          }
# starts actors
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    datasets={"train": train_ds}, # Names the shard for access in the loop
    scaling_config=ScalingConfig( num_workers=num_workers, 
                                 use_gpu=use_gpu,
                                 # resources_per_worker={"CPU": 5}, # this is simply for scheduling and reserving cores, not an OS cap on the process
                                 label_selector={"worker-label":"True"}, # only schedule trainer actors on worker nodes
                                 resources_per_worker={"worker-resource": 1}, # each worker node has a capacity of 1, so 1 trainer for each worker node
                                ),
    run_config=RunConfig(
        # "By default, when you call report(), Ray Train synchronously pushes your checkpoint 
        # from checkpoint.path on local disk to checkpoint_dir_name on your **storage_path.**"
        storage_path="s3://<BUCKET-NAME>/ads/earlyRanker/",
        # ADD THIS: Automatically retry the job up to 3 times
        failure_config=FailureConfig(max_failures=3)
        )
)

result = trainer.fit()
print(result)
shutdown() # disconnect ray client, does not terminate the cluster 