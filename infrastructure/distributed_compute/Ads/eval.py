
import ray
from s3torchconnector import S3Checkpoint
import torch
from ray.data import ActorPoolStrategy
from ray import init, shutdown
from Ads import AdsDataset, EarlyRanker, EmbeddingLayer, Solver

init(address='auto')

class TorchPredictor:
    def __init__(self, s3_path: str, region: str, device: str, user_v: int, campaign_v: int):
        self.device = device
        self.user_v = user_v
        self.campaign_v = campaign_v
        # Load model on the actor
        self.model = self.get_model()
        
        checkpoint = S3Checkpoint(region=region)
        
        with checkpoint.reader(f"s3://{s3_path}") as reader:
            state_dict = torch.load(reader,  map_location=device)
        
        print(state_dict.keys())
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.model = self.model.to(device)
        self.model.eval()

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        # creates batch feature tensors from the preprocessed data
        batch = AdsDataset._collate_batch(batch, user_v=self.user_v, campaign_v=self.campaign_v)
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # further processing to define emb_indices and emb_offsets, and return x, y
        x, targets = Solver._build_x(batch)

        # Batch inference
        with torch.no_grad():
            outputs = self.model(x=x)
        # CTR, CVR, CTR distill, CVR distill
        return {
            "ctr_prob": torch.sigmoid(outputs[0]).squeeze(1).cpu().numpy(),
            "cvr_prob": torch.sigmoid(outputs[1]).squeeze(1).cpu().numpy(),
            "ctr_distill_prob": torch.sigmoid(outputs[2]).squeeze(1).cpu().numpy(),
            "cvr_distill_prob": torch.sigmoid(outputs[3]).squeeze(1).cpu().numpy(),
            "ctr_target": targets[0].squeeze(1).cpu().numpy(),
            "cvr_target": targets[1].squeeze(1).cpu().numpy(),

        }
        
    @staticmethod
    def get_model():
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


# Ray will move the block from A to B, then move the transformed block from B to C if needed.

# creates a dataset plan (lazy)
# during execution, Reader tasks start producing blocks on nodes A (whatever matches the selector, e.g. worker nodes)
data_path = "s3://<BUCKET-NAME>/criteo/val.parquet"
ds = ray.data.read_parquet(data_path, 
                           ray_remote_args={
                                "label_selector": {
                                    "worker-label":"True" # schedule only on worker nodes
                                },
                            })

# starts 4 long-lived actors, which instantiate TorchPredictor per actor only once (model is not reloaded many times) on nodes B
predictions = ds.map_batches(
    TorchPredictor,
    fn_constructor_kwargs={
        "s3_path": "<BUCKET-NAME>/ads/earlyRanker/ray_train_run-2026-03-11_05-36-31/checkpoint_2026-03-11_06-27-25.881891/model.pt",
        "region": "us-west-2",
        "device": "cpu",
        "user_v": int(2e6),
        "campaign_v": 401,
    },
    compute=ActorPoolStrategy(size=4), # Number of workers
    batch_format="numpy",
    label_selector={"worker-label": "True"} # schedule only on worker nodes
)

# function that actually triggers the execution
# creates writer tasks that write blocks from nodes C
predictions.write_parquet("s3://<BUCKET-NAME>/ads/earlyRanker/predictions", 
                          ray_remote_args={
                            "label_selector": {
                                "worker-label":"True" # schedule only on worker nodes
                            },
                        })

shutdown()