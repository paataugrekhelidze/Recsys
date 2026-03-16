from __future__ import annotations

import asyncio
import io
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import boto3
import torch
import torch.nn as nn
from ray import serve
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request

from Ads import DLRMTower, EmbeddingLayer, FullRanker, MultiplexedFinalRankerMMoE
import logging
logger = logging.getLogger("ray.serve") # routes into Ray's serve log files


DEFAULT_EMB_LAYERS = {
    "uid": int(2e6),
    "campaign": 401,
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

DEFAULT_SPARSE_FEATURES = {
    "user": ["uid", "last_n_click_campaigns_1D", "last_n_conversion_campaigns_1D"],
    "ad": ["campaign", "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9"],
}


@dataclass(slots=True)
class TeacherModelConfig:
    projection_layer: int = 128
    bottom_mlp_layers: list[int] = field(default_factory=lambda: [512, 256, 64])
    emb_layers: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_EMB_LAYERS))
    sparse_features: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_SPARSE_FEATURES))
    expert_num: int = 4
    expert_dims: list[int] = field(default_factory=lambda: [256])
    gate_dims: list[int] = field(default_factory=list)
    gate_dropout: float | None = None
    task_dims: list[list[int]] = field(default_factory=lambda: [[256, 128, 64, 1], [256, 128, 64, 1]])
    top_k: int = 4

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None = None) -> "TeacherModelConfig":
        config = {} if config is None else dict(config)
        return cls(**config)


def _build_expert_module(input_size: int, expert_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_size
    for idx, dim in enumerate(expert_dims):
        layers.append(nn.Linear(previous_dim, dim, bias=True))
        if idx < len(expert_dims) - 1:
            layers.append(nn.ReLU())
        previous_dim = dim
    return nn.Sequential(*layers)


def _load_torch_checkpoint(checkpoint_path: str, aws_region: str | None = None) -> Any:
    if checkpoint_path.startswith("s3://"):
        bucket, key = _parse_s3_uri(checkpoint_path)
        s3_client = boto3.client("s3", region_name=aws_region)
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return torch.load(io.BytesIO(response["Body"].read()), map_location="cpu")

    return torch.load(checkpoint_path, map_location="cpu")


def _extract_model_state(checkpoint_path: str, aws_region: str | None = None) -> dict[str, torch.Tensor]:
    checkpoint = _load_torch_checkpoint(checkpoint_path, aws_region=aws_region)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected S3 URI, got {uri}")
    bucket_and_key = uri[len("s3://"):]
    bucket, _, key_prefix = bucket_and_key.partition("/")
    if not bucket or not key_prefix:
        raise ValueError(f"Expected S3 URI with bucket and key prefix, got {uri}")
    return bucket, key_prefix.rstrip("/")


def _build_expert_artifact_key(weights_path: str, expert_id: int) -> str:
    return f"{weights_path}/expert_{expert_id}.pt"
def _build_tower_artifact_key(weights_path: str, entity: Literal["user", "ad"]) -> str:
    return f"{weights_path}/{entity}_tower.pt"
def _build_task_head_key(weights_path: str) -> str:
    # should not be named shared since task heads are independent
    # they just share memory, unlike the experts
    return f"{weights_path}/final_ranker_shared.pt"


def build_teacher_ranker(
    config: TeacherModelConfig,
    *,
    load_local_experts: bool,
) -> FullRanker:
    embedding_layer = EmbeddingLayer(
        emb_layers=config.emb_layers,
        emb_dim=config.bottom_mlp_layers[-1],
    )

    base_params = {
        "bottom_mlp_layers": config.bottom_mlp_layers,
        "projection_layer": config.projection_layer,
        "embs": embedding_layer.embs,
        "dense_num": 0,
    }
    user_params = {**base_params, "sparse_num": len(config.sparse_features["user"])}
    ad_params = {**base_params, "sparse_num": len(config.sparse_features["ad"])}

    user_tower = DLRMTower(**user_params)
    ad_tower = DLRMTower(**ad_params)

    final_ranker = MultiplexedFinalRankerMMoE(
        input_size=config.projection_layer * len(("user", "ad")),
        expert_num=config.expert_num,
        expert_dims=config.expert_dims,
        gate_dims=config.gate_dims,
        gate_dropout=config.gate_dropout,
        task_dims=config.task_dims,
        top_k=config.top_k,
        load_local_experts=load_local_experts,
    )

    return FullRanker(
        towers={"user": user_tower, "ad": ad_tower},
        final_ranker=final_ranker,
    )


def load_ranker_from_checkpoint(
    checkpoint_path: str,
    config: TeacherModelConfig,
    *,
    load_local_experts: bool,
    aws_region: str | None = None
) -> FullRanker:
    model = build_teacher_ranker(config, load_local_experts=load_local_experts)

    s3_client = boto3.client("s3", region_name=aws_region)
    bucket, weights_path = _parse_s3_uri(checkpoint_path)
    # extract towers and task heads that will share memory
    # user tower
    response = s3_client.get_object(
        Bucket=bucket,
        Key=_build_tower_artifact_key(weights_path, "user"),
    )
    state_dict = torch.load(io.BytesIO(response["Body"].read()), map_location="cpu")
    model.towers["user"].load_state_dict(state_dict)
    
    # ad tower
    response = s3_client.get_object(
        Bucket=bucket,
        Key=_build_tower_artifact_key(weights_path, "ad"),
    )
    state_dict = torch.load(io.BytesIO(response["Body"].read()), map_location="cpu")
    model.towers["ad"].load_state_dict(state_dict)
    
    # task heads
    response = s3_client.get_object(
        Bucket=bucket,
        Key=_build_task_head_key(weights_path),
    )
    state_dict = torch.load(io.BytesIO(response["Body"].read()), map_location="cpu")
    model.final_ranker.load_state_dict(state_dict, strict=False)

    return model


@serve.deployment(max_ongoing_requests=512, 
                  num_replicas=4, 
                  max_replicas_per_node=1,
                  ray_actor_options={ "resources": {"expert-resource": 1}})
class ExpertShard:
    def __init__(
        self,
        *,
        input_size: int,
        expert_dims: list[int],
        expert_weights_s3_uri: str | None = None,
        aws_region: str | None = None,
        device: str = "cpu",
    ):
        expert_weights_s3_uri = expert_weights_s3_uri or os.environ["ADS_CHECKPOINT_PATH"]
        aws_region = aws_region or os.environ.get("AWS_REGION")
        self.input_size = input_size
        self.expert_dims = expert_dims
        self.device = torch.device(device)
        self.expert_bucket, self.expert_weights_path = _parse_s3_uri(expert_weights_s3_uri)
        self.s3_client = boto3.client("s3", region_name=aws_region)

    @serve.multiplexed(max_num_models_per_replica=1)
    async def get_expert(self, model_id: str) -> nn.Sequential:
        expert_id = int(model_id)
        # @serve.multiplexed calls this only on a cache miss (LRU eviction or first load)
        logger.info("Cache MISS — loading expert %d from S3", expert_id)
        # only load a specific expert
        response = self.s3_client.get_object(
            Bucket=self.expert_bucket,
            Key=_build_expert_artifact_key(self.expert_weights_path, expert_id),
        )
        state_dict = torch.load(io.BytesIO(response["Body"].read()), map_location="cpu")

        expert = _build_expert_module(self.input_size, self.expert_dims)
        expert.load_state_dict(state_dict)
        expert.to(self.device)
        expert.eval()
        logger.info("Expert %d loaded and cached", expert_id)
        return expert

    @serve.batch(max_batch_size=128, batch_wait_timeout_s=0.01)
    async def batched_forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        model_id = serve.get_multiplexed_model_id()
        expert = await self.get_expert(model_id)

        batch_tensors = [torch.as_tensor(tensor, dtype=torch.float32, device=self.device) for tensor in inputs]
        batch_sizes = [tensor.shape[0] for tensor in batch_tensors]
        batch = torch.cat(batch_tensors, dim=0)

        with torch.inference_mode():
            outputs = expert(batch).to("cpu")

        return list(outputs.split(batch_sizes, dim=0))

    async def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return await self.batched_forward(inputs)

# NOTE: shared-resource is about model components sharing the memory, which is completely different from the ML concept called shared-bottom layer (we use experts instead of shared-bottom).
@serve.deployment(max_ongoing_requests=32, num_replicas=5, ray_actor_options={ "num_cpus": 0.5, "resources": {"shared-resource": 0.2}})
class Ranker:
    def __init__(
        self,
        expert_handle: DeploymentHandle,
        *,
        checkpoint_path: str | None = None,
        model_config: dict[str, Any] | None = None,
        aws_region: str | None = None,
        device: str = "cpu",
    ):
        checkpoint_path = checkpoint_path or os.environ["ADS_CHECKPOINT_PATH"]
        aws_region = aws_region or os.environ.get("AWS_REGION")
        self.device = torch.device(device)
        self.model_config = TeacherModelConfig.from_dict(model_config)
        # only loads tower and task head components
        logger.info("loading DLRM towers and task heads from S3")
        self.model = load_ranker_from_checkpoint(
            checkpoint_path,
            self.model_config,
            load_local_experts=False,
            aws_region=aws_region
        )
        self.model.to(self.device)
        self.model.eval()
        self.expert_handle = expert_handle

    def _tensorize_tower_input(self, tower_payload: dict[str, Any]) -> dict[str, Any]:
        dense_payload = tower_payload.get("dense")
        dense = None
        if dense_payload is not None:
            dense = torch.as_tensor(dense_payload, dtype=torch.float32, device=self.device)

        emb_indices = {
            key: torch.as_tensor(value, dtype=torch.long, device=self.device)
            for key, value in tower_payload["emb_indices"].items()
        }
        emb_offsets = {
            key: torch.as_tensor(value, dtype=torch.long, device=self.device)
            for key, value in tower_payload.get("emb_offsets", {}).items()
        }
        return {
            "dense": dense,
            "emb_indices": emb_indices,
            "emb_offsets": emb_offsets,
        }

    def _tensorize_batch(self, batch_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            tower_name: self._tensorize_tower_input(tower_payload)
            for tower_name, tower_payload in batch_payload.items()
        }

    async def predict(self, batch_payload: dict[str, Any]) -> dict[str, Any]:
        batch = self._tensorize_batch(batch_payload)

        with torch.inference_mode():
            tower_outputs = [tower(**batch[name]) for name, tower in self.model.towers.items()]
            combined = torch.cat(tower_outputs, dim=1)
            routing_plan = self.model.final_ranker.plan_routing(combined)
            expert_requests = self.model.final_ranker.prepare_expert_inputs(combined, routing_plan)

        expert_futures = {
            expert_id: self.expert_handle.options(multiplexed_model_id=str(expert_id)).remote(
                request["inputs"].detach().to("cpu")
            )
            for expert_id, request in expert_requests.items()
        }
        expert_outputs = dict(zip(expert_futures.keys(), await asyncio.gather(*expert_futures.values())))

        with torch.inference_mode():
            task_logits = self.model.final_ranker.reduce_expert_outputs(
                routing_plan,
                expert_outputs,
            )

        return {
            "logits": [tensor.squeeze(-1).tolist() for tensor in task_logits],
            "topk_indices": [indices.tolist() for indices in routing_plan["topk_indices"]],
            "topk_weights": [weights.tolist() for weights in routing_plan["topk_weights"]],
        }

    async def __call__(self, request: Request) -> dict[str, Any]:
        payload = await request.json()
        return await self.predict(payload)


def build_app(
    *,
    checkpoint_path: str | None = None,
    expert_weights_s3_uri: str | None = None,
    model_config: dict[str, Any] | None = None,
    aws_region: str | None = None,
    ranker_device: str = "cpu",
    expert_device: str = "cpu",
):
    config = TeacherModelConfig.from_dict(model_config)

    expert = ExpertShard.bind(
        input_size=config.projection_layer * len(("user", "ad")),
        expert_dims=config.expert_dims,
        expert_weights_s3_uri=expert_weights_s3_uri,  # None → resolved in __init__ on actor
        aws_region=aws_region,
        device=expert_device,
    )
    return Ranker.bind(
        expert,
        checkpoint_path=checkpoint_path,  # None → resolved in __init__ on actor
        model_config=asdict(config),
        aws_region=aws_region,
        device=ranker_device,
    )


# env vars (ADS_CHECKPOINT_PATH, AWS_REGION) are NOT read here
# because this module is imported locally by `serve run` before the runtime-env is applied.
# They are resolved inside ExpertShard.__init__ and Ranker.__init__, which run on cluster actors
# where the env_vars from --runtime-env-json are already injected.
app = build_app(
    model_config={
        "bottom_mlp_layers": [512, 256, 64],
        "projection_layer": 128,
        "expert_dims": [256],
        "expert_num": 4,
        "top_k": 2,
        "task_dims": [[256, 128, 64, 1], [256, 128, 64, 1]]
    }
)