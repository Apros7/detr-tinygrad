from dataclasses import dataclass
import yaml
from typing import Optional

@dataclass
class TrainingConfig:
    lr: float
    lr_backbone: float 
    batch_size: int
    weight_decay: float
    epochs: int
    lr_drop: int
    clip_max_norm: float

@dataclass 
class BackboneConfig:
    type: str
    dilation: bool
    position_embedding: str

@dataclass
class TransformerConfig:
    enc_layers: int
    dec_layers: int
    dim_feedforward: int
    hidden_dim: int
    dropout: float
    nheads: int
    num_queries: int
    pre_norm: bool

@dataclass
class ModelConfig:
    frozen_weights: Optional[str]
    backbone: BackboneConfig
    transformer: TransformerConfig
    masks: bool
    aux_loss: bool

@dataclass
class MatcherConfig:
    set_cost_class: int
    set_cost_bbox: int
    set_cost_giou: int

@dataclass
class CoefficientConfig:
    mask: int
    dice: int
    bbox: int
    giou: int
    eos: float

@dataclass
class LossConfig:
    matcher: MatcherConfig
    coefficients: CoefficientConfig

@dataclass
class DatasetConfig:
    type: str
    coco_path: Optional[str]
    coco_panoptic_path: Optional[str]
    remove_difficult: bool

@dataclass
class RuntimeConfig:
    output_dir: str
    device: str
    seed: int
    resume: str
    start_epoch: int
    eval: bool
    num_workers: int

@dataclass
class DistributedConfig:
    world_size: int
    dist_url: str

@dataclass
class Config:
    training: TrainingConfig
    model: ModelConfig
    loss: LossConfig
    dataset: DatasetConfig
    runtime: RuntimeConfig
    distributed: DistributedConfig

def load_config(path: str) -> Config:
    with open(path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        
    # Convert nested dictionaries to dataclass instances
    training = TrainingConfig(**config_dict['training'])
    backbone = BackboneConfig(**config_dict['model']['backbone'])
    transformer = TransformerConfig(**config_dict['model']['transformer'])
    model = ModelConfig(**{**config_dict['model'], 'backbone': backbone, 'transformer': transformer})
    matcher = MatcherConfig(**config_dict['loss']['matcher'])
    coefficients = CoefficientConfig(**config_dict['loss']['coefficients'])
    loss = LossConfig(matcher=matcher, coefficients=coefficients)
    dataset = DatasetConfig(**config_dict['dataset'])
    runtime = RuntimeConfig(**config_dict['runtime'])
    distributed = DistributedConfig(**config_dict['distributed'])
    
    return Config(
        training=training,
        model=model,
        loss=loss,
        dataset=dataset,
        runtime=runtime,
        distributed=distributed
    )