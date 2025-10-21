"""Configuration schema validated by Pydantic.

The JSON object embedded inside ``main.ipynb`` is validated against this schema.
The schema mirrors the specification from the project instructions.  The module
exposes :func:`load_config` which accepts the raw JSON string, parses it and
returns a ``Config`` instance with convenient helpers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, root_validator, validator


class IdentifierCols(BaseModel):
    encounter_id: str
    patient_id: str


class TargetConfig(BaseModel):
    name: str
    positive_values: List[str]


class FilterConfig(BaseModel):
    min_los: int
    max_los: int
    exclude_discharge_to_ids: List[int]
    first_encounter_per_patient: bool

    @validator("max_los")
    def _check_los(cls, v: int, values: Dict[str, int]) -> int:
        min_los = values.get("min_los", 0)
        if v < min_los:
            raise ValueError("max_los must be >= min_los")
        return v


class ColumnsConfig(BaseModel):
    numeric: List[str]
    categorical_low_card: List[str]
    icd_cols: List[str]
    drug_cols: List[str]
    hospital_col: str
    specialty_col: str
    admission_type_col: str
    admission_source_col: str
    discharge_disposition_col: str


class PreprocessingConfig(BaseModel):
    numeric_imputer: Literal["median", "mean"]
    categorical_imputer: Literal["most_frequent", "constant"]
    scaler: Literal["standard", "minmax"]
    categorical_handling: Literal["embedding", "onehot"]
    use_unknown_category: bool
    unknown_label: str = "UNKNOWN"
    min_freq_for_category: int = 1
    truncate_icd_to_3_digits: bool
    map_icd_to_group: bool
    map_drug_to_class: bool


class SplitStrategy(BaseModel):
    strategy: Literal["group_k_fold", "group_shuffle", "group_time"]
    group_by: Literal["patient", "hospital"]
    n_splits: int = 5
    seed: int
    stratify_by_target: bool = False

    @validator("n_splits")
    def _n_splits(cls, v: int) -> int:
        if v < 2:
            raise ValueError("n_splits must be >= 2")
        return v


class DataConfig(BaseModel):
    csv_path: str
    id_mapping_path: Optional[str]
    identifier_cols: IdentifierCols
    target: TargetConfig
    filters: FilterConfig
    columns: ColumnsConfig
    preprocessing: PreprocessingConfig
    splits: SplitStrategy

    @validator("csv_path", "id_mapping_path", pre=True, always=True)
    def _expand_paths(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return str(Path(v).expanduser())


class NodeToggle(BaseModel):
    encounter: bool = True
    icd: bool = True
    icd_group: bool = True
    drug: bool = True
    drug_class: bool = True
    hosp: bool = True
    specialty: bool = True
    admission_source: bool = True
    admission_type: bool = True
    discharge_disposition: bool = True


class EdgeToggle(BaseModel):
    encounter__has_icd__icd: bool = True
    icd__is_a__icd_group: bool = True
    encounter__has_drug__drug: bool = True
    drug__belongs_to__drug_class: bool = True
    encounter__at_hospital__hosp: bool = True
    encounter__has_specialty__specialty: bool = True
    encounter__has_admission_source__admission_source: bool = True
    encounter__has_admission_type__admission_type: bool = True
    encounter__has_discharge__discharge_disposition: bool = True
    reverse_edges: bool = True


class EdgeFeatureConfig(BaseModel):
    relation_subtypes_by_status: bool = False
    edge_attr_status: bool = False


class EdgeFeatureingConfig(BaseModel):
    has_drug: EdgeFeatureConfig = Field(default_factory=EdgeFeatureConfig)


class FeatureDims(BaseModel):
    encounter_tabular_proj_dim: int
    embedding_dims: Dict[str, int]


class GraphConfig(BaseModel):
    node_types_enabled: NodeToggle
    edge_types_enabled: EdgeToggle
    edge_featureing: EdgeFeatureingConfig
    feature_dims: FeatureDims
    oov_nodes: bool = True
    artifacts_dir: str = "./artifacts"

    @validator("artifacts_dir")
    def _expand_dir(cls, v: str) -> str:
        return str(Path(v).expanduser())


class LossConfig(BaseModel):
    type: Literal["bce_with_logits"] = "bce_with_logits"
    pos_weight: Optional[Literal["auto"]] = None


class OptimizerConfig(BaseModel):
    name: Literal["Adam", "AdamW", "SGD"]
    lr: float
    weight_decay: float = 0.0


class SchedulerConfig(BaseModel):
    name: Literal["cosine", "none", "step"]
    warmup_epochs: Optional[int] = 0


class BatchingConfig(BaseModel):
    batch_size_encounters: int
    fanouts_per_layer_by_relation: Dict[str, List[int]]

    @validator("fanouts_per_layer_by_relation")
    def _ensure_monotonic(cls, v: Dict[str, List[int]]) -> Dict[str, List[int]]:
        for rel, fanouts in v.items():
            if not all(n >= 0 for n in fanouts):
                raise ValueError(f"Negative fanout in relation {rel}")
        return v


class TrainConfig(BaseModel):
    epochs: int
    early_stopping_patience: int
    early_stopping_metric: str
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    batching: BatchingConfig
    val_every: int
    gradient_clip_norm: float
    deterministic: bool
    seed: int
    tensorboard_logdir: str
    save_best_on: str

    @validator("tensorboard_logdir")
    def _expand_tb(cls, v: str) -> str:
        return str(Path(v).expanduser())


class EvaluationPlots(BaseModel):
    roc: bool = True
    pr: bool = True
    calibration: bool = True
    confusion: bool = True
    decision_curves: bool = True


class ThresholdTuning(BaseModel):
    optimize_for: Literal["f1_pos", "precision_pos", "recall_pos", "balanced_accuracy"]
    grid: List[float]
    calibration: Literal["isotonic", "platt", "none"]

    @validator("grid")
    def _grid_between_zero_one(cls, v: List[float]) -> List[float]:
        for threshold in v:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("Threshold grid values must be in [0, 1]")
        return v


class EvaluationConfig(BaseModel):
    metrics_primary: List[str]
    metrics_secondary: List[str]
    threshold_tuning: ThresholdTuning
    plots: EvaluationPlots
    subgroup_metrics: List[str]


class InferenceConfig(BaseModel):
    inductive: bool
    build_star_subgraph_on_the_fly: bool
    oov_handling: Literal["UNKNOWN", "drop"]
    batch_predict_csv_path: Optional[str]
    output_predictions_path: str

    @validator("output_predictions_path")
    def _expand_output(cls, v: str) -> str:
        return str(Path(v).expanduser())


class BaselineMLPConfig(BaseModel):
    enabled: bool = True
    hidden_dims: List[int]
    dropout: float
    epochs: int
    batch_size: int
    optimizer: OptimizerConfig


class BaselineConfig(BaseModel):
    tabular_mlp: BaselineMLPConfig
    xgboost: Dict[str, bool] = Field(default_factory=lambda: {"enabled": False})


class ModelConfig(BaseModel):
    arch: Literal["HGT", "RGCN", "GraphSAGE"] = "HGT"
    hidden_dim: int
    num_layers: int
    dropout: float
    heads: int
    rgcn_bases: int
    act: Literal["relu", "gelu", "elu"]
    norm: Literal["layer", "batch", "none"]
    use_edge_attr_for_drug_status: bool
    loss: LossConfig


class Config(BaseModel):
    data: DataConfig
    graph: GraphConfig
    model: ModelConfig
    train: TrainConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig
    baseline: BaselineConfig

    @property
    def artifacts_dir(self) -> Path:
        return Path(self.graph.artifacts_dir)

    def ensure_artifact_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        Path(self.train.tensorboard_logdir).mkdir(parents=True, exist_ok=True)


def load_config(raw_json: str) -> Config:
    """Parse JSON text into a validated :class:`Config` instance."""
    payload = json.loads(raw_json)
    config = Config.parse_obj(payload)
    config.ensure_artifact_dirs()
    return config


__all__ = ["Config", "load_config"]
