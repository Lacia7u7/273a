import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData

from src.data.mappings import DEFAULT_MAPPINGS
from src.data.vocab import Vocab
from src.graph.builder import GraphBuilder
from src.utils.config import ColumnsConfig, GraphConfig, NodeToggle, EdgeToggle, FeatureDims, EdgeFeatureingConfig, PreprocessingConfig


def build_config():
    columns = ColumnsConfig(
        numeric=["num"],
        categorical_low_card=["gender"],
        icd_cols=["diag_1"],
        drug_cols=["drug_a"],
        hospital_col="hospital_id",
        specialty_col="medical_specialty",
        admission_type_col="admission_type_id",
        admission_source_col="admission_source_id",
        discharge_disposition_col="discharge_disposition_id",
    )
    graph = GraphConfig(
        node_types_enabled=NodeToggle(),
        edge_types_enabled=EdgeToggle(),
        edge_featureing=EdgeFeatureingConfig(),
        feature_dims=FeatureDims(encounter_tabular_proj_dim=64, embedding_dims={}),
        oov_nodes=True,
        artifacts_dir="./artifacts",
    )
    preprocessing = PreprocessingConfig(
        numeric_imputer="median",
        categorical_imputer="most_frequent",
        scaler="standard",
        categorical_handling="embedding",
        use_unknown_category=True,
        unknown_label="UNKNOWN",
        min_freq_for_category=1,
        truncate_icd_to_3_digits=True,
        map_icd_to_group=False,
        map_drug_to_class=False,
    )
    return columns, graph, preprocessing


def test_graph_builder_edges_and_reverse():
    df = pd.DataFrame(
        {
            "diag_1": ["250.1", "401"],
            "drug_a": ["Up", "No"],
            "hospital_id": ["1", "2"],
            "medical_specialty": ["cardiology", "neuro"],
            "admission_type_id": [1, 2],
            "admission_source_id": [7, 1],
            "discharge_disposition_id": [1, 2],
            "target": [1, 0],
            "gender": ["F", "M"],
        }
    )
    encounter_features = np.random.randn(len(df), 1)
    cat_indices = {"gender": np.array([0, 1])}
    vocabs = {
        "icd": Vocab({"250": 0, "401": 1, "UNKNOWN": 2}, ["250", "401", "UNKNOWN"], "UNKNOWN"),
        "drug": Vocab({"drug_a": 0, "UNKNOWN": 1}, ["UNKNOWN", "drug_a"], "UNKNOWN"),
        "hosp": Vocab({"1": 0, "2": 1}, ["1", "2"], "1"),
        "specialty": Vocab({"cardiology": 0, "neuro": 1}, ["cardiology", "neuro"], "cardiology"),
        "admission_source": Vocab({"1": 0, "7": 1}, ["1", "7"], "1"),
        "admission_type": Vocab({"1": 0, "2": 1}, ["1", "2"], "1"),
        "discharge_disposition": Vocab({"1": 0, "2": 1}, ["1", "2"], "1"),
    }
    columns, graph, preprocessing = build_config()
    builder = GraphBuilder(graph, preprocessing, columns, DEFAULT_MAPPINGS, vocabs)
    hetero = builder.build(df, encounter_features, cat_indices)
    assert isinstance(hetero, HeteroData)
    assert ("encounter", "has_icd", "icd") in hetero.edge_types
    assert ("icd", "rev_has_icd", "encounter") in hetero.edge_types
    assert hetero["encounter"].x.shape[0] == len(df)
