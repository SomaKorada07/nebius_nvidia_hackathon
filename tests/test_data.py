"""Checks on the standardized splits written by src/prepare_data.py."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA_DIR = ROOT / CONFIG["standardized_dir"]
PROCESSED_DIR = ROOT / CONFIG["processed_dir"]
SPLITS = ("train", "validation", "test")


@pytest.fixture(scope="module")
def candidate_sets() -> dict[str, pd.DataFrame]:
    paths = {split: PROCESSED_DIR / f"{split}.parquet" for split in SPLITS}
    if not all(path.exists() for path in paths.values()):
        pytest.skip("run src/build_candidates.py first")
    return {split: pd.read_parquet(path) for split, path in paths.items()}


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    paths = {split: DATA_DIR / f"{split}.parquet" for split in SPLITS}
    if not all(path.exists() for path in paths.values()):
        pytest.skip("run src/prepare_data.py first")
    return {split: pd.read_parquet(path) for split, path in paths.items()}


@pytest.fixture(scope="module")
def labels() -> dict[str, dict]:
    return json.loads((DATA_DIR / "labels.json").read_text())


def test_in_scope_rows_have_known_labels(splits, labels):
    for frame in splits.values():
        in_scope = frame[~frame.is_oos]
        assert in_scope.label_id.notna().all()
        assert in_scope.label_id.isin(list(labels)).all()
        assert frame[frame.is_oos].label_id.isna().all()


def test_example_ids_do_not_overlap_across_splits(splits):
    seen = set()
    for frame in splits.values():
        ids = set(frame.example_id)
        assert len(ids) == len(frame)
        assert not ids & seen
        seen |= ids


def test_unseen_labels_are_held_out_of_train(splits, labels):
    unseen = {label_id for label_id, info in labels.items() if info["group"] == "unseen"}
    assert unseen
    assert not set(splits["train"].label_id.dropna()) & unseen
    assert unseen <= set(splits["test"].label_id.dropna())


def test_candidate_sets_have_consistent_targets(candidate_sets):
    for frame in candidate_sets.values():
        assert frame.num_candidates.isin(CONFIG["candidate_sizes"]).all()
        for ids, target, label, is_none in zip(frame.candidate_ids, frame.target_id, frame.label_id, frame.is_none):
            ids = list(ids)
            assert len(set(ids)) == len(ids)
            if is_none:
                assert target is None and label not in ids
            else:
                assert target is not None and target == label and target in ids


def test_train_sets_use_only_seen_labels(candidate_sets, labels):
    unseen = {label_id for label_id, info in labels.items() if info["group"] == "unseen"}
    used = {label_id for ids in candidate_sets["train"].candidate_ids for label_id in ids}
    assert not used & unseen


def test_every_label_has_a_description(labels):
    descriptions = json.loads((ROOT / CONFIG["label_descriptions"]).read_text())
    assert set(descriptions) == set(labels)
    assert all(text.strip() for text in descriptions.values())


def test_v3_sets_keep_simulated_and_real_unseen_labels_apart(labels):
    paths = [PROCESSED_DIR / name for name in ("v3_split.json", "v3_stage1.parquet", "v3_stage2.parquet")]
    if not all(path.exists() for path in paths):
        pytest.skip("run src/build_candidates.py first")
    split = json.loads(paths[0].read_text())
    stage1, stage2 = pd.read_parquet(paths[1]), pd.read_parquet(paths[2])
    unseen = {label_id for label_id, info in labels.items() if info["group"] == "unseen"}
    pseudo, held_out = set(split["pseudo_unseen_labels"]), set(split["held_out_example_ids"])
    assert pseudo and not pseudo & set(split["base_labels"])
    assert not {label for ids in stage1.candidate_ids for label in ids} & (pseudo | unseen)
    assert not set(stage1.example_id) & held_out
    assert not {label for ids in stage2.candidate_ids for label in ids} & unseen
    assert set(stage2[stage2.none_group == "seen"].example_id) <= held_out
    assert all(pseudo & set(ids) for ids in stage2.candidate_ids)
    assert stage2.groupby(["none_group", "is_none"]).size().nunique() == 1
