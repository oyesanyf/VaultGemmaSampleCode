"""Deterministic unit and CLI tests for the VaultGemma DP data pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import LLM_Diffrential_Privacy as dp


def test_normalize_text_and_token_f1() -> None:
    assert dp._normalize_text("  Hypertension, Type-2! ") == "hypertension type 2"
    assert dp.compute_token_f1("Lisinopril 10mg", "lisinopril 10mg!") == 1.0
    assert dp.compute_token_f1("", "") == 1.0
    assert dp.compute_token_f1("Asthma", "") == 0.0
    partial = dp.compute_token_f1("Type 2 Diabetes", "Diabetes")
    assert 0.4 < partial < 0.6


def test_generate_and_dp_mask_phi_data() -> None:
    df = dp.create_original_phi_data(num_records=12)
    assert len(df) == 12
    for col in ("name", "ssn", "mrn", "phone", "email", "address", "condition", "medication"):
        assert col in df.columns

    masked = dp.apply_dp_to_data(df, epsilon=1.0)
    assert len(masked) == 12
    for sensitive in ("name", "phone", "email", "address", "ssn", "insurance_id", "mrn"):
        assert (masked[sensitive] == "[DP-MASKED]").all()
    assert masked["medical_notes"].str.contains("DP-PROCESSED").all()


def test_encrypt_list_and_clean_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    dp_path = dp.encrypt_data(num_records=10, dp_eps=1.0)
    assert os.path.isfile(dp_path)

    files = dp.list_available_data(reveal_dp=True, unmask=True)
    assert len(files) == 2

    dp.clean_all_data_and_models()
    assert dp.list_available_data() == []
