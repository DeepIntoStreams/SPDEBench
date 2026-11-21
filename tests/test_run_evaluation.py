import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import run_evaluation


def save_predictions(path, reference, prediction):
    torch.save({"u_real": reference, "u_pred": prediction}, path)
    return path


def comparison_rows(output, files, metrics):
    lines = output.splitlines()
    header = lines[len(files)]
    assert header.split() == ["Model", *metrics]
    assert lines[len(files) + 1] == "-" * len(header)
    rows = [line.split() for line in lines[len(files) + 2 :]]
    assert [row[0] for row in rows] == [path.name for path in files]
    assert all(len(row) == len(metrics) + 1 for row in rows)
    return [row[1:] for row in rows]


def run_comparison(monkeypatch, capsys, files, metrics):
    monkeypatch.setattr(
        sys, "argv", ["run_evaluation.py", "--files", *map(str, files), "--metrics", *metrics]
    )
    run_evaluation.main()
    return comparison_rows(capsys.readouterr().out, files, metrics)


def pooled_rmse(reference, prediction):
    residuals = prediction.double() - reference.double()
    return math.sqrt(residuals.square().mean().item())


@pytest.mark.parametrize(
    ("samples", "field_shape", "dtype"),
    [
        (20, (2, 3), torch.float32),
        (40, (2, 3), torch.float64),
        (27, (2, 3, 4), torch.float32),
        (21, (2, 3), torch.float64),
        (43, (2, 2, 3, 4), torch.float32),
    ],
)
def test_rmse_matches_whole_tensor(tmp_path, monkeypatch, capsys, samples, field_shape, dtype):
    shape = (samples, *field_shape)
    values = torch.arange(math.prod(shape), dtype=dtype).reshape(shape)
    reference = 1 + values % 11
    prediction = reference + (values % 13 - 7) / 4
    prediction[-1] -= 6
    stored_prediction = prediction.unsqueeze(1) if samples == 21 else prediction
    path = save_predictions(tmp_path / "model.pt", reference, stored_prediction)

    [[value]] = run_comparison(monkeypatch, capsys, [path], ["RMSE"])

    assert float(value) == pytest.approx(pooled_rmse(reference, prediction), abs=1e-6)


@pytest.mark.parametrize("error_index", [0, 7, 16, 20])
def test_rmse_invariant_to_error_position(tmp_path, monkeypatch, capsys, error_index):
    reference = torch.ones(21, 2, 3)
    prediction = reference.clone()
    prediction[error_index, 1, 2] -= 10
    path = save_predictions(tmp_path / "model.pt", reference, prediction)

    [[first, repeated]] = run_comparison(monkeypatch, capsys, [path], ["RMSE", "RMSE"])

    assert first == repeated
    assert float(first) == pytest.approx(pooled_rmse(reference, prediction), abs=1e-6)


def test_rmse_counts_existing_broadcast_elements(tmp_path, monkeypatch, capsys):
    reference = torch.ones(21, 1, 2, 3)
    prediction = reference.clone()
    prediction[-1] += 10
    path = save_predictions(tmp_path / "model.pt", reference, prediction)

    [[value]] = run_comparison(monkeypatch, capsys, [path], ["RMSE"])

    # Keep the CLI's prediction-channel squeeze and resulting per-batch broadcasting.
    residuals = torch.cat([
        (xp.squeeze(1) - xr).flatten()
        for xr, xp in zip(reference.split(20), prediction.split(20))
    ])
    assert residuals.numel() == (20 * 20 + 1) * 6
    expected = math.sqrt(residuals.double().square().mean().item())
    assert float(value) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("metrics", [[], ["--metrics", "RMSE"]])
def test_comparison_cli_multiple_files(tmp_path, metrics):
    reference = torch.ones(21, 2, 3)
    paths = []
    for name, err_idx in [("early", 0), ("late", 20), ("perfect", None)]:
        prediction = reference.clone()
        if err_idx is not None:
            prediction[err_idx] += 10
        paths.append(save_predictions(tmp_path / f"{name}.pt", reference, prediction))
    originals = [path.read_bytes() for path in paths]
    script = Path(run_evaluation.__file__).resolve()

    result = subprocess.run(
        [sys.executable, "-B", str(script), "--files", *map(str, paths), *metrics],
        check=True,
        capture_output=True,
        text=True,
    )

    assert [path.read_bytes() for path in paths] == originals
    columns = ["RMSE"] if metrics else ["LpLoss", "RMSE"]
    rows = comparison_rows(result.stdout, paths, columns)
    if not metrics:
        assert [row[0] for row in rows] == ["0.250000", "5.000000", "0.000000"]
    expected = f"{math.sqrt(100 / 21):.6f}"
    assert [row[-1] for row in rows] == [expected, expected, "0.000000"]


def test_comparison_keeps_absolute_l2(tmp_path, monkeypatch, capsys):
    reference = torch.ones(21, 2, 3, dtype=torch.float64)
    prediction = reference.clone()
    prediction[-1] += 6
    path = save_predictions(tmp_path / "model.pt", reference, prediction)

    [[value]] = run_comparison(monkeypatch, capsys, [path], ["LpLossAbs"])

    assert value == "2.000000"


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_rmse_preserves_nonfinite_result(tmp_path, monkeypatch, capsys, value):
    reference = torch.ones(21, 2, 3)
    prediction = reference.clone()
    prediction[-1] = value
    path = save_predictions(tmp_path / "model.pt", reference, prediction)

    [[score]] = run_comparison(monkeypatch, capsys, [path], ["RMSE"])

    assert score == str(value)


def test_rmse_preserves_zero_element_result(tmp_path, monkeypatch, capsys):
    empty_fields = torch.empty(21, 0, 3)
    path = save_predictions(tmp_path / "model.pt", empty_fields, empty_fields)

    [[score]] = run_comparison(monkeypatch, capsys, [path], ["RMSE"])

    assert score == "nan"
