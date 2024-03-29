import unittest
from typing import Mapping

import numpy as np
import torch

from ss_recon.config.config import get_cfg
from ss_recon.evaluation.recon_evaluation import ReconEvaluator


class MockReconEvaluator(ReconEvaluator):
    def __init__(
        self,
        dataset_name="mock_dataset",
        group_by_scan: bool = False,
        metrics=None,
        flush_period: int = None,
    ):
        if metrics is None:
            metrics = ["nrmse", "psnr", "ssim (Wang)", "nrmse_scan", "psnr_scan"]
        cfg = get_cfg()
        super().__init__(
            dataset_name,
            cfg,
            output_dir=None,
            group_by_scan=group_by_scan,
            metrics=metrics,
            flush_period=flush_period,
            skip_rescale=True,
        )
        self._output_dir = None
        self._cpu_device = torch.device("cpu")
        self._normalizer = None

        self.reset()


def _build_mock_data(scan_slices=(2, 2, 2), batch_size: int = 2, pred=None, target=None):
    if isinstance(scan_slices, int):
        scan_slices = (scan_slices,)

    if pred is None:
        pred = torch.rand(sum(scan_slices), 20, 20, 1, 2)
    else:
        assert len(pred) == sum(scan_slices)

    if target is None:
        target = torch.rand(sum(scan_slices), 20, 20, 1, 2)
    else:
        assert len(target) == sum(scan_slices)

    scan_ids = [chr(ord("A") + i) for i in range(len(scan_slices))]
    metadata = [
        {"scan_id": scan_id, "slice_id": slice_id}
        for s_idx, scan_id in enumerate(scan_ids)
        for slice_id in range(scan_slices[s_idx])
    ]

    data = []
    for i in range(0, len(pred), batch_size):
        data.append(
            (
                {"metadata": metadata[i : i + batch_size]},
                {"pred": pred[i : i + batch_size], "target": target[i : i + batch_size]},
            )
        )
    return data


class TestReconEvaluator(unittest.TestCase):
    def _cmp_results(self, a: Mapping, b: Mapping, cmp_func="eq"):
        if cmp_func == "allclose":
            cmp_func = lambda x, y: np.allclose(x, y)  # noqa: E731
        elif cmp_func in ("equal", "eq"):
            cmp_func = lambda x, y: np.all(x == y)  # noqa: E731

        assert a.keys() == b.keys()
        for k in a.keys():
            assert cmp_func(a[k], b[k])

    def test_evaluation_metrics(self):
        # 2D
        evaluator = MockReconEvaluator()
        prediction = {
            "pred": torch.rand(384, 384, 1, 2),
            "target": torch.rand(384, 384, 1, 2),
        }
        vals = evaluator.evaluate_prediction(prediction)
        expected = evaluator.evaluate_prediction_old(prediction)

        # Maps from old strings to new strings
        key_mapping = {
            "l1": "l1",
            "l2": "l2",
            "psnr": "psnr",
            "ssim": "ssim_old",
        }

        assert all(
            np.allclose(vals[key_mapping[k]], expected[k]) for k in expected.keys()
        ), "\n".join(
            "{}\tValue: {:.6f}\tExpected: {:.6f}".format(k, vals[key_mapping[k]], expected[k])
            for k in expected
        )

    def test_flush(self):
        scan_slices = (2, 2, 2)
        pred = torch.rand(sum(scan_slices), 20, 20, 1, 2)
        target = torch.rand(sum(scan_slices), 20, 20, 1, 2)

        data = _build_mock_data(scan_slices=scan_slices, batch_size=3, pred=pred, target=target)

        # Expected results (no flushing).
        evaluator = MockReconEvaluator()
        for inputs, outputs in data:
            evaluator.process(inputs, outputs)
        expected_results = evaluator.evaluate()
        expected_running_results = evaluator._running_results

        # Results with flushing every 3 examples.
        evaluator = MockReconEvaluator()
        for (inputs, outputs), num_preds_remaining in zip(data, [1, 2]):
            evaluator.process(inputs, outputs)
            evaluator.flush(skip_last_scan=True)
            assert len(evaluator._predictions) == num_preds_remaining
        results = evaluator.evaluate()
        running_results = evaluator._running_results

        self._cmp_results(results, expected_results)
        self._cmp_results(running_results, expected_running_results)

        # Results with flushing every 4 examples.
        data = _build_mock_data(scan_slices=scan_slices, batch_size=4, pred=pred, target=target)
        evaluator = MockReconEvaluator()
        for (inputs, outputs), num_preds_remaining in zip(data, [2, 2]):
            evaluator.process(inputs, outputs)
            evaluator.flush(skip_last_scan=True)
            assert len(evaluator._predictions) == num_preds_remaining
        results = evaluator.evaluate()
        running_results = evaluator._running_results

        self._cmp_results(results, expected_results)
        self._cmp_results(running_results, expected_running_results)

    def test_process_flush(self):
        scan_slices = (3, 8, 2, 5, 10, 5, 17, 7)
        batch_size = 6
        flush_period = 20
        data = _build_mock_data(scan_slices=scan_slices, batch_size=batch_size)

        # Expected results (no flushing).
        evaluator = MockReconEvaluator()
        for inputs, outputs in data:
            evaluator.process(inputs, outputs)
        expected_results = evaluator.evaluate()
        expected_running_results = evaluator._running_results

        # Test setting flush period.
        evaluator = MockReconEvaluator(flush_period=flush_period)
        for inputs, outputs in data:
            evaluator.process(inputs, outputs)
        results = evaluator.evaluate()
        running_results = evaluator._running_results

        self._cmp_results(results, expected_results)
        self._cmp_results(running_results, expected_running_results)


if __name__ == "__main__":
    unittest.main()
