from __future__ import annotations

import numpy as np
import torch

from domain_gated_mf.demo import AnalyticalDemoEvaluator, make_demo_records, make_demo_tasks
from domain_gated_mf.model import DomainGatedSurrogate
from domain_gated_mf.training import TorchSurrogateTrainer, records_to_samples


def test_residual_chain_and_shared_adapted_outputs_have_expected_shapes():
    model = DomainGatedSurrogate(4, 5, 3, ["LCE"], hidden=(8,), adapter_hidden=(4,))
    output = model(torch.randn(2, 4), "LCE")
    assert output.responses["L"].shape == (2, 3)
    assert torch.allclose(
        output.responses["M"],
        model.lf_head(output.adapted_latent) + model.mf_residual_head(output.adapted_latent),
    )
    assert torch.allclose(
        output.responses["H"],
        output.responses["M"] + model.hf_residual_head(output.adapted_latent),
    )
    assert output.hf_scale.shape == (2, 3)


def test_field_operator_and_loss_are_executable():
    tasks = make_demo_tasks()
    task = tasks[0]
    evaluator = AnalyticalDemoEvaluator()
    records = make_demo_records(task, evaluator, 2, 8, task.fidelities[2].level)
    model = DomainGatedSurrogate(10, 6, 4, [item.name for item in tasks], hidden=(8,), adapter_hidden=(4,), field_coordinate_dim=1, field_dim=1)
    samples = records_to_samples(task, records)
    trainer = TorchSurrogateTrainer(model, learning_rate=1.0e-3, lambda_phys=0.0)
    before = [value.detach().clone() for value in model.parameters()]
    loss = trainer.fit(samples, epochs=1, seed=4)
    after = list(model.parameters())
    assert np.isfinite(loss.total)
    assert any(not torch.equal(first, second.detach()) for first, second in zip(before, after))
    output = model(torch.as_tensor(task.descriptor(records[0].design), dtype=torch.float32).unsqueeze(0), task.name, torch.zeros(5, 1))
    assert output.fields is not None
    assert output.fields["H"].shape == (1, 5, 1)

