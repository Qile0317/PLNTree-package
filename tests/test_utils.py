"""Tests for patched functions in plntree/utils/utils.py.

Each test class documents which PATCH it covers, so regressions after upstream
updates are immediately attributable.
"""

import pytest
import torch

from plntree.utils.utils import batch_matrix_product


# ---------------------------------------------------------------------------
# Reference implementation (the original expand-based code, kept here for
# ground-truth comparisons in tests — NOT used in production).
# ---------------------------------------------------------------------------

def _batch_matrix_product_reference(matrix: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Original implementation before PATCH (2026-04-05). Used as ground truth."""
    X = x.unsqueeze(2)
    return (matrix.unsqueeze(0).expand(X.size(0), -1, -1) @ X).squeeze(-1)


# ---------------------------------------------------------------------------
# PATCH (2026-04-05): batch_matrix_product — replaced expand+matmul with
# x @ matrix.T to avoid OOM on large levels (e.g. genus, 3211 features).
# ---------------------------------------------------------------------------

class TestBatchMatrixProductCorrectness:
    """Verify the patched batch_matrix_product matches the reference for all
    matrix shapes encountered in PLN-Tree (affiliation matrices, projectors)."""

    def test_square_values_match_reference(self):
        torch.manual_seed(0)
        matrix = torch.randn(8, 8)
        x = torch.randn(16, 8)
        assert torch.allclose(
            batch_matrix_product(matrix, x),
            _batch_matrix_product_reference(matrix, x),
            atol=1e-5,
        )

    def test_non_square_wide_matrix_values_match_reference(self):
        """Projector-like: out < in (compression)."""
        torch.manual_seed(1)
        matrix = torch.randn(4, 16)   # (out=4, in=16)
        x = torch.randn(32, 16)       # (batch=32, in=16)
        assert torch.allclose(
            batch_matrix_product(matrix, x),
            _batch_matrix_product_reference(matrix, x),
            atol=1e-5,
        )

    def test_non_square_tall_matrix_values_match_reference(self):
        """Affiliation-like: out > in (expansion from children to parents is
        unusual, but the API must be shape-agnostic)."""
        torch.manual_seed(2)
        matrix = torch.randn(16, 4)   # (out=16, in=4)
        x = torch.randn(32, 4)        # (batch=32, in=4)
        assert torch.allclose(
            batch_matrix_product(matrix, x),
            _batch_matrix_product_reference(matrix, x),
            atol=1e-5,
        )

    def test_binary_affiliation_matrix_aggregates_correctly(self):
        """Simulate a real affiliation matrix: binary (out_parents x in_children).
        Each parent gets the sum of its children's counts."""
        # 2 parents, 4 children: parent 0 owns children 0,1; parent 1 owns 2,3
        affil = torch.tensor([[1., 1., 0., 0.],
                               [0., 0., 1., 1.]])   # (2, 4)
        x = torch.tensor([[10., 20., 30., 40.],
                           [ 1.,  2.,  3.,  4.]])   # (batch=2, 4)
        result = batch_matrix_product(affil, x)
        expected = torch.tensor([[30., 70.],
                                  [ 3.,  7.]])
        assert torch.allclose(result, expected)

    def test_single_sample_batch(self):
        torch.manual_seed(3)
        matrix = torch.randn(5, 10)
        x = torch.randn(1, 10)
        assert torch.allclose(
            batch_matrix_product(matrix, x),
            _batch_matrix_product_reference(matrix, x),
            atol=1e-5,
        )

    def test_float64_dtype(self):
        torch.manual_seed(4)
        matrix = torch.randn(6, 12, dtype=torch.float64)
        x = torch.randn(8, 12, dtype=torch.float64)
        assert torch.allclose(
            batch_matrix_product(matrix, x),
            _batch_matrix_product_reference(matrix, x),
            atol=1e-10,
        )


class TestBatchMatrixProductOutputShape:
    """Verify output shapes are correct — (batch, out) in all cases."""

    @pytest.mark.parametrize("batch,out,inp", [
        (1, 3, 3),       # square, single sample
        (32, 70, 201),   # affiliation: phylum <- class (MicrobeAtlas scale)
        (32, 201, 453),  # affiliation: class <- order
        (32, 453, 845),  # affiliation: order <- family
        (32, 845, 3211), # affiliation: family <- genus (was the OOM case)
        (32768, 8, 16),  # large batch, small matrices
    ])
    def test_output_shape(self, batch, out, inp):
        matrix = torch.randn(out, inp)
        x = torch.randn(batch, inp)
        result = batch_matrix_product(matrix, x)
        assert result.shape == (batch, out)


class TestBatchMatrixProductGradients:
    """Verify gradients flow through the patched implementation."""

    def test_gradient_flows_through_x(self):
        matrix = torch.randn(4, 8)
        x = torch.randn(16, 8, requires_grad=True)
        result = batch_matrix_product(matrix, x)
        result.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_flows_through_matrix(self):
        matrix = torch.randn(4, 8, requires_grad=True)
        x = torch.randn(16, 8)
        result = batch_matrix_product(matrix, x)
        result.sum().backward()
        assert matrix.grad is not None
        assert matrix.grad.shape == matrix.shape

    def test_gradient_values_match_reference(self):
        """Gradient of the patch must equal gradient of the original code."""
        torch.manual_seed(5)

        matrix_ref = torch.randn(6, 10, requires_grad=True)
        x_ref = torch.randn(20, 10, requires_grad=True)
        _batch_matrix_product_reference(matrix_ref, x_ref).sum().backward()

        matrix_new = matrix_ref.detach().clone().requires_grad_(True)
        x_new = x_ref.detach().clone().requires_grad_(True)
        batch_matrix_product(matrix_new, x_new).sum().backward()

        assert torch.allclose(matrix_ref.grad, matrix_new.grad, atol=1e-6)
        assert torch.allclose(x_ref.grad, x_new.grad, atol=1e-6)
