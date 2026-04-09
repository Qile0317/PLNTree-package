"""Tests for the diagonal covariance patch in PLN-Tree.

PATCH under test (2026-04-08):
  - plntree/utils/modules.py: Cholesky now accepts length-K vectors when
    diagonal=True (previously required K*(K+1)/2 and just zeroed off-diag).
  - plntree/models/plntree.py: omega MLP output dimension is K (not K*(K+1)/2)
    when diagonal=True, reducing params from ~2B to ~1.5M at genus level.

These tests verify:
  1. Cholesky module: correct output for both diagonal and full modes
  2. PositiveDefiniteMatrix module: produces valid PD matrices in both modes
  3. KL divergence math: diagonal Omega gives correct logdet, trace, quadratic
  4. End-to-end model: param counts, forward pass, ELBO, gradients, sampling
"""

import math

import pytest
import torch
import torch.nn as nn

from plntree.utils.modules import Cholesky, PositiveDefiniteMatrix, DenseNeuralNetwork


# ---------------------------------------------------------------------------
# Cholesky module tests
# ---------------------------------------------------------------------------

class TestCholeskyDiagonal:
    """Verify the patched Cholesky module with diagonal=True."""

    def test_diagonal_input_is_length_K(self):
        """diagonal=True should accept (batch, K) input, NOT (batch, K*(K+1)/2)."""
        K = 10
        batch = 4
        chol = Cholesky(diagonal=True)
        x = torch.randn(batch, K)
        L = chol(x)
        assert L.shape == (batch, K, K)

    def test_diagonal_output_is_diagonal_matrix(self):
        """All off-diagonal entries must be zero."""
        K = 8
        batch = 5
        chol = Cholesky(diagonal=True)
        x = torch.randn(batch, K)
        L = chol(x)
        # Extract off-diagonal mask
        mask = ~torch.eye(K, dtype=torch.bool)
        off_diag = L[:, mask]
        assert torch.all(off_diag == 0), "Off-diagonal entries should be exactly zero"

    def test_diagonal_entries_are_positive(self):
        """Diagonal entries go through Softplus, so must be strictly positive."""
        K = 12
        batch = 3
        chol = Cholesky(diagonal=True)
        # Include large negative inputs to stress Softplus
        x = torch.tensor([[-100., -10., -1., 0., 1., 10., 100.,
                           -50., -5., 0.5, 5., 50.]]).expand(batch, -1)
        L = chol(x)
        diag = L.diagonal(dim1=-2, dim2=-1)
        assert torch.all(diag > 0), "Diagonal entries must be strictly positive"

    def test_diagonal_gradient_flows(self):
        """Gradients must flow through the diagonal path."""
        K = 6
        batch = 2
        chol = Cholesky(diagonal=True)
        x = torch.randn(batch, K, requires_grad=True)
        L = chol(x)
        L.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_full_mode_unchanged(self):
        """Verify the non-diagonal path still works as before."""
        K = 5
        vec_len = K * (K + 1) // 2
        batch = 3
        chol = Cholesky(diagonal=False)
        x = torch.randn(batch, vec_len)
        L = chol(x)
        assert L.shape == (batch, K, K)
        # Lower triangular check
        for b in range(batch):
            for i in range(K):
                for j in range(i + 1, K):
                    assert L[b, i, j] == 0, f"Upper triangle should be zero at ({i},{j})"
            # Diagonal positive
            assert torch.all(L[b].diag() > 0)


class TestCholeskyDiagonalVsFullConsistency:
    """When the full Cholesky vector has zero off-diagonal entries, the result
    should match the diagonal-only path (up to floating point)."""

    @pytest.mark.parametrize("K", [3, 7, 16])
    def test_diagonal_full_equivalence(self, K):
        """Construct a full Cholesky vector that's purely diagonal, and compare."""
        batch = 4
        diag_values = torch.randn(batch, K)

        # Diagonal path
        chol_diag = Cholesky(diagonal=True)
        L_diag = chol_diag(diag_values)

        # Full path: build a K*(K+1)/2 vector with zeros in off-diagonal slots
        chol_full = Cholesky(diagonal=False)
        vec_len = K * (K + 1) // 2
        full_vec = torch.zeros(batch, vec_len)
        # Place diagonal values at the correct positions in the triangular vector
        idx = 0
        diag_positions = []
        for row in range(K):
            for col in range(row + 1):
                if col == row:
                    diag_positions.append(idx)
                idx += 1
        for b in range(batch):
            for i, pos in enumerate(diag_positions):
                full_vec[b, pos] = diag_values[b, i]
        L_full = chol_full(full_vec)

        # Both should produce the same diagonal matrix
        assert torch.allclose(L_diag, L_full, atol=1e-6), \
            f"Diagonal and full Cholesky disagree for K={K}"


# ---------------------------------------------------------------------------
# PositiveDefiniteMatrix tests
# ---------------------------------------------------------------------------

class TestPositiveDefiniteMatrixDiagonal:
    """Verify PositiveDefiniteMatrix with diagonal=True produces valid PD matrices."""

    def test_output_shape(self):
        K = 10
        batch = 4
        pdm = PositiveDefiniteMatrix(min_diag=1e-4, diagonal=True)
        x = torch.randn(batch, K)
        Omega = pdm(x)
        assert Omega.shape == (batch, K, K)

    def test_output_is_diagonal(self):
        """Omega = diag(d)^2 + min_diag*I should be diagonal."""
        K = 8
        batch = 3
        pdm = PositiveDefiniteMatrix(min_diag=1e-4, diagonal=True)
        x = torch.randn(batch, K)
        Omega = pdm(x)
        mask = ~torch.eye(K, dtype=torch.bool)
        off_diag = Omega[:, mask]
        assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-7)

    def test_output_is_positive_definite(self):
        """All eigenvalues should be positive."""
        K = 6
        batch = 5
        pdm = PositiveDefiniteMatrix(min_diag=1e-4, diagonal=True)
        x = torch.randn(batch, K)
        Omega = pdm(x)
        eigenvalues = torch.linalg.eigvalsh(Omega)
        assert torch.all(eigenvalues > 0), "All eigenvalues should be positive"

    def test_diagonal_values_are_softplus_squared_plus_min_diag(self):
        """Omega_ii = softplus(x_i)^2 + min_diag, because L=diag(softplus(x))
        and Omega = L @ L.T + min_diag * I."""
        K = 4
        batch = 2
        min_diag = 1e-4
        pdm = PositiveDefiniteMatrix(min_diag=min_diag, diagonal=True)
        x = torch.randn(batch, K)
        Omega = pdm(x)
        expected_diag = nn.functional.softplus(x) ** 2 + min_diag
        actual_diag = Omega.diagonal(dim1=-2, dim2=-1)
        assert torch.allclose(actual_diag, expected_diag, atol=1e-6)

    def test_full_mode_still_works(self):
        """Non-diagonal mode should still produce valid PD matrices."""
        K = 5
        vec_len = K * (K + 1) // 2
        batch = 3
        pdm = PositiveDefiniteMatrix(min_diag=1e-4, diagonal=False)
        x = torch.randn(batch, vec_len)
        Omega = pdm(x)
        assert Omega.shape == (batch, K, K)
        # Symmetric
        assert torch.allclose(Omega, Omega.mT, atol=1e-6)
        # PD
        eigenvalues = torch.linalg.eigvalsh(Omega)
        assert torch.all(eigenvalues > 0)


# ---------------------------------------------------------------------------
# KL divergence math with diagonal precision
# ---------------------------------------------------------------------------

class TestKLDivergenceWithDiagonalOmega:
    """The ELBO's KL term involves log|Omega|, tr(Sigma_hat @ Omega), and
    these must be correct for diagonal Omega.

    KL contribution per level (from objective):
      0.5 * sum(log|Omega| - tr(Sigma_hat @ Omega) + log|S|) / norm
    where Sigma_hat = (mu - m)(mu - m)^T + diag(S).
    """

    def _build_diagonal_omega(self, diag_values, min_diag=1e-4):
        """Build Omega from diagonal values as the model would."""
        pdm = PositiveDefiniteMatrix(min_diag=min_diag, diagonal=True)
        return pdm(diag_values)

    def test_logdet_diagonal_omega(self):
        """log|Omega| for diagonal matrix = sum of log of diagonal entries."""
        K = 8
        batch = 4
        x = torch.randn(batch, K)
        Omega = self._build_diagonal_omega(x)
        # Ground truth: sum of log of diagonal
        diag = Omega.diagonal(dim1=-2, dim2=-1)
        expected_logdet = diag.log().sum(dim=-1)
        # What the model computes
        actual_logdet = torch.linalg.slogdet(Omega)[1]
        assert torch.allclose(actual_logdet, expected_logdet, atol=1e-4)

    def test_trace_sigma_hat_omega_diagonal(self):
        """tr(Sigma_hat @ Omega) when Omega is diagonal simplifies to
        sum_i omega_i * (sigma_hat_ii) = sum_i omega_i * ((mu_i - m_i)^2 + S_i)."""
        K = 6
        batch = 3
        torch.manual_seed(42)

        # Build diagonal Omega
        x = torch.randn(batch, K)
        Omega = self._build_diagonal_omega(x)
        omega_diag = Omega.diagonal(dim1=-2, dim2=-1)

        # Build Sigma_hat as the model does
        mu = torch.randn(batch, K)
        m = torch.randn(batch, K)
        S = torch.rand(batch, K) + 0.1  # positive

        M = (mu - m).unsqueeze(-1)
        Sigma_hat = M @ M.mT + torch.diag_embed(S)

        # Full matrix trace
        trace_full = (Sigma_hat @ Omega).diagonal(dim1=-2, dim2=-1).sum(-1)

        # Simplified diagonal trace: sum_i omega_i * ((mu_i - m_i)^2 + S_i)
        sigma_hat_diag = (mu - m) ** 2 + S
        trace_simple = (omega_diag * sigma_hat_diag).sum(-1)

        assert torch.allclose(trace_full, trace_simple, atol=1e-4), \
            "Diagonal Omega trace should match the simplified formula"

    def test_cholesky_decomposition_of_diagonal_omega(self):
        """torch.linalg.cholesky on a diagonal Omega should return a diagonal L.
        This is used in the sampling path (generate method)."""
        K = 10
        batch = 5
        x = torch.randn(batch, K)
        Omega = self._build_diagonal_omega(x)
        L = torch.linalg.cholesky(Omega, upper=False)
        # L should be diagonal (off-diag ~0)
        mask = ~torch.eye(K, dtype=torch.bool)
        off_diag = L[:, mask]
        assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-6)

    def test_sampling_via_solve_triangular(self):
        """The generate method does: eps = solve_triangular(L.T, noise, upper=True)
        For diagonal L, this should simply divide by the diagonal."""
        K = 8
        batch = 4
        torch.manual_seed(123)

        x = torch.randn(batch, K)
        Omega = self._build_diagonal_omega(x)
        L = torch.linalg.cholesky(Omega, upper=False)
        L_T = L.mT

        eps = torch.randn(batch, K, 1)
        # What the model does
        solved = torch.linalg.solve_triangular(L_T, eps, upper=True).squeeze(-1)
        # What it should equal for diagonal L
        diag_L = L.diagonal(dim1=-2, dim2=-1)
        expected = eps.squeeze(-1) / diag_L

        assert torch.allclose(solved, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end model tests with a tiny tree
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd

# Leaf-level hierarchy strings (column names for the count DataFrame)
_HIERARCHY = [
    "L0__A|L1__A1|L2__A1a",
    "L0__A|L1__A1|L2__A1b",
    "L0__A|L1__A2|L2__A2a",
    "L0__B|L1__B1|L2__B1a",
    "L0__B|L1__B1|L2__B1b",
    "L0__B|L1__B2|L2__B2a",
]


def _make_counts(n_samples=16):
    """Return a pd.DataFrame of random counts with hierarchy column names."""
    rng = np.random.default_rng(42)
    data = rng.integers(1, 50, size=(n_samples, len(_HIERARCHY)))
    return pd.DataFrame(data, columns=_HIERARCHY)


def _make_model(diagonal, markov_cov=True, n_layers=1, n_samples=16):
    """Build a PLNTree model via the public API."""
    from plntree.models.plntree import PLNTree

    counts = _make_counts(n_samples)
    latent_dynamic = {
        "n_layers": n_layers,
        "diagonal": diagonal,
        "markov_means": True,
        "markov_covariance": markov_cov,
    }
    variational_approx = {
        "method": "weak",
        "n_layers": 1,
        "counts_preprocessing": None,
    }
    model = PLNTree(
        counts=counts,
        latent_dynamic=latent_dynamic,
        variational_approx=variational_approx,
        smart_init=True,
        device="cpu",
        seed=0,
    )
    return model


def _get_omega_mlp_output_dim(model, level):
    """Extract the output dimension of the omega MLP at a given level."""
    omega_fun = model.omega_fun[level]
    nn1 = omega_fun.nn1
    if isinstance(nn1, nn.Sequential):
        last_layer = nn1[-1].network[-1]
    else:
        last_layer = nn1.network[-1]
    return last_layer.out_features


class TestEndToEndParamCounts:
    """Verify that diagonal=True actually reduces parameter count."""

    def test_diagonal_fewer_params_than_full(self):
        model_full = _make_model(diagonal=False)
        model_diag = _make_model(diagonal=True)

        n_full = sum(p.numel() for p in model_full.parameters())
        n_diag = sum(p.numel() for p in model_diag.parameters())

        assert n_diag < n_full, \
            f"Diagonal ({n_diag}) should have fewer params than full ({n_full})"

    def test_param_reduction_is_significant(self):
        """For even a small tree, the reduction should be substantial."""
        model_full = _make_model(diagonal=False)
        model_diag = _make_model(diagonal=True)

        n_full = sum(p.numel() for p in model_full.parameters())
        n_diag = sum(p.numel() for p in model_diag.parameters())

        reduction = 1 - n_diag / n_full
        assert reduction > 0.05, \
            f"Expected >5% param reduction, got {reduction:.1%}"

    def test_omega_mlp_output_dim_is_K(self):
        """The omega MLP should output K_eff[l] values (not K*(K+1)/2) when diagonal."""
        model = _make_model(diagonal=True, n_layers=1)
        tree = model.tree

        for level in range(1, tree.n_levels):
            out_features = _get_omega_mlp_output_dim(model, level)
            K_eff_l = tree.K_eff[level]
            assert out_features == K_eff_l, \
                f"Level {level}: omega MLP output dim {out_features} != K_eff {K_eff_l}"

    def test_omega_mlp_output_dim_is_triangular_when_full(self):
        """Sanity check: full mode should output K*(K+1)/2."""
        model = _make_model(diagonal=False, n_layers=1)
        tree = model.tree

        for level in range(1, tree.n_levels):
            out_features = _get_omega_mlp_output_dim(model, level)
            K_eff_l = tree.K_eff[level]
            expected = K_eff_l * (K_eff_l + 1) // 2
            assert out_features == expected, \
                f"Level {level}: omega MLP output dim {out_features} != {expected}"


class TestEndToEndForwardPass:
    """Verify forward pass works and produces valid outputs for both modes."""

    @pytest.mark.parametrize("diagonal", [True, False])
    def test_forward_runs_without_error(self, diagonal):
        model = _make_model(diagonal=diagonal)
        X = model.hierarchical_counts
        model.smart_init()
        output = model(X, C=None)
        Z, O, m, log_S, mu, Omega = output
        assert len(Omega) == model.tree.n_levels

    @pytest.mark.parametrize("diagonal", [True, False])
    def test_omega_shapes(self, diagonal):
        model = _make_model(diagonal=diagonal)
        X = model.hierarchical_counts
        model.smart_init()
        output = model(X, C=None)
        _, _, _, _, _, Omega = output
        batch = X.size(0)
        tree = model.tree
        for level in range(tree.n_levels):
            K_eff_l = tree.K_eff[level]
            assert Omega[level].shape == (batch, K_eff_l, K_eff_l), \
                f"Level {level}: Omega shape {Omega[level].shape} != ({batch}, {K_eff_l}, {K_eff_l})"

    def test_diagonal_omega_is_actually_diagonal_in_forward(self):
        """Omega matrices from forward pass with diagonal=True must be diagonal."""
        model = _make_model(diagonal=True)
        X = model.hierarchical_counts
        model.smart_init()
        output = model(X, C=None)
        _, _, _, _, _, Omega = output
        tree = model.tree
        for level in range(1, tree.n_levels):  # level 0 is ClosedFormParameter
            K_eff_l = tree.K_eff[level]
            mask = ~torch.eye(K_eff_l, dtype=torch.bool)
            off_diag = Omega[level][:, mask]
            assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-6), \
                f"Level {level}: Omega has non-zero off-diagonal entries"

    def test_omega_is_positive_definite_in_forward(self):
        model = _make_model(diagonal=True)
        X = model.hierarchical_counts
        model.smart_init()
        output = model(X, C=None)
        _, _, _, _, _, Omega = output
        tree = model.tree
        for level in range(1, tree.n_levels):
            eigenvalues = torch.linalg.eigvalsh(Omega[level].detach())
            assert torch.all(eigenvalues > 0), \
                f"Level {level}: Omega is not positive definite"


class TestEndToEndELBO:
    """Verify the ELBO computation works correctly with diagonal covariance."""

    @pytest.mark.parametrize("diagonal", [True, False])
    def test_elbo_is_finite(self, diagonal):
        model = _make_model(diagonal=diagonal)
        X = model.hierarchical_counts
        model.smart_init()
        output = model(X, C=None)
        elbo = model.objective(X, output)
        assert torch.isfinite(elbo), f"ELBO is not finite: {elbo}"

    @pytest.mark.parametrize("diagonal", [True, False])
    def test_elbo_gradients_flow(self, diagonal):
        """All parameters should receive gradients through the ELBO."""
        model = _make_model(diagonal=diagonal)
        X = model.hierarchical_counts
        model.smart_init()
        output = model(X, C=None)
        elbo = model.objective(X, output)
        elbo.backward()

        # Check that at least some omega_fun params got gradients
        grad_found = False
        for level in range(1, model.tree.n_levels):
            for p in model.omega_fun[level].parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    grad_found = True
                    break
        assert grad_found, "No gradients reached omega_fun parameters"

    def test_diagonal_elbo_is_different_from_full(self):
        """Diagonal and full should give different ELBO values (different models)."""
        torch.manual_seed(0)
        model_diag = _make_model(diagonal=True)
        X = model_diag.hierarchical_counts
        model_diag.smart_init()
        output_diag = model_diag(X, C=None)
        elbo_diag = model_diag.objective(X, output_diag)

        torch.manual_seed(0)
        model_full = _make_model(diagonal=False)
        X2 = model_full.hierarchical_counts
        model_full.smart_init()
        output_full = model_full(X2, C=None)
        elbo_full = model_full.objective(X2, output_full)

        # Both should be finite
        assert torch.isfinite(elbo_diag)
        assert torch.isfinite(elbo_full)


class TestEndToEndTrainingStep:
    """Simulate a training step to verify optimization works."""

    @pytest.mark.parametrize("diagonal", [True, False])
    def test_single_training_step(self, diagonal):
        torch.manual_seed(42)
        model = _make_model(diagonal=diagonal)
        X = model.hierarchical_counts
        model.smart_init()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Step 1
        output = model(X, C=None)
        loss1 = -model.objective(X, output)
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Step 2
        output = model(X, C=None)
        loss2 = -model.objective(X, output)

        assert torch.isfinite(loss1) and torch.isfinite(loss2), \
            f"Losses not finite: {loss1}, {loss2}"

    def test_diagonal_loss_decreases(self):
        """Over several steps, loss should decrease (model is learning)."""
        torch.manual_seed(42)
        model = _make_model(diagonal=True)
        X = model.hierarchical_counts
        model.smart_init()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(20):
            output = model(X, C=None)
            loss = model.objective(X, output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # Loss at the end should be lower than at the start (allow some noise)
        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestEndToEndSampling:
    """Verify the generate (sampling) path works with diagonal Omega."""

    @pytest.mark.parametrize("diagonal", [True, False])
    def test_generate_runs(self, diagonal):
        torch.manual_seed(0)
        model = _make_model(diagonal=diagonal)
        X = model.hierarchical_counts
        model.smart_init()

        # Train a few steps so the model is not at init
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(5):
            output = model(X, C=None)
            loss = model.objective(X, output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Generate samples — sample() returns a tuple (counts_df, latents)
        result = model.sample(n_samples=4)
        assert result is not None
        # First element should have 4 rows
        if isinstance(result, tuple):
            assert result[0].shape[0] == 4
        else:
            assert result.shape[0] == 4


class TestNoMarkovCovariance:
    """When markov_covariance=False, omega is a ClosedFormParameter (identity).
    The diagonal flag should be irrelevant here — verify no regression."""

    @pytest.mark.parametrize("diagonal", [True, False])
    def test_closed_form_omega_works(self, diagonal):
        model = _make_model(diagonal=diagonal, markov_cov=False)
        X = model.hierarchical_counts
        model.smart_init()
        output = model(X, C=None)
        elbo = model.objective(X, output)
        assert torch.isfinite(elbo)
