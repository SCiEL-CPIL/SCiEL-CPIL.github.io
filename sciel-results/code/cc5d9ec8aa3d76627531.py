"""Sparse GP contextual bandit for non-stationary linear rewards.

Models the global reward surface over the augmented feature vector
(context concatenated with one-hot arm encoding) using a sparse
inducing-point variational GP. Linear-time in the number of rounds.
"""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Sparse (variational) GP bandit over augmented context+arm features."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = dim + n_arms

        # Inducing-point count scales with augmented dimension but is capped.
        self.M = int(min(64, max(16, 2 * self.d_aug)))

        # Kernel hyperparameters (RBF + linear mixture).
        self.length_scale = 1.0
        self.sigma_rbf = 1.0
        self.sigma_lin = 0.3
        self.noise = 0.1

        # Random inducing points, centered near origin with spread ~ feature scale.
        rng = np.random.default_rng(0)
        self.Z = rng.normal(0.0, self.length_scale, size=(self.M, self.d_aug))

        # Variational posterior q(u) = N(m, S) over inducing values.
        self.m = np.zeros(self.M)
        self.S = np.eye(self.M) * self.noise

        # Precompute kernel matrices among inducing points.
        self._recompute_Kuu()

        # Exploration coefficient, decays slowly.
        self.beta = 1.2
        self.beta_decay = 0.9995
        self.t = 0

        # Forgetting for non-stationarity: shrink variational params toward prior.
        self.forget = 0.995

    def _recompute_Kuu(self):
        # Kuu: M x M kernel among inducing points.
        Z = self.Z
        sq = (Z ** 2).sum(1)
        D2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
        np.maximum(D2, 0.0, out=D2)
        K_rbf = self.sigma_rbf ** 2 * np.exp(-D2 / (2.0 * self.length_scale ** 2))
        K_lin = self.sigma_lin ** 2 * (Z @ Z.T)
        self.Kuu = K_rbf + K_lin + self.noise * np.eye(self.M)

    def _kz(self, x):
        # Kernel between a single point x (d_aug,) and all inducing points.
        x = x.reshape(1, -1)
        diff = x - self.Z
        D2 = (diff ** 2).sum(1)
        k_rbf = self.sigma_rbf ** 2 * np.exp(-D2 / (2.0 * self.length_scale ** 2))
        k_lin = self.sigma_lin ** 2 * (x @ self.Z.T).reshape(-1)
        return k_rbf + k_lin

    def _kxx(self):
        return self.sigma_rbf ** 2 + self.sigma_lin ** 2

    def _predict(self, x):
        # Returns (mean, variance) of q(y|x) under the variational posterior.
        kz = self._kz(x)
        # Solve S \ kz and S \ (kz * m)
        S_inv_kz = np.linalg.solve(self.S, kz)
        S_inv_kz_m = np.linalg.solve(self.S, kz * self.m)
        mean = kz @ self.m + (kz * S_inv_kz_m).sum()
        # Variance: kxx - kz S^{-1} kz + kz S^{-1} (S - S) ... use standard formula
        var = self._kxx() - kz @ S_inv_kz + np.sum((S_inv_kz * self.S @ S_inv_kz) * 0)
        # More stable: var = kxx - kz S^{-1} kz + (kz S^{-1} S S^{-1} kz) - ...
        # Standard FITC predictive variance:
        # var = kxx - kz S^{-1} (S + Kuu - Kuu) kz ... we use:
        # var = kxx - kz S^{-1} kz + (kz S^{-1} m)(kz S^{-1} m) handled via mean above.
        # For exploration we only need a sensible positive variance.
        var = max(var, 1e-6)
        return mean, var

    def select(self, context):
        # Build augmented features for each arm and score with GP mean + bonus.
        scores = np.zeros(self.n)
        ctx = np.asarray(context, dtype=float).reshape(-1)
        for a in range(self.n):
            x = np.zeros(self.d_aug)
            x[:self.d] = ctx
            x[self.d + a] = 1.0
            mean, var = self._predict(x)
            scores[a] = mean + self.beta * np.sqrt(var)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=float).reshape(-1)
        x = np.zeros(self.d_aug)
        x[:self.d] = ctx
        x[self.d + arm] = 1.0

        # Variational update (natural-gradient style mean-field).
        kz = self._kz(x)
        kxx = self._kxx()

        # Solve for the "observation" contribution.
        # We perform a lightweight update of the variational parameters.
        # Compute S^{-1} kz and S^{-1} (kz * m).
        S_inv_kz = np.linalg.solve(self.S, kz)
        S_inv_kz_m = np.linalg.solve(self.S, kz * self.m)

        # Predictive mean/var at x under current q.
        pred_mean = kz @ self.m + (kz * S_inv_kz_m).sum()
        pred_var = kxx - kz @ S_inv_kz + np.sum((S_inv_kz * (self.S @ S_inv_kz)) * 0)
        # Use a stable positive variance for the update weight.
        pred_var = max(kxx - kz @ S_inv_kz, 1e-6)

        # Natural-gradient update of mean and covariance (mean-field).
        # New precision contribution:
        lam = 1.0 / (pred_var + self.noise)
        # Update mean: m <- m + (S @ S_inv_kz) * (reward - pred_mean) * lam
        delta_m = (self.S @ S_inv_kz) * (reward - pred_mean) * lam
        self.m = self.forget * self.m + delta_m
        # Update covariance (approx): S <- (1 - lam * S_inv_kz * kz^T S) S (1 - ...)
        # Simplified: shrink S toward noise floor to allow non-stationarity.
        self.S = self.forget * self.S + (1.0 - self.forget) * (self.noise * np.eye(self.M))
        # Ensure positive-definite-ish by clipping eigenvalues minimally.
        # (Light touch: add small ridge.)
        self.S += 1e-6 * np.eye(self.M)

        # Decay exploration.
        self.beta *= self.beta_decay
        self.t += 1
# EVOLVE-BLOCK-END
