"""Sparse GP contextual bandit for non-stationary linear rewards.

Models the global reward surface over the augmented feature vector
(context concatenated with one-hot arm encoding) using a sparse
inducing-point variational GP with proper predictive variance and
Sherman-Morrison-Woodbury covariance updates for non-stationarity.
"""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Sparse (variational) GP bandit over augmented context+arm features."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = dim + n_arms

        # Inducing-point count
        self.M = int(min(48, max(12, 2 * self.d_aug)))

        # Kernel hyperparameters (RBF + linear mixture).
        self.length_scale = 1.0
        self.sigma_rbf = 1.0
        self.sigma_lin = 0.2
        self.noise = 0.1

        # Random inducing points
        rng = np.random.default_rng(42)
        self.Z = rng.normal(0.0, self.length_scale, size=(self.M, self.d_aug))

        # Variational posterior q(u) = N(m, S) over inducing values.
        self.m = np.zeros(self.M)
        self.S = np.eye(self.M) * self.noise

        # Precompute kernel matrices among inducing points.
        self._recompute_Kuu()

        # Exploration coefficient, decays slowly.
        self.beta = 1.0
        self.beta_decay = 0.999
        self.t = 0

        # Forgetting for non-stationarity
        self.forget = 0.99

    def _recompute_Kuu(self):
        Z = self.Z
        sq = (Z ** 2).sum(1)
        D2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
        np.maximum(D2, 0.0, out=D2)
        K_rbf = self.sigma_rbf ** 2 * np.exp(-D2 / (2.0 * self.length_scale ** 2))
        K_lin = self.sigma_lin ** 2 * (Z @ Z.T)
        self.Kuu = K_rbf + K_lin + self.noise * np.eye(self.M)

    def _kz(self, x):
        x = x.reshape(1, -1)
        diff = x - self.Z
        D2 = (diff ** 2).sum(1)
        k_rbf = self.sigma_rbf ** 2 * np.exp(-D2 / (2.0 * self.length_scale ** 2))
        k_lin = self.sigma_lin ** 2 * (x @ self.Z.T).reshape(-1)
        return k_rbf + k_lin

    def _kxx(self):
        return self.sigma_rbf ** 2 + self.sigma_lin ** 2

    def _predict(self, x):
        """Returns (mean, variance) of q(y|x) under the variational posterior."""
        kz = self._kz(x)
        kxx = self._kxx()

        # Solve S \ kz
        S_inv_kz = np.linalg.solve(self.S, kz)
        S_inv_kz_m = np.linalg.solve(self.S, kz * self.m)

        # Mean: k(x,Z) m + (k(x,Z) S^{-1} Z^T (S - Kuu) m)
        # Simplified: mean = kz @ m + (kz * S_inv_kz_m).sum()
        mean = kz @ self.m + np.dot(kz, S_inv_kz_m)

        # Variance: kxx - kz S^{-1} (S - Kuu) S^{-1} kz^T
        # = kxx - kz S^{-1} kz + kz S^{-1} Kuu S^{-1} kz^T
        term1 = np.dot(kz, S_inv_kz)
        # Compute kz S^{-1} Kuu S^{-1} kz^T
        S_inv_Kuu = np.linalg.solve(self.S, self.Kuu)
        S_inv_Kuu_S_inv_kz = np.linalg.solve(self.S, S_inv_Kuu.T @ S_inv_kz)
        term2 = np.dot(kz, S_inv_Kuu_S_inv_kz)
        var = kxx - term1 + term2
        var = max(var, 1e-6)
        return mean, var

    def select(self, context):
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

        kz = self._kz(x)
        kxx = self._kxx()

        # Predictive mean/var at x under current q
        S_inv_kz = np.linalg.solve(self.S, kz)
        S_inv_kz_m = np.linalg.solve(self.S, kz * self.m)
        pred_mean = kz @ self.m + np.dot(kz, S_inv_kz_m)

        # Variance for update
        S_inv_Kuu = np.linalg.solve(self.S, self.Kuu)
        S_inv_Kuu_S_inv_kz = np.linalg.solve(self.S, S_inv_Kuu.T @ S_inv_kz)
        pred_var = kxx - np.dot(kz, S_inv_kz) + np.dot(kz, S_inv_Kuu_S_inv_kz)
        pred_var = max(pred_var, 1e-6)

        # Natural-gradient update
        lam = 1.0 / (pred_var + self.noise)
        delta_m = (self.S @ S_inv_kz) * (reward - pred_mean) * lam
        self.m = self.forget * self.m + delta_m

        # Sherman-Morrison-Woodbury covariance update
        # S_new = S - S S^{-1} kz kz^T S^{-1} S / (1 + kz^T S^{-1} kz)
        kz_S_inv_kz = np.dot(kz, S_inv_kz)
        denom = 1.0 + kz_S_inv_kz
        v = self.S @ S_inv_kz
        correction = np.outer(v, kz) / denom
        S_updated = self.S - correction

        # Apply forgetting and ensure positive-definite-ish
        self.S = self.forget * S_updated + (1.0 - self.forget) * (self.noise * np.eye(self.M))
        self.S += 1e-6 * np.eye(self.M)

        # Decay exploration
        self.beta *= self.beta_decay
        self.t += 1
# EVOLVE-BLOCK-END
