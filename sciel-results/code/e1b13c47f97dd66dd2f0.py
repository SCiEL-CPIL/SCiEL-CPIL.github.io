"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Deterministic multi-start Bayesian optimizer with GP surrogate and EI acquisition."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Initial design: LHS + corners + center
        n_init = min(15, budget)
        X = np.zeros((n_init, d))
        for i in range(n_init):
            # Latin hypercube-like points
            X[i] = rng.uniform(-5, 5, d)
        # Add a few deterministic points
        if n_init < budget:
            extra = np.array([
                np.zeros(d),
                np.full(d, 5.0),
                np.full(d, -5.0),
                np.array([5.0, -5.0, 5.0, -5.0, 5.0])[:d],
                np.array([-5.0, 5.0, -5.0, 5.0, -5.0])[:d],
            ])
            for pt in extra:
                if n_init >= budget:
                    break
                X[n_init] = pt
                n_init += 1

        X = X[:n_init]
        Y = np.array([func(x) for x in X])
        best_idx = np.argmin(Y)
        x_best = X[best_idx].copy()
        f_best = Y[best_idx]

        # GP hyperparameters
        length_scale = 2.0
        noise_var = 1e-6
        signal_var = (np.std(Y) ** 2 + 1e-6) if len(Y) > 1 else 1.0

        def _gp_predict(X_train, Y_train, X_pred, length_scale, noise_var, signal_var):
            """Return mean and variance of GP at X_pred."""
            X_train = np.atleast_2d(X_train)
            X_pred = np.atleast_2d(X_pred)
            n = len(X_train)
            m = len(X_pred)

            # RBF kernel
            def rbf(A, B):
                # Squared distance
                A2 = np.sum(A * A, axis=1, keepdims=True)
                B2 = np.sum(B * B, axis=1, keepdims=True)
                D2 = A2 + B2.T - 2 * A @ B.T
                np.maximum(D2, 0, out=D2)
                return np.exp(-D2 / (2 * length_scale * length_scale))

            K = rbf(X_train, X_train) + noise_var * np.eye(n)
            Ks = rbf(X_train, X_pred)
            Kpp = rbf(X_pred, X_pred)

            # Cholesky
            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(n))

            # Mean
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            mu = Ks.T @ alpha

            # Variance
            v = np.linalg.solve(L, Ks)
            var = signal_var - np.sum(v * v, axis=0)
            var = np.maximum(var, 1e-10)

            return mu, var

        # Adaptive EI acquisition with local search refinement
        for step in range(n_init, budget):
            # Update GP hyperparameters occasionally
            if (step - n_init) % 20 == 0 and len(X) > 5:
                # Simple adaptation: adjust length_scale based on best improvement distance
                pass  # Keep fixed for determinism and speed

            # Generate candidate points for EI
            n_candidates = 200
            # Mix of random and local perturbations around best
            candidates = np.zeros((n_candidates, d))
            n_local = n_candidates // 3
            # Local around best
            for i in range(n_local):
                candidates[i] = x_best + rng.uniform(-1, 1, d)
            # Random in box
            for i in range(n_local, n_candidates):
                candidates[i] = rng.uniform(-5, 5, d)
            candidates = np.clip(candidates, -5, 5)

            # Compute EI
            X_train = np.vstack([X, candidates])
            mu, var = _gp_predict(X, Y, candidates, length_scale, noise_var, signal_var)
            f_min = np.min(Y)
            sigma_pred = np.sqrt(var)
            Z = (f_min - mu) / (sigma_pred + 1e-10)
            EI = (f_min - mu) * norm.cdf(Z) + sigma_pred * norm.pdf(Z)

            # Pick best candidate by EI
            idx = np.argmax(EI)
            x_new = candidates[idx].copy()

            # Local refinement around x_new using Nelder-Mead-like steps
            # Do a few local steps to exploit
            f_new = func(x_new)
            x_curr = x_new.copy()
            f_curr = f_new
            for _ in range(3):
                # Try small perturbations
                improved = False
                for scale in [0.1, 0.05]:
                    for _ in range(5):
                        x_try = x_curr + scale * rng.standard_normal(d)
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        if f_try < f_curr:
                            x_curr = x_try
                            f_curr = f_try
                            improved = True
                            break
                    if improved:
                        break

            x_new = x_curr
            f_new = f_curr

            # Update GP training data
            X = np.vstack([X, x_new])
            Y = np.append(Y, f_new)

            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()

        return x_best, f_best
# EVOLVE-BLOCK-END
