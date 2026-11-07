"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned).

Uses restartable CMA-ES with lightweight covariance updates.
Structural change: single long CMA-ES run with stagnation-triggered restarts
instead of fixed multi-phase structure. Faster per-generation computation
via Cholesky-based sampling and periodic eigendecomposition only when needed.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Restartable CMA-ES for ill-conditioned BBOB-style problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_single(self, func, rng, x0, sigma0, max_calls, deadline, stagnation_window=30):
        """
        Single CMA-ES run with stagnation detection.
        Returns best_x, best_f, calls_used, stalled (bool).
        """
        dim = self.dim
        # Population size: larger for better covariance estimation
        n = max(6, int(4 + 3 * np.log(dim)))
        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(dim)
        calls = 0
        
        # CMA-ES parameters
        c_sigma = (1 + 2 * max(0, np.sqrt((n - 1) / dim) - 1)) / (dim + 2)
        c_c = (4 + n / dim) / (dim + 4 + 2 * n / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + n)
        c_mu = min(1 - c_1, 2 * (2 - (dim + 2) / (n + 4)) / ((dim + 2) ** 2 + n))
        mu = n // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Use Cholesky for sampling, update eigendecomp less frequently
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-12, None)
        C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        gen = 0
        best_f_history = [best_f]
        stalled = False
        
        while calls < max_calls and time.time() < deadline:
            gen += 1
            # Generate population
            y_list = []
            f_list = []
            for i in range(n):
                if calls >= max_calls or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                y_list.append(y)
                f_list.append(fy)
            
            if len(y_list) < 2:
                break
            
            # Sort by fitness
            idx = np.argsort(f_list)
            y_sorted = np.array([y_list[i] for i in idx])
            f_sorted = [f_list[i] for i in idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()
            
            best_f_history.append(best_f)
            
            # Stagnation check
            if gen >= stagnation_window:
                recent = best_f_history[-stagnation_window:]
                if recent[0] - recent[-1] < 1e-6 * max(1.0, abs(recent[-1])):
                    stalled = True
                    break
            
            # Recombination
            y_w = np.zeros(dim)
            for i in range(min(mu, len(y_sorted))):
                y_w += weights[i] * y_sorted[i]
            
            x_old = x.copy()
            x = y_w
            
            # Compute C update from best mu solutions
            C_update = np.zeros((dim, dim))
            for i in range(min(mu, len(y_sorted))):
                d = (y_sorted[i] - x_old) / sigma
                C_update += weights[i] * np.outer(d, d)
            
            # Update path variables
            h_sigma = np.sqrt((1 - np.exp(-2 / (calls * c_sigma))) / (1 - np.exp(-2 / (calls * c_sigma))))
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ (x - x_old))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x - x_old)
            
            # Update C
            C = (1 + c_1 * (mu_eff - 1) - c_1 - c_mu) * C \
                + c_1 / sigma ** 2 * np.outer(x - x_old, x - x_old) \
                + c_mu / sigma ** 2 * C_update
            
            # Ensure PSD and update C_inv_sqrt every 5 generations
            C = (C + C.T) / 2
            if gen % 5 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            else:
                # Quick PSD fix without full eigendecomp
                eigvals_quick = np.linalg.eigvalsh(C)
                if eigvals_quick.min() < 1e-12:
                    C += (1e-12 - eigvals_quick.min()) * np.eye(dim)
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)
            
            if sigma < 1e-10:
                stalled = True
                break
        
        return best_x, best_f, calls, stalled

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Initial sampling: spread out to find a good basin
        n_init = min(15, budget // 8)
        for i in range(n_init):
            if total_calls >= budget or time.time() > deadline - 10.0:
                break
            x = rng.uniform(-4, 4, size=dim)
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Main CMA-ES loop with restarts on stagnation
        remaining = budget - total_calls
        max_restarts = 5
        restart_count = 0
        
        while remaining > 20 and time.time() < deadline - 10.0 and restart_count < max_restarts:
            # Determine sigma based on how far we've progressed
            if restart_count == 0:
                sigma0 = 1.5
            else:
                # Shrink sigma for restarts to refine around current best
                sigma0 = max(0.05, 1.5 * (0.5 ** restart_count))
            
            x0 = best_x.copy()
            
            x_best, f_best, calls_used, stalled = self._cma_es_single(
                func, rng, x0, sigma0, remaining, deadline
            )
            total_calls += calls_used
            remaining = budget - total_calls
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
            
            restart_count += 1
            
            # If not stalled and we still have budget, continue from current best
            # with slightly perturbed starting point for diversity
            if not stalled and remaining > 30:
                # Small perturbation to escape potential local minima
                perturbation = 0.1 * rng.standard_normal(dim)
                x0 = np.clip(best_x + perturbation, -4.5, 4.5)
                fx0 = func(x0)
                total_calls += 1
                remaining = budget - total_calls
                if fx0 < best_f:
                    best_f = fx0
                    best_x = x0.copy()
        
        # Final polish if budget remains
        if remaining > 10 and time.time() < deadline - 5.0:
            x_polish, f_polish, calls_polish, _ = self._cma_es_single(
                func, rng, best_x, 0.05, remaining, deadline
            )
            total_calls += calls_polish
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
