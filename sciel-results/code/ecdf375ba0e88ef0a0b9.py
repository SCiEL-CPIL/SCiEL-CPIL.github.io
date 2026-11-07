"""CMA-ES optimizer for BBOB f19 (Griewank-Rosenbrock).

Uses a robust CMA-ES implementation with a two-phase strategy:
1. Global exploration with a moderate step size to find the general basin.
2. Local refinement with a smaller step size to converge to the minimum.

The covariance matrix adaptation in CMA-ES is key to navigating the rotated
landscape efficiently. We also add a simple local search polish using Nelder-Mead
or gradient-free local search if time permits.
"""
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES optimizer with adaptive restarts and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, x0, f0, rng, budget, deadline, sigma0=2.0):
        """Run CMA-ES optimization."""
        n = self.dim
        # Standard CMA-ES parameters
        pop = max(4, int(4 + 3 * np.log(n)))
        mu = pop // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = float(w.sum() ** 2)
        
        # Step size control
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + mueff)) + cs
        
        # Covariance matrix adaptation
        c1 = (2 + (mueff - 2) / (n + 2)) / ((n + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        
        # State variables
        c = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(n)
        Csqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = c.copy()
        best_f = f0
        
        it = 0
        stagnation_count = 0
        last_best_f = f0
        
        while it < budget and time.time() < deadline:
            Y = np.zeros((pop, n))
            F = np.empty(pop)
            
            # Generate population
            for i in range(pop):
                z = rng.standard_normal(n)
                y = Csqrt @ z
                x = c + sigma * y
                x = np.clip(x, -5.0, 5.0)
                F[i] = float(self._f(x))
                Y[i] = y
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x.copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            
            # Sort and update
            idx = np.argsort(F)
            Ys = Y[idx]
            Fs = F[idx]
            
            ymean = (w[:, None] * Ys[:mu]).sum(axis=0)
            d = Csqrt @ ymean
            
            # Update mean
            c = c + cs * d
            c = np.clip(c, -5.0, 5.0)
            
            # Update step size control
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * ymean
            
            # Check if sigma is too small (stagnation)
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (it + 1)) < \
                    (1.4 + 2 / (n + 1)) * chiN:
                p_c = (1 - cs) * p_c + cs * d
                hsig = 1.0
            else:
                hsig = 0.0
            
            # Update covariance matrix
            delta_sig = (1 - (1 - cs) ** 2) / (1 - (1 - cs) ** ds)
            C = ((1 - c1 - cmu + c1 * hsig - cmu * hsig) * C
                 + c1 * np.outer(d, d)
                 + cmu * (w[:, None] * (Ys[:mu] @ Csqrt)).sum(axis=0)[:, None]
                 * (Ys[:mu] @ Csqrt))
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chiN - 1))
            sigma = float(np.clip(sigma, 1e-4, 10.0))
            
            # Refresh square root of C
            try:
                evals, evecs = np.linalg.eigh(C)
                evals = np.clip(evals, 1e-12, None)
                Csqrt = (evecs * np.sqrt(evals)) @ evecs.T
            except np.linalg.LinAlgError:
                # Reset if numerical issues
                C = np.eye(n)
                Csqrt = np.eye(n)
                sigma = sigma0 * 0.5
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                c = best_x.copy()
                
            it += 1
            
            # Early stopping if no progress for a long time
            if it > 10 and best_f == last_best_f and it % 10 == 0:
                stagnation_count += 10
            else:
                last_best_f = best_f
                
            if stagnation_count > 3 * pop:
                break
                
        return best_x, best_f

    def _local_polish(self, x0, f0, rng, budget, deadline):
        """Local refinement using Nelder-Mead or simple coordinate descent."""
        n = self.dim
        best_x = x0.copy()
        best_f = f0
        
        # Try Nelder-Mead with limited function evaluations
        if budget < 5:
            return best_x, best_f
            
        def wrapper(x):
            x = np.clip(x, -5.0, 5.0)
            return float(self._f(x))
        
        try:
            res = minimize(wrapper, best_x, method='Nelder-Mead',
                          options={'maxiter': min(budget, 200),
                                   'xatol': 1e-6, 'fatol': 1e-6,
                                   'adaptive': True})
            if res.success and res.fun < best_f:
                best_x = np.clip(res.x, -5.0, 5.0)
                best_f = float(res.fun)
        except Exception:
            pass
        
        # Additional coordinate descent polish if time permits
        if time.time() < deadline - 5:
            step = 0.1
            for _ in range(min(50, budget)):
                improved = False
                for i in range(n):
                    for delta in [step, -step]:
                        x_trial = best_x.copy()
                        x_trial[i] += delta
                        x_trial = np.clip(x_trial, -5.0, 5.0)
                        f_trial = float(self._f(x_trial))
                        if f_trial < best_f:
                            best_x = x_trial
                            best_f = f_trial
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break
        
        return best_x, best_f

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin
        rng = np.random.default_rng(42)
        self._f = func
        
        best_x = None
        best_f = np.inf
        
        # Initial evaluation to have a baseline
        x0 = np.zeros(self.dim)
        f0 = float(func(x0))
        best_x = x0.copy()
        best_f = f0
        calls = 1
        
        # Phase 1: Global exploration with CMA-ES
        # Use about 70% of the budget for this phase
        budget_phase1 = int(0.7 * self.budget)
        x1, f1 = self._cmaes(x0, f0, rng, budget_phase1, deadline, sigma0=2.5)
        if f1 < best_f:
            best_f = f1
            best_x = x1.copy()
        calls += budget_phase1
        
        # Check time
        if time.time() >= deadline - 10 or calls >= self.budget:
            return best_x, best_f
            
        # Phase 2: Local polish
        budget_phase2 = self.budget - calls
        x2, f2 = self._local_polish(best_x, best_f, rng, budget_phase2, deadline)
        if f2 < best_f:
            best_f = f2
            best_x = x2.copy()
            
        return best_x, best_f
