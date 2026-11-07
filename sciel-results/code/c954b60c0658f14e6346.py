"""Rastrigin-aware optimizer: grid search on periodic structure + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Exploit Rastrigin periodicity: grid on 2*pi lattice + local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin (330s limit)

        rng = np.random.default_rng(12345)

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Period is 2*pi ≈ 6.283, local minima at x_i = 0 (after shift/rotation)
        # The search box is [-5,5]^5, so within one period in each dimension.
        # Strategy:
        # 1. Coarse grid on the periodic structure to find basin
        # 2. Fine local search to refine

        # Phase 1: Structured exploration exploiting periodicity
        # Since the box [-5,5] is roughly one period (2*pi ≈ 6.28),
        # we sample a grid that covers the likely region.
        
        # Try origin first
        counted_func(np.zeros(dim))
        
        # Sample a coarse grid: divide [-4.5, 4.5] into ~4 points per dim
        # That's 4^5 = 1024 points, too many. Use adaptive approach.
        
        # Use a smaller structured set: key points near 0 and at +/- pi
        # Actually, let's use random + structured mix
        n_phase1 = 150
        
        # Structured points: 0, +/-pi/2, +/-pi, +/-3pi/2 in each dim
        # But that's too many combinations. Use random sampling with bias toward 0.
        
        for i in range(1, n_phase1 + 1):
            if time.time() > deadline or calls_made >= budget:
                break
            # Mix: some uniform, some biased toward center
            if i % 3 == 0:
                # Uniform in [-4.5, 4.5]
                x = rng.uniform(-4.5, 4.5, dim)
            elif i % 3 == 1:
                # Biased toward 0 (Rastrigin minimum structure)
                x = rng.normal(0, 1.5, dim)
                x = np.clip(x, -4.5, 4.5)
            else:
                # Try points near multiples of pi
                base = rng.choice([-np.pi, 0, np.pi], size=dim)
                x = base + rng.normal(0, 0.5, dim)
                x = np.clip(x, -5, 5)
            counted_func(x)

        # Phase 2: Local refinement from best point and nearby points
        # Use Powell's method for derivative-free local optimization
        
        # Collect candidate starting points
        candidates = [best_x.copy()]
        
        # Add perturbed versions of best_x
        for sigma in [1.0, 0.5, 0.2, 0.1]:
            for _ in range(3):
                pert = best_x + rng.normal(0, sigma, dim)
                pert = np.clip(pert, -5, 5)
                candidates.append(pert)
        
        # Also try a few other good points from phase 1 (we don't track them,
        # so just use random points in the promising region)
        for _ in range(5):
            x = rng.normal(0, 1.0, dim)
            x = np.clip(x, -4, 4)
            candidates.append(x)
        
        # Sort candidates by distance to best_x (prefer closer ones first)
        # Actually, just try them in order
        
        max_local_runs = 8
        run_count = 0
        for x_start in candidates:
            if calls_made >= budget or time.time() > deadline - 15:
                break
            if run_count >= max_local_runs:
                break
            
            # Determine maxfev for this run based on remaining budget
            remaining = budget - calls_made
            maxfev = min(80, max(20, remaining // 2))
            
            try:
                res = minimize(
                    counted_func,
                    x_start,
                    method='Powell',
                    options={'maxiter': 50, 'maxfev': maxfev}
                )
            except Exception:
                pass
            
            run_count += 1
            
            # If we found a significantly better point, add more perturbations
            if run_count < max_local_runs and calls_made < budget:
                for sigma in [0.3, 0.1]:
                    pert = best_x + rng.normal(0, sigma, dim)
                    pert = np.clip(pert, -5, 5)
                    candidates.append(pert)
        
        # Phase 3: Final polish with Nelder-Mead if budget remains
        if calls_made < budget and time.time() < deadline - 5:
            remaining = budget - calls_made
            maxfev = min(50, remaining)
            try:
                res = minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 30, 'maxfev': maxfev, 'xatol': 1e-8, 'fatol': 1e-8}
                )
            except Exception:
                pass
        
        # Phase 4: If still have budget, try a few more random restarts
        while calls_made < budget and time.time() < deadline - 10:
            x = rng.normal(0, 0.5, dim)
            x = np.clip(x, -4, 4)
            counted_func(x)
            
            # Quick local refinement
            if calls_made < budget - 30:
                try:
                    minimize(
                        counted_func,
                        x,
                        method='Powell',
                        options={'maxiter': 20, 'maxfev': 30}
                    )
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
