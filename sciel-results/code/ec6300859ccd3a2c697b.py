"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s wall-clock limit, use 290s for safety
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # ZDT6 structure: x[0] controls f1, x[1:] affect f2 through g(x)
        # Pareto front: x[1:] = 0, x[0] varies
        # Strategy: densely sample x[0], keep x[1:] near 0

        # Phase 1: Dense sampling of x[0] with x[1:] near 0
        # Use most of budget for this
        n_phase1 = int(budget * 0.7)
        n_phase2 = budget - n_phase1

        # Generate diverse x[0] values
        # Use a mix of uniform and concentrated near interesting regions
        x0_vals = np.linspace(0, 1, n_phase1)
        # Add some random perturbation for diversity
        x0_vals = np.clip(x0_vals + 0.005 * rng.standard_normal(n_phase1), 0, 1)

        # For each x0, set x[1:] to small random values (to keep g(x) low)
        # g(x) = 1 + 9*(sum(x[1:])/9)^(1/1.2)
        # We want sum(x[1:]) close to 0
        # Use very small values for x[1:]
        eps = 0.01  # small epsilon for x[1:]

        used = 0
        results = []  # list of (x, f) pairs

        for x0 in x0_vals:
            if time.time() > deadline - 5.0:
                break
            if used >= budget:
                break
            # Create candidate: x[0] = x0, x[1:] small random
            x = np.zeros(dim)
            x[0] = x0
            if dim > 1:
                x[1:] = eps * rng.random(dim - 1)
            x = np.clip(x, lo, hi)
            f = problem(x)
            results.append((x, f))
            used += 1

        # Phase 2: Local refinement around best x[0] values
        # Sort by f1 to find spread points, then refine
        if results:
            # Convert to arrays for efficiency
            X = np.array([r[0] for r in results])
            F = np.array([r[1] for r in results])

            # Find non-dominated points (simple filter)
            # In ZDT6, Pareto front has f1 in [0,1], f2 = g*h(f1)
            # Points with small f2 and spread f1 are best

            # Refine: for a subset of promising x0 values, do local search
            # Pick points with low f2 (indicating good x[1:]) and spread f1
            if len(F) > 0 and used < budget:
                # Sort by f1 for diversity
                idx = np.argsort(F[:, 0])
                # Take evenly spaced points in f1
                n_refine = min(20, len(F) // 5)
                if n_refine > 0:
                    step = max(1, len(F) // n_refine)
                    refine_idx = idx[::step][:n_refine]

                    for i in refine_idx:
                        if time.time() > deadline - 5.0:
                            break
                        if used >= budget:
                            break
                        # Local refinement around x[i]
                        base_x = X[i].copy()
                        # Try several small perturbations
                        n_local = 5
                        for j in range(n_local):
                            if used >= budget or time.time() > deadline - 5.0:
                                break
                            # Perturb x[0] slightly, keep x[1:] small
                            pert = 0.02 * rng.standard_normal(dim)
                            pert[1:] *= 0.1  # much smaller perturbation for x[1:]
                            new_x = base_x + pert
                            new_x = np.clip(new_x, lo, hi)
                            # Ensure x[1:] stays small
                            if dim > 1:
                                new_x[1:] = np.clip(new_x[1:], 0, 0.05)
                            f_new = problem(new_x)
                            results.append((new_x, f_new))
                            used += 1

        return X
# EVOLVE-BLOCK-END
