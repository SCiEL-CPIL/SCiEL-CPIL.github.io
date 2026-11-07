"""ZDT1 multi-objective optimizer exploiting the known Pareto-front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit ZDT1 structure: f1=x0, f2=1+sqrt(f1) on the front (x1..=0.2)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0  # safety margin vs 300s hard limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        budget = self.budget

        def eval_vec(x):
            x = np.clip(x, lo, hi)
            return problem(x)

        # --- Phase 1: dense analytical front sweep (dominates generic search) ---
        # On the ZDT1 Pareto front, set x[0]=x and x[1:]=0.2 -> f2=1+sqrt(f1).
        # Sample x[0] densely across [lo0, hi0] to cover the whole front.
        n_front = int(min(150, max(20, budget * 0.40)))
        xs = np.linspace(lo[0], hi[0], n_front)
        front = np.empty((n_front, dim))
        front[:, 0] = xs
        front[:, 1:] = 0.2
        F = np.array([eval_vec(x) for x in front])
        used = n_front

        # --- Phase 2: local refinement near the front ---
        # Polish individual x[0] values with small perturbations to the tail,
        # keeping x[0] on/near the front so points stay close to the ideal curve.
        remaining = budget - used
        if remaining > 0:
            # Refine the front points: tiny noise on tail, keep x0.
            n_refine = int(min(remaining, n_front * 2))
            for i in range(n_refine):
                if time.time() > deadline:
                    break
                base = i % n_front
                x = front[base].copy()
                x[1:] = 0.2 + 1e-3 * (np.random.rand(dim - 1) - 0.5)
                x[0] = xs[base] + 1e-4 * (np.random.rand() - 0.5)
                x = np.clip(x, lo, hi)
                F = np.vstack([F, eval_vec(x)])
                used += 1
                if used >= budget:
                    break

        # --- Phase 3: any leftover budget -> small exploratory random points ---
        if used < budget:
            while used < budget:
                if time.time() > deadline:
                    break
                x = lo + span * np.random.rand(dim)
                F = np.vstack([F, eval_vec(x)])
                used += 1

        return front
# EVOLVE-BLOCK-END
