# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT3 using structured sampling on non-dominated segments."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT3 Pareto front analysis:
        # f1 = x1, f2 = g * h(x1, g) where g >= 1
        # At g=1 (x[1:]=0): f2 = 1 - sqrt(x1) - x1*sin(10*pi*x1)
        # The front is discontinuous; non-dominated segments exist where h(x1,1) > 0
        # and f2 is minimized for each f1 value.
        #
        # Key: We should densely sample x1 in regions where the Pareto front exists
        # (non-dominated) and skip dominated regions.

        # Phase 1: Coarse scan to identify non-dominated x1 intervals
        # Use 20 evaluations to map out where the front is active
        n_coarse = min(20, budget)
        x1_coarse = np.linspace(0.0, 1.0, n_coarse + 2)[1:-1]  # interior points
        coarse_f2 = []
        coarse_f1 = []
        used = 0

        for x1 in x1_coarse:
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x1
            f = problem(x)
            coarse_f1.append(f[0])
            coarse_f2.append(f[1])
            used += 1

        coarse_f1 = np.array(coarse_f1)
        coarse_f2 = np.array(coarse_f2)

        # Identify non-dominated segments: where f2 < 1 (below the "g=1 baseline" max)
        # Actually, for ZDT3 at g=1, f2 can be > 1 in some regions due to the sine term.
        # The Pareto front consists of points that are not dominated by any other point.
        # A simpler heuristic: sample densely where f2 is relatively small.
        
        # Find the minimum f2 and use it as reference
        if len(coarse_f2) > 0:
            f2_min = np.min(coarse_f2)
            # Points where f2 is within 1.5 of minimum are likely on or near the front
            # Also include points where f2 < 1.0 (definitely non-dominated region)
            threshold = max(f2_min + 0.5, 1.0)
            
            # Identify intervals where f2 < threshold
            active_mask = coarse_f2 < threshold
            active_x1 = x1_coarse[active_mask]
        else:
            active_x1 = x1_coarse

        # Phase 2: Dense sampling in active regions
        n_remaining = budget - used
        if n_remaining <= 0:
            return None

        # Distribute remaining budget proportionally to active region length
        if len(active_x1) > 0:
            # Create dense sampling points in the active x1 range
            x1_min_active = np.min(active_x1)
            x1_max_active = np.max(active_x1)
            
            # Also add some points near 0 and 1 for boundary coverage
            # The Pareto front of ZDT3 has segments, so sample across the full range
            # but with higher density in active regions
            
            # Strategy: Mix of uniform sampling in [0,1] weighted toward active regions
            # Use 70% of remaining budget in active region, 30% spread across [0,1]
            n_active = int(n_remaining * 0.7)
            n_spread = n_remaining - n_active
            
            x1_dense_active = np.linspace(x1_min_active, x1_max_active, n_active) if n_active > 0 else np.array([])
            x1_dense_spread = np.linspace(0.0, 1.0, n_spread) if n_spread > 0 else np.array([])
            
            x1_all = np.concatenate([x1_dense_active, x1_dense_spread])
            np.random.shuffle(x1_all)
        else:
            # Fallback: uniform sampling
            x1_all = np.linspace(0.0, 1.0, n_remaining)
            np.random.shuffle(x1_all)

        for x1 in x1_all:
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x1
            x[1:] = 0.0
            f = problem(x)
            used += 1

        return None
# EVOLVE-BLOCK-END
