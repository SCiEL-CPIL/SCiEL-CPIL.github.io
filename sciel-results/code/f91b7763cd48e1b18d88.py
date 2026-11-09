"""Hill-Climbing Vertex optimizer for the BBOB-style linear function f5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Vertex-search optimizer: f5 is linear, so its box minimum is a vertex.

    Strategy:
      1. Evaluate the center to get a baseline.
      2. Probe each coordinate with a large step to determine the sign of the
         gradient (direction of descent) for that axis.
      3. Immediately evaluate the vertex implied by those signs (call 6).
         This is the exact global minimum of a linear function on the box,
         so the best-so-far collapses to ~0 error very early (good for AOCC).
      4. Spend remaining budget polishing / confirming with small perturbations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit

        # Center of the box
        center = np.zeros(dim)
        fx = func(center)
        best_x = center.copy()
        best_f = fx
        calls = 1

        # --- Phase 1: determine gradient sign per coordinate (5 probes) ---
        # For a linear f, f(center + s*e_i) vs f(center) tells us the sign of
        # the partial derivative along axis i. We use a large probe step so the
        # comparison is robust to the offset/rotation.
        probe = 4.0
        signs = np.zeros(dim)
        for i in range(dim):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            p = center.copy()
            p[i] += probe
            p = np.clip(p, -5, 5)
            fp = func(p)
            calls += 1
            # if moving +probe lowers f, gradient along i is negative -> go +
            if fp < fx:
                signs[i] = 1.0
            else:
                signs[i] = -1.0

        # --- Phase 2: evaluate the implied vertex (the exact linear min) ---
        if calls < budget:
            vertex = signs * 5.0  # each coord at its best extreme
            vertex = np.clip(vertex, -5, 5)
            fv = func(vertex)
            calls += 1
            if fv < best_f:
                best_f = fv
                best_x = vertex.copy()

        # --- Phase 3: polish with small perturbations around the best point ---
        sigma = 0.5
        for _ in range(budget - calls):
            if time.time() > deadline:
                break
            y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                sigma *= 1.3
            else:
                sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
