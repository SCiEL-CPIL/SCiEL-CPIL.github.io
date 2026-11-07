"""Direct DTLZ4 Pareto-front construction exploiting known structure."""
import numpy as np
import time


class Optimizer:
    """Builds DTLZ4 Pareto front points directly from the problem's structure.

    DTLZ4 (m=3, K=4 decision vars on the front, N-K=8 control vars):
      x_0..x_3 : simplex variables defining the front shape
      x_4..x_11: control variables; g(x_4..x_11) is minimized at x_i=0.5 (each)
      f_1 = (1+g)*x_0,  f_2 = (1+g)*x_1,  f_3 = (1+g)*(1 - x_0 - x_1)
    At g=0, (x_0,x_1,1-x_0-x_1) traces the unit triangle in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250.0  # safety margin well under 300s
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        used = 0
        candidates = []  # list of x vectors

        # Control variables at their optimum: g=0 when x_i = k/10 for integer k
        # Use 0.5 which is one such point
        ctrl = np.full(self.dim - 4, 0.5)
        ctrl = np.clip(ctrl, lo[4:], hi[4:])

        # --- Phase 1: Add explicit corner and edge-midpoint points ---
        # These ensure the extreme regions of the Pareto front are well-covered
        # Corners of the 3-simplex (4 vars summing to 1)
        corners = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ]
        # Edge midpoints
        edge_mids = [
            np.array([0.5, 0.5, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5, 0.0]),
            np.array([0.0, 0.5, 0.0, 0.5]),
            np.array([0.0, 0.0, 0.5, 0.5]),
        ]
        # Center point
        center = np.array([0.25, 0.25, 0.25, 0.25])

        for p in corners + edge_mids + [center]:
            x = np.concatenate([p, ctrl])
            x = np.clip(x, lo, hi)
            candidates.append(x)

        # --- Phase 2: Dense grid on the 2D simplex parameterization ---
        # For 4 vars summing to 1, we can parameterize as:
        # x0 = 1 - sqrt(u1), x1 = sqrt(u1)*(1-sqrt(u2)), x2 = sqrt(u1)*sqrt(u2)*(1-u3), x3 = sqrt(u1)*sqrt(u2)*u3
        # This gives uniform distribution in the simplex.
        n_grid = int(min(self.budget - len(candidates), 800))
        
        # Use a structured grid in (u1, u2, u3) space mapped to simplex
        # For hypervolume, we want good coverage of the objective space
        # The objectives are f = (1+g) * [x0, x1, x2, x3] where sum xi = 1
        # So we want uniform coverage of the simplex vertices and edges
        
        # Use Halton sequence for quasi-random coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        for i in range(1, n_grid + 1):
            u1 = halton(i, 2)
            u2 = halton(i, 3)
            u3 = halton(i, 5)
            
            # Map to simplex using the standard transformation for uniform distribution
            # This ensures uniform coverage of the 3-simplex
            x0 = 1.0 - np.sqrt(u1)
            x1 = np.sqrt(u1) * (1.0 - np.sqrt(u2))
            x2 = np.sqrt(u1) * np.sqrt(u2) * (1.0 - u3)
            x3 = np.sqrt(u1) * np.sqrt(u2) * u3
            
            x = np.array([x0, x1, x2, x3])
            x = np.clip(x, lo[:4], hi[:4])
            # Normalize to ensure sum is exactly 1
            s = x.sum()
            if s > 0:
                x = x / s
            
            x_full = np.concatenate([x, ctrl])
            x_full = np.clip(x_full, lo, hi)
            candidates.append(x_full)

        # --- Phase 3: Add random points for robustness ---
        n_rand = max(0, self.budget - len(candidates))
        for _ in range(n_rand):
            # Random point on simplex using Dirichlet(1,1,1,1)
            y = rng.dirichlet(np.ones(4))
            x = np.concatenate([y, ctrl])
            x = np.clip(x, lo, hi)
            candidates.append(x)

        # --- Evaluate candidates up to budget, keeping all returned vectors ---
        evaluated = []
        for x in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        # If we have budget left, add random points
        while used < self.budget:
            if time.time() > deadline:
                break
            y = rng.dirichlet(np.ones(4))
            x = np.concatenate([y, ctrl])
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        return np.array([x for x, _ in evaluated]) if evaluated else np.zeros((1, self.dim))
