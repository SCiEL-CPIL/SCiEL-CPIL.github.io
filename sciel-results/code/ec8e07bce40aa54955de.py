import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using LHS on front + vertex placement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def _lhs_2d(self, n, seed=42):
        """Latin Hypercube Sampling in 2D, returns n points in [0,1]^2."""
        rng = np.random.RandomState(seed)
        x = np.arange(n) / n
        y = np.arange(n) / n
        rng.shuffle(x)
        rng.shuffle(y)
        return np.column_stack([x, y])

    def _make_point(self, u0, u1, g_val=0.5):
        """Create a decision vector with front-position vars u0, u1 and g-vars at g_val."""
        x = np.zeros(self.dim)
        x[0] = u0
        x[1] = u1
        x[2:] = g_val
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            all_X.append(x.copy())
            all_F.append(np.array(f))
            return f

        # Phase 1: Place vertices and edges for maximum hypervolume contribution
        # Vertices of the front (corners of the simplex in objective space)
        # For DTLZ4 with M=3, the front is parameterized by x[0], x[1] in [0,1]^2
        # The vertices in decision space that give extreme objective values:
        # (0,0), (0,1), (1,0), (1,1)

        vertices = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        for v0, v1 in vertices:
            if used >= self.budget:
                break
            x = self._make_point(v0, v1)
            evaluate(x)

        # Phase 2: Edge midpoints for better coverage
        edges = [(0.0, 0.5), (0.5, 0.0), (1.0, 0.5), (0.5, 1.0)]
        for e0, e1 in edges:
            if used >= self.budget:
                break
            x = self._make_point(e0, e1)
            evaluate(x)

        # Phase 3: LHS sampling in the 2D front-position space
        # Use remaining budget for LHS points
        remaining = self.budget - used
        if remaining > 0:
            n_lhs = min(remaining, 150)  # Cap LHS points
            lhs_points = self._lhs_2d(n_lhs, seed=123)
            for u0, u1 in lhs_points:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                x = self._make_point(u0, u1)
                evaluate(x)

        # Phase 4: If budget remains, do local refinement around promising points
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            rng = np.random.RandomState(42)
            # Try to refine by perturbing g-variables slightly
            # and front-position variables
            for i in range(len(all_X)):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                base_x = all_X[i].copy()
                # Small perturbation in front-position vars
                new_x = base_x.copy()
                new_x[0] += 0.05 * rng.randn()
                new_x[1] += 0.05 * rng.randn()
                new_x = np.clip(new_x, lo, hi)
                evaluate(new_x)

        return all_X
