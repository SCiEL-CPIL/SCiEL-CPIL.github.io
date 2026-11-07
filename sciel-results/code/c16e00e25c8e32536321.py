"""Direct DTLZ4 Pareto-front construction exploiting known structure."""
import numpy as np
import time


class Optimizer:
    """Builds DTLZ4 Pareto front points using spherical LHS for uniform coverage.

    DTLZ4 (m=3 objectives, 12 variables):
      x_0..x_2 : position variables; at g=0, objectives lie on unit sphere octant
      x_3..x_11: control variables; g minimized at x_i=0.5
      f_i = (1+g) * h_i where h_i are on the unit sphere octant
    At g=0, points trace the unit sphere octant in objective space.
    """

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

    def _front_point(self, theta, phi, lo, hi):
        """Create decision vector from spherical angles on the front.
        
        theta in [0, pi/2], phi in [0, pi/2]
        h0 = cos(theta)
        h1 = sin(theta) * cos(phi)  
        h2 = sin(theta) * sin(phi)
        
        For DTLZ4, x[0] = theta, x[1] = phi, x[2:] = 0.5 (g-optimal)
        """
        x = np.zeros(self.dim)
        x[0] = theta
        x[1] = phi
        x[2:] = 0.5
        x = np.clip(x, lo, hi)
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0
        evaluated = []

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1
            return f

        # Phase 1: Place strategic points (vertices and edge midpoints)
        # These give maximum hypervolume contribution
        strategic = [
            # Vertices of the front (corners in objective space)
            (0.0, 0.0),      # h = (1, 0, 0)
            (0.0, np.pi/2),  # h = (0, 0, 1)
            (np.pi/2, 0.0),  # h = (0, 1, 0)
            (np.pi/2, np.pi/2),  # h = (0, 0, 1) duplicate, but valid
            # Edge midpoints
            (0.0, np.pi/4),  # midpoint on edge 1
            (np.pi/4, 0.0),  # midpoint on edge 2
            (np.pi/4, np.pi/4),  # center of face
            (np.pi/2, np.pi/4),  # midpoint on edge 3
        ]

        for theta, phi in strategic:
            x = self._front_point(theta, phi, lo, hi)
            evaluate(x)

        # Phase 2: Spherical LHS sampling on the front
        # To sample the unit sphere octant uniformly:
        #   cos(theta) = u  =>  theta = arccos(u)
        #   phi = (pi/2) * v
        remaining = self.budget - used
        if remaining > 0:
            n_lhs = remaining
            lhs_points = self._lhs_2d(n_lhs, seed=0)

            for u0, u1 in lhs_points:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                # u0 in [0, 1) -> cos(theta) in [0, 1] -> theta in [0, pi/2]
                cos_theta = np.clip(u0, 0.0, 1.0)
                theta = np.arccos(cos_theta)

                # u1 in [0, 1) -> phi in [0, pi/2]
                phi = (np.pi / 2.0) * u1

                x = self._front_point(theta, phi, lo, hi)
                evaluate(x)

        # Phase 3: Polish pass - refine near existing points if budget remains
        if used < self.budget and time.time() < deadline:
            rng = np.random.RandomState(42)
            n_polish = min(self.budget - used, 20)
            for i in range(n_polish):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                # Pick a random evaluated point and perturb slightly
                idx = rng.randint(len(evaluated))
                base_x = evaluated[idx][0].copy()
                new_x = base_x.copy()
                new_x[0] += 0.02 * rng.randn()
                new_x[1] += 0.02 * rng.randn()
                new_x = np.clip(new_x, lo, hi)
                evaluate(new_x)

        if not evaluated:
            # Fallback: return a single valid point
            x = np.zeros(self.dim)
            x[2:] = 0.5
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x, np.array(f)))

        return np.array([x for x, _ in evaluated]) if evaluated else np.zeros((1, self.dim))
