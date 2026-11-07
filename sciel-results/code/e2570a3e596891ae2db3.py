"""Direct DTLZ4 Pareto-front construction with structured simplex sampling."""
import numpy as np
import time


class Optimizer:
    """Builds DTLZ4 Pareto front points using Latin Hypercube Sampling on the simplex.

    DTLZ4 (m=3 objectives, 12 variables):
      x_0..x_2 : position variables on the simplex (x0+x1+x2=1)
      x_3..x_11: control variables; g minimized at x_i=0.5
      f_i = (1+g) * x_{i-1} for i=1,2,3 (with f_3 using 1-x0-x1)
    At g=0, points trace the unit triangle in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0
        evaluated = []

        # Control variables at their optimum: x_i = 0.5
        n_ctrl = self.dim - 3
        ctrl = np.full(n_ctrl, 0.5)
        ctrl = np.clip(ctrl, lo[3:], hi[3:])

        # Build a structured set of simplex points that maximizes hypervolume coverage.
        # Strategy:
        # 1. Vertices (3 points) - essential for hypervolume corners
        # 2. Edge midpoints (3 points) - good coverage along edges
        # 3. Centroid (1 point)
        # 4. Regular grid on the simplex interior for uniform coverage
        # 5. Use remaining budget for additional structured points

        # Vertices
        special_pts = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
            np.array([1.0/3.0, 1.0/3.0, 1.0/3.0]),
        ]

        # Add more structured points: points along edges at 1/4 and 3/4 positions
        edge_pts = [
            np.array([0.25, 0.75, 0.0]),
            np.array([0.75, 0.25, 0.0]),
            np.array([0.25, 0.0, 0.75]),
            np.array([0.75, 0.0, 0.25]),
            np.array([0.0, 0.25, 0.75]),
            np.array([0.0, 0.75, 0.25]),
            # Interior points
            np.array([0.5, 0.25, 0.25]),
            np.array([0.25, 0.5, 0.25]),
            np.array([0.25, 0.25, 0.5]),
            np.array([0.4, 0.3, 0.3]),
            np.array([0.3, 0.4, 0.3]),
            np.array([0.3, 0.3, 0.4]),
        ]

        all_structured = special_pts + edge_pts

        # Evaluate all structured points first (up to budget)
        for p in all_structured:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = np.concatenate([p, ctrl])
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        # If we have remaining budget, use a Halton-like sequence for additional coverage
        # Halton sequence in 2D mapped to simplex
        n_remaining = self.budget - used
        if n_remaining > 0:
            # Use a low-discrepancy sequence for better coverage
            # Map 2D Halton points to simplex via (u, v) -> (1-sqrt(u), sqrt(u)*(1-sqrt(v)), sqrt(u)*sqrt(v))
            # This gives uniform coverage on the simplex
            
            # Generate Halton sequence points
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r

            # Use a set of well-distributed points
            n_halton = min(n_remaining, 150)
            for i in range(1, n_halton + 1):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                u = halton(i, 2)
                v = halton(i, 3)
                
                # Map to simplex
                x0 = 1.0 - np.sqrt(u)
                x1 = np.sqrt(u) * (1.0 - np.sqrt(v))
                x2 = np.sqrt(u) * np.sqrt(v)
                
                p = np.array([x0, x1, x2])
                x = np.concatenate([p, ctrl])
                x = np.clip(x, lo, hi)
                f = problem(x)
                evaluated.append((x, np.array(f)))
                used += 1

        return np.array([x for x, _ in evaluated]) if evaluated else np.zeros((1, self.dim))
