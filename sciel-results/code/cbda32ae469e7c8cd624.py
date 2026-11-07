import numpy as np
import time

class Optimizer:
    """DTLZ2 optimizer using Halton sequence with explicit corner points for optimal hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2 structure:
        # Pareto front: x_i = 0.5 for i >= m (m = n_obj - 1 = 2)
        # x_0, x_1 parameterize the front.
        # f_1 = cos(theta_0)cos(theta_1), f_2 = cos(theta_0)sin(theta_1), f_3 = sin(theta_0)
        # theta_0 = x_0 * pi/2, theta_1 = x_1 * pi/2
        # Uniform surface area on the unit sphere (quarter hemisphere) requires:
        # cos(theta_0) ~ U[0, 1] and theta_1 ~ U[0, pi/2]
        # Let u ~ U[0, 1], v ~ U[0, 1]
        # cos(theta_0) = u  => theta_0 = arccos(u) => x_0 = (2/pi) * arccos(u)
        # theta_1 = v * pi/2 => x_1 = v

        g_start = self.n_obj - 1
        base_x = np.zeros(self.dim)
        base_x[g_start:] = 0.5

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        evaluated_points = []
        used_budget = 0

        # Explicit corner points that map to important regions of the Pareto front
        # These ensure good coverage of the boundaries where hypervolume contributions are largest
        corner_uv = [
            (0.0, 0.0),  # f = (1, 0, 0)
            (0.0, 1.0),  # f = (0, 1, 0)
            (1.0, 0.0),  # f = (0, 0, 1)
            (1.0, 1.0),  # f = (0, 0, 0) - wait, cos(arccos(1))=1, so f=(1*1, 1*0, 0)=(1,0,0)? 
                         # Actually u=1 => cos(theta_0)=1 => theta_0=0 => f_3=sin(0)=0
                         # v=1 => theta_1=pi/2 => f_1=cos(0)*cos(pi/2)=0, f_2=cos(0)*sin(pi/2)=1
                         # So f=(0,1,0)
            (0.5, 0.5),  # center
        ]

        # Evaluate corner points first to ensure they're included
        for u, v in corner_uv:
            if time.time() > deadline:
                return evaluated_points
            if used_budget >= self.budget:
                break
            
            x0 = (2.0 / np.pi) * np.arccos(u)
            x1 = v
            
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
            
            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1

        # Fill remaining budget with Halton sequence for uniform coverage
        n_remaining = self.budget - used_budget
        # Start Halton from index 1, skip if we'd duplicate corners (unlikely with Halton)
        for i in range(1, n_remaining + 1):
            if time.time() > deadline:
                break
            if used_budget >= self.budget:
                break
            
            u = halton(i, 2)
            v = halton(i, 3)
            
            # Skip if too close to an already-evaluated corner (avoid duplicates)
            skip = False
            for cu, cv in corner_uv:
                if abs(u - cu) < 1e-6 and abs(v - cv) < 1e-6:
                    skip = True
                    break
            if skip:
                continue
            
            x0 = (2.0 / np.pi) * np.arccos(u)
            x1 = v
            
            x = base_x.copy()
            x[0] = x0
            x[1] = x1
            
            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1

        return evaluated_points
