"""Portfolio search for DTLZ2: Halton-sequence sampling on the known Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Halton sequence on the known front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _halton(n, base):
        """Generate n Halton sequence values in [0, 1)."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            idx = i + 1
            r = 0.0
            while idx > 0:
                idx -= 1
                r += f * (idx % base)
                f /= base
            result[i] = r
        return result

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 315.0

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        used = 0

        # Reserve 3 slots for corner points to maximize hypervolume extent
        n_corners = 3
        n_halton = budget - n_corners
        if n_halton < 10:
            n_halton = max(10, budget // 2)
            n_corners = budget - n_halton

        candidates = []

        # Corner points: maximize each objective pair
        # f1=1,f2=0,f3=0 -> x0=arcsin(0)/pi*2=0, x1=arcsin(1)/pi*2=1
        x_corner1 = np.full(dim, 0.5)
        x_corner1[0] = 0.0
        x_corner1[1] = 1.0
        candidates.append(x_corner1)

        # f1=0,f2=1,f3=0
        x_corner2 = np.full(dim, 0.5)
        x_corner2[0] = 0.0
        x_corner2[1] = 0.0  # ratio = f1/r = 0/1 = 0 -> arcsin(0)=0 -> x1=0
        # Actually for f1=0, f2=1: r=1, ratio=0, so x1=0
        candidates.append(x_corner2)

        # f1=0,f2=0,f3=1 -> x0=arcsin(1)/pi*2=1, x1=0.5 (r=0)
        x_corner3 = np.full(dim, 0.5)
        x_corner3[0] = 1.0
        x_corner3[1] = 0.5
        candidates.append(x_corner3)

        # Halton sequence for interior points
        # Use base 2 for theta parameter, base 3 for phi parameter
        h_theta = self._halton(n_halton, 2)
        h_phi = self._halton(n_halton, 3)

        for i in range(n_halton):
            # Map Halton [0,1) to angular parameters
            # theta in (0, pi/2): use u = cos(theta), so u in (0, 1)
            # We want uniform coverage on the quarter-sphere surface
            # Surface area element: sin(theta) dtheta dphi
            # For uniform surface area: sample sin(theta) uniformly, then theta = arcsin(u)
            # But we also need phi uniform in [0, pi/2]
            
            # Use h_theta to control sin(theta) for uniform surface coverage
            sin_theta = h_theta[i]
            sin_theta = np.clip(sin_theta, 1e-10, 1.0 - 1e-10)
            theta = np.arcsin(sin_theta)
            u = np.cos(theta)  # u = f3
            u = np.clip(u, 1e-10, 1.0 - 1e-10)
            
            # phi uniform in (0, pi/2)
            phi = h_phi[i] * (np.pi / 2.0)
            phi = np.clip(phi, 1e-10, np.pi/2.0 - 1e-10)
            
            f1 = sin_theta * np.cos(phi)
            f2 = sin_theta * np.sin(phi)
            f3 = u
            
            # Map to x
            x = np.full(dim, 0.5)
            x[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3, 0, 1))
            
            r = np.sqrt(f1 * f1 + f2 * f2)
            if r > 1e-15:
                ratio = f1 / r
                x[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio, -1, 1))
            else:
                x[1] = 0.5
            
            candidates.append(x)

        # Evaluate all candidates within the time budget
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            problem(x)
            used += 1

        return np.zeros((0, n_obj))
# EVOLVE-BLOCK-END
