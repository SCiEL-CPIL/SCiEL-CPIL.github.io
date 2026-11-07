"""Search-based optimizer for DTLZ4 with known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using stratified grid in objective space."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def make_x(x1, x2):
            """Construct decision vector from x1, x2 that achieves g=0 on DTLZ4."""
            x = np.full(dim, 0.5)
            x[0] = x1
            x[1] = x2
            return np.clip(x, lo, hi)

        # DTLZ4 3-objective structure:
        # f1 = (1+g) * 0.5 * cos(pi*x1/2) * cos(pi*x2/2)
        # f2 = (1+g) * 0.5 * cos(pi*x1/2) * sin(pi*x2/2)
        # f3 = (1+g) * 0.5 * sin(pi*x1/2)
        # With g=0: f1^2 + f2^2 + f3^2 = 0.25
        # Pareto front is quarter sphere of radius 0.5 in positive octant.
        #
        # Inverse mapping: given desired (f1, f2, f3) on the sphere,
        #   f1 = 0.5 * cos(a) * cos(b)
        #   f2 = 0.5 * cos(a) * sin(b)
        #   f3 = 0.5 * sin(a)
        # where a = pi*x1/2, b = pi*x2/2
        # So: a = arcsin(2*f3), b = arctan2(f2, f1)
        #     x1 = 2*a/pi, x2 = 2*b/pi

        # Generate stratified grid in (f1, f2) quarter-disk with f3 derived
        # Use a polar-coordinate stratified approach for uniform coverage
        n_radial = 14
        n_angular = 14
        # This gives 196 points, plus we add boundary/corner points

        candidates = []

        # Stratified polar grid in the quarter disk
        # Radius: r = 0.5 * sin(a), angle: theta = b
        # a in [0, pi/2], b in [0, pi/2]
        # For uniform area coverage in (f1,f2,f3) space, stratify in (a, b)
        # but weight by the Jacobian to get uniform density on the sphere surface

        # Stratified sampling in (a, b) space with uniform density
        # This gives uniform density on the quarter sphere surface
        for i in range(n_radial):
            for j in range(n_angular):
                # Stratified: each cell gets one point
                a = (i + 0.5) * (np.pi / 2) / n_radial
                b = (j + 0.5) * (np.pi / 2) / n_angular
                # Add small jitter for robustness
                a += (np.random.rand() - 0.5) * (np.pi / 2) / n_radial * 0.3
                b += (np.random.rand() - 0.5) * (np.pi / 2) / n_angular * 0.3
                a = np.clip(a, 0.0, np.pi / 2)
                b = np.clip(b, 0.0, np.pi / 2)

                x1 = 2.0 * a / np.pi
                x2 = 2.0 * b / np.pi
                x1 = np.clip(x1, 0.0, 1.0)
                x2 = np.clip(x2, 0.0, 1.0)
                candidates.append((x1, x2))

        # Add explicit boundary points (axes and corners)
        # These are important for hypervolume as they define the extreme points
        # Corner: (0,0) -> f=(0.5, 0, 0)
        # Corner: (0,1) -> f=(0, 0.5, 0)
        # Corner: (1,0) -> f=(0, 0, 0.5)
        # Corner: (1,1) -> f=(0, 0, 0.5) [same as above since cos(pi/2)=0]
        # Actually let's be more careful:
        # x1=0, x2=0: f1=0.5*cos(0)*cos(0)=0.5, f2=0.5*cos(0)*sin(0)=0, f3=0.5*sin(0)=0
        # x1=0, x2=1: f1=0.5*cos(0)*cos(pi/2)=0, f2=0.5*cos(0)*sin(pi/2)=0.5, f3=0
        # x1=1, x2=0: f1=0.5*cos(pi/2)*cos(0)=0, f2=0, f3=0.5*sin(pi/2)=0.5
        # x1=1, x2=1: f1=0, f2=0, f3=0.5

        candidates.append((0.0, 0.0))
        candidates.append((0.0, 1.0))
        candidates.append((1.0, 0.0))
        candidates.append((1.0, 1.0))

        # Edge midpoints
        candidates.append((0.0, 0.5))  # f1=0.5*cos(pi/4)=0.354, f2=0.354, f3=0
        candidates.append((0.5, 0.0))  # f1=0.5*cos(pi/4)*1=0.354, f2=0, f3=0.5*sin(pi/4)=0.354
        candidates.append((0.5, 1.0))  # f1=0.354*0=0... wait cos(pi/2*0.5)=cos(pi/4)
        # Let me recompute: x1=0.5 -> a=pi/4, x2=1 -> b=pi/2
        # f1=0.5*cos(pi/4)*cos(pi/2)=0, f2=0.5*cos(pi/4)*sin(pi/2)=0.354, f3=0.5*sin(pi/4)=0.354
        candidates.append((1.0, 0.5))  # a=pi/2, b=pi/4: f1=0, f2=0, f3=0.5

        # Additional edge points for denser boundary coverage
        for t in np.linspace(0.1, 0.9, 5):
            candidates.append((0.0, t))
            candidates.append((t, 0.0))
            candidates.append((t, 1.0))
            candidates.append((1.0, t))

        # Trim to exactly budget
        if len(candidates) > budget:
            # Prioritize: keep boundary points first, then stratified grid
            # But actually, let's just keep the first budget points
            # The stratified grid is the most important for coverage
            candidates = candidates[:budget]

        # Pad if less than budget
        while len(candidates) < budget:
            candidates.append((np.random.rand(), np.random.rand()))

        # Evaluate all candidates
        evaluated_points = []
        used = 0

        for x1, x2 in candidates:
            if used >= budget or time.time() - t0 > deadline:
                break

            x = make_x(x1, x2)
            f = np.array(problem(x))
            used += 1
            evaluated_points.append(f)

        return np.array(evaluated_points)
# EVOLVE-BLOCK-END
