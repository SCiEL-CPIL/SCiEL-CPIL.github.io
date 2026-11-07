# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2.
    
    Strategy:
    - DTLZ2 Pareto front for 3 objectives is a quarter-sphere in the positive octant.
    - Distance variables (last dim - n_obj + 1 = 10 vars) set to 0.5 for optimality.
    - Angle variables (first n_obj - 1 = 2 vars) determine position on Pareto front.
    - Use deterministic low-discrepancy sequence (Halton) to sample 2D angle space
      maximally space-filling, ensuring good hypervolume coverage.
    - Additionally, use a small portion of budget for local refinement around
      promising regions to capture any residual sub-optimality.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, n, dim):
        """Generate n points from the Halton sequence in [0,1)^dim."""
        points = []
        for i in range(1, n + 1):
            pt = []
            for base in range(2, dim + 2):  # bases 2, 3, ...
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    f /= base
                    r += f * (i_val % base)
                    i_val //= base
                pt.append(r)
            points.append(pt)
        return np.array(points)

    def _dtlz2_objectives_from_angles(self, angles):
        """
        Compute DTLZ2 objectives from angle variables.
        For DTLZ2 with n_obj objectives:
        - g = sum((x_i - 0.5)^2) for i = n_obj-1 to dim-1 (distance vars)
        - f_i = (1 + g) * prod_{j<i} cos(theta_j) * sin(theta_i) for i < n_obj-1
        - f_{n_obj-1} = (1+g) * prod_{j<n_obj-1} sin(theta_j)
        
        If distance vars are set to 0.5, g=0, so:
        - f_0 = cos(theta_0) * sin(theta_1)
        - f_1 = cos(theta_0) * cos(theta_1) * sin(theta_2)  [wait, let me recheck]
        
        Actually for DTLZ2:
        f_i = (1+g) * prod_{j=0}^{i-1} cos(theta_j) * sin(theta_i) for i = 0..n_obj-2
        f_{n_obj-1} = (1+g) * prod_{j=0}^{n_obj-2} sin(theta_j)
        
        For 3 objectives (n_obj=3):
        f_0 = (1+g) * sin(theta_1)  [since prod over empty set = 1, and i=0: cos of nothing * sin(theta_0)?]
        
        Let me be more careful. DTLZ2 definition:
        f_i(x) = (1 + g(x_{M+1})) * prod_{j=0}^{i-1} cos(x_j * pi/2) * sin(x_i * pi/2)  for i = 0..M-1
        f_M(x) = (1 + g(x_{M+1})) * prod_{j=0}^{M-1} sin(x_j * pi/2)
        
        where M = n_obj - 1 = 2, and x_0, x_1 are the angle variables in [0,1].
        
        So for 3 objectives:
        f_0 = (1+g) * sin(x_1 * pi/2)  [prod over j=0 to -1 is 1]
        Wait, that doesn't look right. Let me re-read.
        
        Standard DTLZ2:
        f_i = (1+g) * prod_{j=0}^{i-1} cos(pi*x_j/2) * sin(pi*x_i/2)  for i=0,...,M-1
        f_M = (1+g) * prod_{j=0}^{M-1} sin(pi*x_j/2)
        
        For M=2 (3 objectives):
        f_0 = (1+g) * sin(pi*x_0/2)  [prod over j=0 to -1 is empty = 1]
        f_1 = (1+g) * cos(pi*x_0/2) * sin(pi*x_1/2)
        f_2 = (1+g) * cos(pi*x_0/2) * cos(pi*x_1/2)
        
        Hmm, but this gives f_0 + f_1^2 + f_2^2 = (1+g)^2 which isn't unit sphere.
        
        Let me look at the standard formulation more carefully.
        
        Actually, I think the correct DTLZ2 for M+1 objectives is:
        f_i = (1+g) * prod_{j=0}^{i-1} cos(pi*x_j/2) * sin(pi*x_i/2)  for i=0,...,M-1
        f_M = (1+g) * prod_{j=0}^{M-1} sin(pi*x_j/2)
        
        For M=2:
        f_0 = (1+g) * sin(pi*x_0/2)
        f_1 = (1+g) * cos(pi*x_0/2) * sin(pi*x_1/2)
        f_2 = (1+g) * cos(pi*x_0/2) * cos(pi*x_1/2)
        
        Sum of squares: (1+g)^2 [sin^2(x0) + cos^2(x0)*sin^2(x1) + cos^2(x0)*cos^2(x1)]
        = (1+g)^2 [sin^2(x0) + cos^2(x0)(sin^2(x1)+cos^2(x1))]
        = (1+g)^2 [sin^2(x0) + cos^2(x0)]
        = (1+g)^2
        
        So the Pareto front when g=0 is a unit sphere in the positive octant, parameterized by x_0, x_1 in [0,1].
        """
        # angles is shape (n, 2) with values in [0,1]
        x0 = angles[:, 0] * np.pi / 2
        x1 = angles[:, 1] * np.pi / 2
        c0 = np.cos(x0)
        s0 = np.sin(x0)
        c1 = np.cos(x1)
        s1 = np.sin(x1)
        f0 = s0
        f1 = c0 * s1
        f2 = c0 * c1
        return np.column_stack([f0, f1, f2])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_angle = self.n_obj - 1  # 2 for 3 objectives
        n_dist = self.dim - n_angle  # 10 for 12 vars
        
        # Set distance variables to 0.5 (optimal for DTLZ2)
        dist_vals = np.full(n_dist, 0.5)
        
        # Generate Halton sequence for angle space
        n_points = min(self.budget, 200)
        angles = self._halton_sequence(n_points, n_angle)
        
        # Build full x vectors and evaluate
        all_F = []
        for a in angles:
            x = np.concatenate([a, dist_vals])
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_F.append(f)
        
        all_F = np.array(all_F)
        
        # Check deadline
        if time.time() > deadline:
            return all_F
        
        # Local refinement: pick a subset of points and do small perturbations
        # to capture any residual sub-optimality
        remaining = self.budget - len(all_F)
        if remaining > 0 and time.time() < deadline:
            # Use a fraction of remaining budget for refinement
            n_refine = min(remaining, 50)
            # Pick spread-out points to refine
            idx = np.linspace(0, len(all_F) - 1, min(10, len(all_F))).astype(int)
            for i in range(n_refine):
                if time.time() > deadline:
                    break
                base_idx = idx[i % len(idx)]
                base_angle = angles[base_idx]
                # Small perturbation
                perturbation = 0.02 * np.random.randn(n_angle)
                new_angle = np.clip(base_angle + perturbation, 0, 1)
                x = np.concatenate([new_angle, dist_vals])
                x = np.clip(x, lo, hi)
                f = problem(x)
                all_F = np.vstack([all_F, f])
        
        return all_F
# EVOLVE-BLOCK-END
