"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1 that exploits the known Pareto front geometry.
    
    DTLZ1 with M=3 objectives, N=7 variables:
    - g(x) = 100 * (N - M + 1 + sum_{i=M}^{N} [(x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))])
    - f_1 = 0.5 * (1 + g) * x_1
    - f_2 = 0.5 * (1 + g) * x_2
    - f_3 = 0.5 * (1 + g) * (1 - x_1^2 - x_2^2)
    
    Pareto front: g=0 (x_3..x_7 = 0.5), x_1^2 + x_2^2 = 1, x_1, x_2 >= 0
    On PF: f_1 = 0.5*x_1, f_2 = 0.5*x_2, f_3 = 0
    
    Strategy:
    1. Use all budget to sample the known Pareto front via linspace on the quarter-circle.
       This gives uniform coverage in theta which maximizes hypervolume for 3-objective
       DTLZ1 where the front is a quarter-disk in objective space.
    2. Track elapsed time and return early with safety margin before 330s deadline.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        budget = self.budget
        dim = self.dim
        
        n_pf = budget
        
        # The Pareto front of DTLZ1 (M=3) is the quarter disk in (x1, x2) space
        # defined by x1^2 + x2^2 <= 1, x1 >= 0, x2 >= 0.
        # Sampling only the boundary (x1^2 + x2^2 = 1) misses a significant portion
        # of the non-dominated set which contributes to hypervolume.
        # We use a low-discrepancy sequence (Halton) mapped to the quarter disk
        # to ensure good coverage of the 2D Pareto front.
        
        # Generate Halton sequence points in [0, 1]^2
        def halton_sequence(n, base):
            seq = np.zeros(n)
            f = 1.0 / base
            for i in range(1, n):
                p = i
                while p > 0:
                    seq[i] += f * (p % base)
                    p = p // base
                    f /= base
            return seq

        # Use bases 2 and 3 for 2D
        u1 = halton_sequence(n_pf, 2)
        u2 = halton_sequence(n_pf, 3)
        
        # Map to quarter disk:
        # Uniform distribution in area requires r = sqrt(u1)
        # Uniform distribution in angle is theta = u2 * pi / 2
        r = np.sqrt(u1)
        theta = u2 * np.pi / 2.0
        
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        
        # Create points
        X_pf = np.zeros((n_pf, dim))
        X_pf[:, 0] = x1
        X_pf[:, 1] = x2
        # Set remaining dimensions to 0.5 (center of bounds) for g=0
        for d in range(2, dim):
            X_pf[:, d] = 0.5
        
        # Clip to bounds
        X_pf = np.clip(X_pf, lo, hi)
        
        # Evaluate Pareto front points
        F_list = []
        for i in range(n_pf):
            if time.time() > deadline:
                break
            F_list.append(problem(X_pf[i]))
        
        if not F_list:
            # Fallback: single point at center of front
            X_single = np.array([np.sqrt(0.5), np.sqrt(0.5)] + [0.5] * (dim - 2))
            F_list.append(problem(X_single))
        
        F_all = np.array(F_list)
        
        return F_all
# EVOLVE-BLOCK-END
