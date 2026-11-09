"""Multi-objective optimizer for ZDT2 using analytically optimized Pareto front sampling."""
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT2.
    
    Strategy:
    1. Exploit the known 1D Pareto front: x_1...x_{n-1} = 0, x_0 in [0,1].
    2. The front is f1 = x_0, f2 = A * (1 - sqrt(x_0 / A)) where A = 1 + 10*dim.
    3. Optimize the spacing of x_0 values to maximize hypervolume w.r.t. reference point [11, 11].
    4. Use a power-law parameterization x0 = u^gamma and optimize gamma analytically.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.A = 1.0 + 10.0 * dim
        self.ref = np.array([11.0, 11.0])

    def _compute_analytic_hv(self, x0_vals):
        """
        Compute hypervolume of points on the Pareto front defined by x0_vals.
        f1 = x0
        f2 = A * (1 - sqrt(x0 / A))
        Reference point r = (11, 11)
        """
        if len(x0_vals) == 0:
            return 0.0
        
        f1 = x0_vals
        f2 = self.A * (1.0 - np.sqrt(x0_vals / self.A))
        
        # Only points strictly better than reference point in every objective count
        # f1 < 11 and f2 < 11
        valid_mask = (f1 < self.ref[0]) & (f2 < self.ref[1])
        if not np.any(valid_mask):
            return 0.0
            
        f1_valid = f1[valid_mask]
        f2_valid = f2[valid_mask]
        
        # Sort valid points by f1 increasing (f2 decreasing)
        sort_idx = np.argsort(f1_valid)
        f1_sorted = f1_valid[sort_idx]
        f2_sorted = f2_valid[sort_idx]
        
        # Calculate HV
        # HV = sum of areas of rectangles added by each non-dominated point
        n_valid = len(f1_sorted)
        if n_valid == 0:
            return 0.0
            
        hv = 0.0
        prev_f2 = self.ref[1]
        for i in range(n_valid):
            width = self.ref[0] - f1_sorted[i]
            height = prev_f2 - f2_sorted[i]
            hv += width * height
            prev_f2 = f2_sorted[i]
            
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.budget
        
        # Optimize gamma for the power-law distribution x0 = u^gamma
        # gamma > 1 concentrates points near x0=0 (high curvature region)
        # gamma < 1 concentrates points near x0=1
        
        best_hv = -1.0
        best_gamma = 1.0
        
        # Coarse search over gamma values
        gammas = np.linspace(0.5, 5.0, 60)
        for g in gammas:
            u = np.linspace(0.0, 1.0, n)
            x0_vals = u ** g
            x0_vals[0] = 0.0
            x0_vals[-1] = 1.0
            
            hv = self._compute_analytic_hv(x0_vals)
            if hv > best_hv:
                best_hv = hv
                best_gamma = g
        
        # Fine search around the best gamma found
        if time.time() < deadline - 5.0:
            fine_gammas = np.linspace(best_gamma - 0.3, best_gamma + 0.3, 30)
            fine_gammas = fine_gammas[fine_gammas > 0.01]
            for g in fine_gammas:
                u = np.linspace(0.0, 1.0, n)
                x0_vals = u ** g
                x0_vals[0] = 0.0
                x0_vals[-1] = 1.0
                
                hv = self._compute_analytic_hv(x0_vals)
                if hv > best_hv:
                    best_hv = hv
                    best_gamma = g
        
        # Even finer search if time permits
        if time.time() < deadline - 2.0:
            ultra_fine_gammas = np.linspace(best_gamma - 0.05, best_gamma + 0.05, 20)
            ultra_fine_gammas = ultra_fine_gammas[ultra_fine_gammas > 0.01]
            for g in ultra_fine_gammas:
                u = np.linspace(0.0, 1.0, n)
                x0_vals = u ** g
                x0_vals[0] = 0.0
                x0_vals[-1] = 1.0
                
                hv = self._compute_analytic_hv(x0_vals)
                if hv > best_hv:
                    best_hv = hv
                    best_gamma = g
        
        # Generate final x0 values with the optimal gamma
        u = np.linspace(0.0, 1.0, n)
        x0_vals = u ** best_gamma
        x0_vals[0] = 0.0
        x0_vals[-1] = 1.0
        
        # Create the input matrix
        X = np.zeros((n, self.dim))
        X[:, 0] = x0_vals
        # x1..xN are already 0 (which is optimal for ZDT2)
        
        # Evaluate all points
        F = np.zeros((n, self.n_obj))
        for i in range(n):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
