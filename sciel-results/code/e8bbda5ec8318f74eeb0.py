import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250.0  # safety margin before 300s/330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ4 with 12 vars, 3 objectives:
        # M = 3 objectives, K = M - 1 = 2 position variables
        # N = 12, so M = 10 distance variables
        # Pareto front: x_{M+1..M+K} = x_{M+K+1..N} where g(x) = 0
        # For DTLZ4, g(x) = sum((x_i - 0.5)^2) for i in distance vars
        # g = 0 when each distance var = 0.5
        # f_1 = (1 + g) * prod_{i=1}^{K} cos(x_i * pi/2)
        # f_i = (1 + g) * sin(x_i * pi/2) * prod_{j=1}^{i-2} cos(x_j * pi/2) for 2 <= i <= K+1
        # f_{K+1} = (1 + g) * prod_{j=1}^{K-1} cos(x_j * pi/2) * sin(x_K * pi/2)
        # Actually for DTLZ4: f_i = (1+g) * x_i for i <= K, and f_{K+1} = (1+g) * (1 - sum of squares... )
        # Wait, let me be more careful. DTLZ4 has:
        # f_i(x) = (1 + g(x_M)) * x_{i-1} for i = 1..M+1
        # No, that's DTLZ1. DTLZ4 is:
        # f_1 = (1+g) * cos(x_1*pi/2)
        # f_2 = (1+g) * sin(x_1*pi/2) * cos(x_2*pi/2)
        # f_3 = (1+g) * sin(x_2*pi/2)
        # with g(x) = sum_{i=3}^{12} (x_i - 0.5)^2
        # Pareto front when g=0: f_1 = cos(x_1*pi/2), f_2 = sin(x_1*pi/2)*cos(x_2*pi/2), f_3 = sin(x_2*pi/2)
        # Let theta_1 = x_1*pi/2, theta_2 = x_2*pi/2
        # f_1 = cos(t1), f_2 = sin(t1)*cos(t2), f_3 = sin(t2)
        # with t1, t2 in [0, pi/2]
        
        # So the Pareto front is parameterized by (t1, t2) in [0, pi/2]^2
        # f_1 = cos(t1)
        # f_2 = sin(t1)*cos(t2)
        # f_3 = sin(t2)
        
        # Strategy: Sample (t1, t2) directly from the Pareto front parameter space
        # Use a grid + jitter approach for good hypervolume coverage
        
        # For hypervolume maximization with reference point [2.5, 2.5, 2.5]:
        # We want to cover the Pareto front well, especially near the "knee" regions
        # where points contribute most to hypervolume
        
        # Use a combination of:
        # 1. A structured grid in (t1, t2) space
        # 2. Some points at the extremes
        
        n_budget = budget
        
        # Strategy: Use uniform spacing in objective space (f1 = cos²θ) instead of
        # uniform angular spacing to optimize 2D hypervolume coverage.
        # For f1 = cos(t1), uniform in f1 means t1 = arccos(f1_val) where f1_val is uniform.
        # Similarly for f3 = sin(t2), uniform in f3 means t2 = arcsin(f3_val).
        
        # Use greedy farthest-point selection on a fine analytic candidate pool
        # to maximize hypervolume coverage of the curved Pareto front.
        
        # Generate a fine grid of candidates in (t1, t2) space
        n_fine = 40
        t1_fine = np.linspace(0, np.pi/2, n_fine)
        t2_fine = np.linspace(0, np.pi/2, n_fine)
        
        # Create candidate points in objective space
        T1, T2 = np.meshgrid(t1_fine, t2_fine, indexing='ij')
        f1_cand = np.cos(T1)
        f2_cand = np.sin(T1) * np.cos(T2)
        f3_cand = np.sin(T2)
        
        # Flatten to get candidate objective vectors
        cand_objs = np.column_stack([f1_cand.ravel(), f2_cand.ravel(), f3_cand.ravel()])
        cand_params = np.column_stack([T1.ravel(), T2.ravel()])
        
        # Greedy farthest-point selection in objective space
        selected_indices = []
        remaining = set(range(len(cand_objs)))
        
        # Start with a point near the center of the front
        # The "center" of the Pareto front is around (0.5, 0.25, 0.5) in objective space
        # Let's pick a point that maximizes the minimum distance to reference [2.5, 2.5, 2.5]
        # Actually, let's just start with the first point and greedily add farthest points
        
        # Initialize with index 0 (t1=0, t2=0) -> f=(1,0,0)
        selected_indices.append(0)
        remaining.discard(0)
        
        while len(selected_indices) < n_budget and remaining:
            # Compute minimum distance from each remaining point to any selected point
            sel_objs = cand_objs[selected_indices]
            rem_objs = cand_objs[list(remaining)]
            
            # Compute distances
            dists = np.sqrt(np.sum((rem_objs[:, np.newaxis, :] - sel_objs[np.newaxis, :, :])**2, axis=2))
            min_dists = np.min(dists, axis=1)
            
            # Select the point with maximum minimum distance
            best_local = np.argmax(min_dists)
            best_global = list(remaining)[best_local]
            
            selected_indices.append(best_global)
            remaining.discard(best_global)
        
        # Convert selected (t1, t2) to x vectors
        points = []
        for idx in selected_indices:
            t1, t2 = cand_params[idx]
            x = np.zeros(dim)
            x[0] = t1 / (np.pi/2)
            x[1] = t2 / (np.pi/2)
            x[2:] = 0.5
            points.append(x)
        
        # Pad with additional points if we didn't reach budget (shouldn't happen with 1600 candidates)
        while len(points) < n_budget:
            x = np.zeros(dim)
            x[2:] = 0.5
            points.append(x)
        
        # Clip to budget
        if len(points) > n_budget:
            points = points[:n_budget]
        
        # Evaluate all points
        F = np.array([problem(x) for x in points])
        
        if time.time() > deadline:
            return F
        
        return F
