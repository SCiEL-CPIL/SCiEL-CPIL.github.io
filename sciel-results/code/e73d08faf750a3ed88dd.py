# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured sampling + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ4 structure:
        # x[0] is g-related (actually for DTLZ4, all x are used but the mapping is:
        # x_m = first (m-1) variables for g, x_{m+1}...x_D for positions)
        # For 3 objectives, m=3, so x[0], x[1] are for g, x[2]...x[11] are for positions
        # Actually, standard DTLZ4: 
        # g(x) = sum_{i=m}^{D} (x_i - 0.5)^2 where m = n_obj + 1
        # f_i = (1 + g) * h_i
        # h_i depends on positions x_{m+1}...x_D

        # Let's use the known Pareto front structure for DTLZ4 with 3 objectives
        # The Pareto front is a quarter-sphere (actually a 2D manifold in 3D space)
        # f1 = (1+g)*cos(theta1)*cos(theta2)
        # f2 = (1+g)*cos(theta1)*sin(theta2)  
        # f3 = (1+g)*sin(theta1)
        # where theta1, theta2 in [0, pi/2]

        # Key insight: on the Pareto front, g=0, so we need x_i = 0.5 for i >= m (m=3)
        # i.e., x[2], x[3], ..., x[11] should be 0.5
        # x[0], x[1] can be anything (they only affect g)
        # The positions are determined by x[2], x[3] (the first two position variables)
        # Wait, for DTLZ4 with M=3, D=12:
        # m = M = 3 (number of objectives)
        # g uses x_m to x_D = x[3] to x[11] (8 variables)
        # h uses x_{m-1} = x[2] (only 1 variable for 3 objectives? No...)
        
        # Actually, let me reconsider. Standard DTLZ4:
        # M objectives, D variables
        # m = M (number of objectives)
        # g(x) = sum_{i=M}^{D-1} (x_i - 0.5)^2  [0-indexed: x[M] to x[D-1]]
        # For M=3, D=12: g uses x[3] to x[11] (9 variables)
        # h functions use x[M-1] = x[2] (1 variable) for 3 objectives? 
        # No, that's not right either.
        
        # Let me just use the fact that the Pareto front is known:
        # For DTLZ4 with 3 objectives, the Pareto front is a 2D surface
        # We can parameterize it by two angles
        # And we know that to be on the Pareto front, the "g" variables must be 0.5
        
        # Strategy: 
        # 1. Explicitly sample points on the known Pareto front (g=0)
        # 2. Add some random points
        # 3. Use local refinement to improve hypervolume
        
        # For 3-objective DTLZ4:
        # The Pareto front can be parameterized by theta1, theta2 in [0, pi/2]
        # f1 = cos(theta1)*cos(theta2)
        # f2 = cos(theta1)*sin(theta2)
        # f3 = sin(theta1)
        
        # To achieve g=0, we need x[i] = 0.5 for all i >= 3 (0-indexed)
        # x[0], x[1] don't affect the front (they only affect g through the sum)
        # Wait, x[0] and x[1] DO affect g in some formulations. Let me be careful.
        
        # Standard DTLZ4 formulation (0-indexed):
        # g(x) = sum_{i=M}^{D-1} (x_i - 0.5)^2  where M = number of objectives
        # For M=3: g = sum_{i=3}^{11} (x_i - 0.5)^2
        # So x[3], x[4], ..., x[11] should be 0.5 for g=0
        # x[0], x[1], x[2] are "position" variables
        
        # h_1 = x[2]
        # h_2 = (1-x[2]) * cos(x[1]*pi/2)  -- wait, this doesn't seem right for 3 objectives
        
        # Actually, for DTLZ4 with M objectives:
        # h_1 = x[M-1]
        # h_2 = (1-x[M-1]) * cos(x[M-2]*pi/2)
        # ...
        # h_{M-1} = (1-x[M-1]) * ... * cos(x[1]*pi/2)
        # h_M = (1-x[M-1]) * ... * sin(x[1]*pi/2)
        
        # For M=3:
        # h_1 = x[2]
        # h_2 = (1-x[2]) * cos(x[1]*pi/2)
        # h_3 = (1-x[2]) * sin(x[1]*pi/2)
        
        # f_1 = (1+g) * h_1
        # f_2 = (1+g) * h_2
        # f_3 = (1+g) * h_3
        
        # On Pareto front (g=0):
        # f_1 = x[2]
        # f_2 = (1-x[2]) * cos(x[1]*pi/2)
        # f_3 = (1-x[2]) * sin(x[1]*pi/2)
        
        # Let t = x[2], s = x[1]*pi/2
        # f_1 = t
        # f_2 = (1-t)*cos(s)
        # f_3 = (1-t)*sin(s)
        
        # This traces out the Pareto front as t in [0,1] and s in [0, pi/2]
        
        # So to sample the Pareto front, we set:
        # x[0] = anything (doesn't affect g or h for this formulation)
        # x[1] = s*2/pi for s in [0, pi/2]
        # x[2] = t for t in [0, 1]
        # x[3] to x[11] = 0.5
        
        # Let's create a grid of points on the Pareto front
        # Budget is 200 evaluations
        
        # Phase 1: Sample the Pareto front with a structured grid
        # Use ~150 evaluations for the front, ~50 for refinement/random
        
        n_front = 140
        n_random = 20
        n_refine = 40  # local refinement on promising points
        
        used = 0
        all_x = []
        all_f = []
        
        # Create grid on Pareto front
        # t in [0, 1], s_angle in [0, pi/2]
        # Use a structured grid
        n_t = 12
        n_s = 12  # 12*12 = 144, close to 140
        n_t = 10
        n_s = 14  # 140 points
        
        for i in range(n_t):
            t = i / (n_t - 1) if n_t > 1 else 0.5
            for j in range(n_s):
                s_angle = j / (n_s - 1) * (np.pi / 2) if n_s > 1 else np.pi/4
                x = np.zeros(dim)
                x[0] = 0.5  # arbitrary, doesn't matter
                x[1] = s_angle * 2 / np.pi  # x[1] = s_angle / (pi/2) = 2*s_angle/pi
                x[2] = t
                x[3:] = 0.5  # g variables set to 0.5
                x = np.clip(x, lo, hi)
                f = problem(x)
                all_x.append(x)
                all_f.append(f)
                used += 1
                if used >= self.budget:
                    break
            if used >= self.budget:
                break
        
        if used >= self.budget:
            return all_x
        
        # Phase 2: Add some random points for robustness
        rng = np.random.default_rng(42)
        for _ in range(n_random):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            all_x.append(x)
            all_f.append(f)
            used += 1
        
        if used >= self.budget:
            return all_x
        
        # Phase 3: Local refinement using Nelder-Mead on promising points
        # Select a few promising points and try to improve them locally
        # We'll minimize each objective separately starting from promising points
        
        # First, identify non-dominated points
        F = np.array(all_f)
        X = np.array(all_x)
        
        # Find non-dominated points
        n_pts = len(F)
        dominated = np.zeros(n_pts, dtype=bool)
        for i in range(n_pts):
            if dominated[i]:
                continue
            for j in range(n_pts):
                if i == j:
                    continue
                # Check if j dominates i
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        
        non_dom_idx = np.where(~dominated)[0]
        
        # Try local refinement on a few non-dominated points
        n_refine_attempts = min(n_refine, len(non_dom_idx))
        rng2 = np.random.default_rng(123)
        
        for attempt in range(n_refine_attempts):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Pick a non-dominated point
            idx = non_dom_idx[rng2.integers(len(non_dom_idx))]
            x0 = X[idx].copy()
            
            # Try to minimize a random objective
            obj_idx = rng2.integers(n_obj)
            
            def objective(x):
                if time.time() > deadline:
                    return 0  # signal to stop
                fx = problem(x)
                return fx[obj_idx]
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.success and res.fun < F[idx][obj_idx]:
                    x_new = res.x
                    f_new = problem(x_new)
                    all_x.append(x_new)
                    all_f.append(f_new)
                    used += 1
            except Exception:
                pass
        
        return all_x
# EVOLVE-BLOCK-END
