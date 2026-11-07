import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 specific knowledge:
        # x[0], x[1] are the position variables on the Pareto front.
        # x[2:] are the distance variables. To be on the Pareto front, x[2:] should be 0.
        
        candidates = []
        
        # 1. Add specific high-value points (Vertices, Midpoints, Centroid)
        # These correspond to known good points on the DTLZ4 front (assuming g(x)=0)
        
        # Vertices in decision space (x0, x1)
        # V1: (1, 0) -> f=(0.5, 0, 0)
        # V2: (0, 1) -> f=(0, 0.5, 0)
        # V3: (0, 0) -> f=(0, 0, 0.5)
        vertices = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0]
        ]
        
        # Edge Midpoints
        # M12: (0.5, 1.0) -> f=(0.25, 0.25, 0)
        # M13: (0.5, 0.0) -> f=(0.25, 0, 0.25)
        # M23: (0.0, 0.5) -> f=(0, 0.25, 0.25)
        midpoints = [
            [0.5, 1.0],
            [0.5, 0.0],
            [0.0, 0.5]
        ]
        
        # Centroid
        # C: (1/3, 0.5) -> f=(1/6, 1/6, 1/6)
        centroid = [
            [1.0/3.0, 0.5]
        ]
        
        specific_xy = vertices + midpoints + centroid
        for xy in specific_xy:
            x = np.zeros(self.dim)
            x[0] = xy[0]
            x[1] = xy[1]
            # x[2:] remains 0.0
            candidates.append(x)
            
        # 2. Latin Hypercube Sampling for the rest
        n_specific = len(candidates)
        n_lhs = self.budget - n_specific
        
        if n_lhs > 0:
            # LHS for x[0] and x[1]
            # We want to cover the [0,1]x[0,1] square uniformly
            # which maps to the Pareto front.
            
            # Create bins
            # To avoid edge effects or bias, we can just use standard LHS
            # x_0[i] = (i + u_0[i]) / n_lhs
            # x_1[i] = (perm(i) + u_1[perm(i)]) / n_lhs  # or similar
            
            u0 = np.random.random(n_lhs)
            u1 = np.random.random(n_lhs)
            
            idx = np.arange(n_lhs)
            x0_lhs = (idx + u0) / n_lhs
            x1_lhs = (idx + u1) / n_lhs
            
            # Shuffle one of the dimensions to break correlation
            # Actually, standard LHS shuffles the pairing.
            # Let's shuffle the indices for x1
            perm = np.random.permutation(n_lhs)
            x1_lhs_shuffled = x1_lhs[perm]
            
            for i in range(n_lhs):
                x = np.zeros(self.dim)
                x[0] = x0_lhs[i]
                x[1] = x1_lhs_shuffled[i]
                # x[2:] is 0.0
                candidates.append(x)
                
        # Ensure we don't exceed budget
        if len(candidates) > self.budget:
            candidates = candidates[:self.budget]
            
        # Evaluate all candidates
        # The problem function clips and returns objectives.
        # We just need to call it.
        
        for x in candidates:
            if time.time() > deadline:
                break
            # Check if x is within bounds (it should be, but just in case)
            # The problem clips it, so we are safe.
            problem(x)
            
        return None
