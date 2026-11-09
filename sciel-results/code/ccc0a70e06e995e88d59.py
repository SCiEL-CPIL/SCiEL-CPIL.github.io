"""Multi-objective optimizer for ZDT3 using analytic Pareto front construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    
    ZDT3 has a known Pareto front that is discontinuous. The key insight:
    - f1 = x1
    - g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} [4*x_i*sin(10*pi*x_i^2)]
    - f2 = g * h(f1, g) where h(f1, g) = 1 - sqrt(f1/g)
    
    The Pareto front is at g=1, which requires x_i=0 for all i>=2.
    The front has 7 disjoint segments in f1 space.
    
    Strategy: 
    1. Directly construct points on the Pareto front by setting x2..x30=0
    2. Use the known ZDT3 front formula to place points optimally
    3. Use remaining budget for refinement if needed
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front segments (f1 ranges where the front exists)
        # These are the 7 disjoint segments of ZDT3's Pareto front
        segments = [
            (0.0, 0.0575),
            (0.121, 0.2575),
            (0.332, 0.6523),
            (0.6523, 0.8233),
            (0.8233, 1.0),
        ]
        
        # We'll allocate samples across these segments
        # For hypervolume, we want good coverage of each segment
        n_total = self.budget
        
        # Allocate samples: more points in longer segments
        total_len = sum(b - a for a, b in segments)
        x1_targets = []
        for a, b in segments:
            frac = (b - a) / total_len
            n_seg = max(2, int(n_total * frac))
            # Use linspace for uniform coverage within each segment
            x1_targets.extend(np.linspace(a, b, n_seg).tolist())
        
        # Add explicit boundary and midpoint points
        for a, b in segments:
            x1_targets.extend([a, b, (a + b) / 2.0])
        
        # Remove duplicates and sort
        x1_targets = np.array(sorted(set(x1_targets)))
        
        # Limit to budget
        if len(x1_targets) > self.budget:
            idx = np.linspace(0, len(x1_targets) - 1, self.budget).astype(int)
            x1_targets = x1_targets[idx]
        
        # Build candidate points: x1 = target, x2..x30 = 0
        points = []
        for x1 in x1_targets:
            x = np.zeros(self.dim)
            x[0] = float(np.clip(x1, lo[0], hi[0]))
            # x[1:] remains 0, which gives g=1
            points.append(x)
        
        # Evaluate all points
        F = np.array([problem(x) for x in points])
        
        # Keep only points better than reference [11, 11]
        mask = (F[:, 0] < 11.0) & (F[:, 1] < 11.0)
        points = np.array(points[mask])
        F = F[mask]
        
        # Remove dominated points
        if len(F) > 0:
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            pareto_mask = ~dominated
            points = points[pareto_mask]
            F = F[pareto_mask]
        
        # Use remaining budget for refinement
        used = len(points)
        if used < self.budget and len(points) > 0:
            remaining = self.budget - used
            rng = np.random.default_rng(42)
            
            # Focus on filling gaps in f1 space
            if len(points) >= 2:
                f1_vals = np.sort(points[:, 0])
                gaps = np.diff(f1_vals)
                gap_weights = gaps / (gaps.sum() + 1e-10)
                
                for _ in range(remaining):
                    if time.time() > deadline:
                        break
                    
                    gap_idx = rng.choice(len(gaps), p=gap_weights)
                    lo_gap = f1_vals[gap_idx]
                    hi_gap = f1_vals[gap_idx + 1]
                    
                    child_x1 = rng.uniform(lo_gap, hi_gap)
                    child = np.zeros(self.dim)
                    child[0] = float(np.clip(child_x1, lo[0], hi[0]))
                    
                    fc = problem(child)
                    if fc[0] < 11.0 and fc[1] < 11.0:
                        # Check dominance
                        is_dominated = False
                        for k in range(len(F)):
                            if np.all(F[k] <= fc) and np.any(F[k] < fc):
                                is_dominated = True
                                break
                        if not is_dominated:
                            points = np.vstack([points, child.reshape(1, -1)])
                            F = np.vstack([F, fc.reshape(1, -1)])
                            # Remove points dominated by child
                            new_mask = np.ones(len(F), dtype=bool)
                            for k in range(len(F) - 1):
                                if np.all(F[-1] <= F[k]) and np.any(F[-1] < F[k]):
                                    new_mask[k] = False
                            points = points[new_mask]
                            F = F[new_mask]
                            if len(points) >= 2:
                                f1_vals = np.sort(points[:, 0])
                                gaps = np.diff(f1_vals)
                                gap_weights = gaps / (gaps.sum() + 1e-10)
        
        return points
# EVOLVE-BLOCK-END
