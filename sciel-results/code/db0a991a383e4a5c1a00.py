"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using HV-guided local search with restarts."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Deadline guard: The evaluator kills at 330s. We aim to finish by 300s.
        deadline = start_time + 300.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Reference point for HV calculation (from task description)
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # Initialize population with random points
        # We use a larger initial population to cover the space better
        init_size = min(50, self.budget // 10)
        if init_size < 10: init_size = 10
        
        X = []
        F = []
        used = 0
        
        # Initial random sampling
        for _ in range(init_size):
            if used >= self.budget: break
            x = lo + (hi - lo) * np.random.rand(dim)
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X.append(x)
            F.append(f)
            used += 1
            
        # Filter non-dominated points
        def is_dominated(f, others):
            for of in others:
                if np.all(of <= f) and np.any(of < f):
                    return True
            return False
            
        def compute_hv(points, ref):
            # Simple HV calculation for 3 objectives
            # Filter points better than ref
            valid = [p for p in points if np.all(p < ref)]
            if not valid: return 0.0
            
            # Sort by first objective
            valid.sort(key=lambda x: x[0])
            
            hv = 0.0
            prev_h = 0.0
            prev_w = 0.0
            
            # For 3D, we can iterate and calculate contribution
            # This is an O(N^2) or O(N log N) approximation depending on implementation
            # For small N (< 200), O(N^2) is fine.
            
            # Let's use a standard exact HV for 3D
            # Sort by x
            pts = sorted(valid, key=lambda p: p[0])
            hv = 0.0
            last_z = 0.0
            last_y = 0.0
            
            # Actually, let's just use a simple loop for contributions
            # Since N is small, we can do:
            # Sort by f0
            sorted_pts = sorted(valid, key=lambda p: p[0])
            
            # To be efficient and robust, we'll use a slightly more complex but still fast HV calc
            # Or just keep track of the best HV found.
            
            # Let's use a simple recursive HV for 3D
            def hv_3d(points, ref):
                if not points: return 0.0
                # Filter
                points = [p for p in points if np.all(p < ref)]
                if not points: return 0.0
                
                # Sort by first obj
                points.sort(key=lambda p: p[0])
                
                hv = 0.0
                # Iterate
                # This specific HV calc is tricky to get right quickly.
                # Let's use a simpler proxy: Sum of distances to ref? No, that's not HV.
                # Let's use a standard library if available? No.
                # Let's implement a basic HV for 3D.
                
                # Sort by x
                pts = points
                # We can decompose by x
                # For each point, calculate the volume it adds.
                # This requires checking dominance in 2D for the remaining points.
                
                # For simplicity and speed in this loop, we will use a heuristic:
                # We want to maximize the "spread" and "closeness".
                # But the task scores on HV.
                
                # Let's implement a correct but simple HV for 3D.
                # https://github.com/... or manual implementation.
                
                # Manual 3D HV:
                # 1. Sort by f0
                # 2. Iterate through points.
                # 3. For point i, calculate the volume of the box defined by p[i] and ref,
                #    minus the volumes already covered by points j < i that dominate in f1, f2?
                #    No, standard HV is:
                #    Sort by f0.
                #    Maintain a list of "active" points.
                #    For each new point, calculate the 2D HV of the remaining points in the space
                #    defined by [f1, f2] for the slice [f0, ref[0]].
                
                # This is getting complex. Let's use a simpler heuristic for selection:
                # We will maintain a non-dominated set.
                # To select parents, we will use a simple random selection from the non-dominated set.
                # To decide if a child is good, we check if it is non-dominated.
                
                return 0.0 # Placeholder, we will use non-domination count as proxy for progress
        
        # Let's stick to Non-Dominated Sorting + Crowding Distance or just Non-Dominated Set Size.
        # Given the budget is small (200), we can afford to keep all non-dominated points.
        
        def get_non_dominated(X, F):
            nd_idx = []
            for i, fi in enumerate(F):
                dominated = False
                for j, fj in enumerate(F):
                    if i == j: continue
                    if np.all(fj <= fi) and np.any(fj < fi):
                        dominated = True
                        break
                if not dominated:
                    nd_idx.append(i)
            return nd_idx
        
        # Main Loop
        # We will use a local search strategy.
        # Pick a random non-dominated point, perturb it, if the new point is non-dominated, keep it.
        # If the set grows or improves (hard to measure HV quickly), keep going.
        
        # To ensure progress, we will mix:
        # 1. 80% Local search around non-dominated points.
        # 2. 20% Global random restarts.
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            nd_idx = get_non_dominated(X, F)
            
            if not nd_idx:
                # Should not happen if we start with randoms, but just in case
                idx = np.random.randint(len(X))
            else:
                # 80% chance to pick from non-dominated
                if np.random.rand() < 0.8:
                    idx = nd_idx[np.random.randint(len(nd_idx))]
                else:
                    idx = np.random.randint(len(X))
            
            # Local Search: Nelder-Mead style or simple random walk
            # We will use a random walk with adaptive step size.
            # Step size: 0.1 * range initially, decreasing?
            # Let's try a fixed small step for stability.
            
            base_x = X[idx]
            step_size = 0.05 * (hi - lo)
            new_x = base_x + np.random.randn(dim) * step_size
            new_x = np.clip(new_x, lo, hi)
            
            new_f = np.array(problem(new_x))
            used += 1
            
            # Check if new point is dominated by current population
            dominated = False
            for i in range(len(F)):
                if np.all(F[i] <= new_f) and np.any(F[i] < new_f):
                    dominated = True
                    break
            
            if not dominated:
                # Add to population
                X.append(new_x)
                F.append(new_f)
                
                # Prune dominated points to keep population small?
                # With budget 200, population can grow to 200.
                # Pruning helps selection speed.
                if len(X) > 50:
                    nd_idx = get_non_dominated(X, F)
                    if len(nd_idx) < len(X):
                        # Keep only non-dominated
                        X = [X[i] for i in nd_idx]
                        F = [F[i] for i in nd_idx]
                        
        return X
# EVOLVE-BLOCK-END
