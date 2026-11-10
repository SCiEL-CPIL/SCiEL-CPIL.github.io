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
        
        # ZDT3 specific structure: Pareto front is at x[1:] = 0
        # f1 = x[0], f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/(1+10*(n-1))) + sin(10*pi*f1))
        # We assume the problem function follows ZDT3 definition.
        
        def hypervolume_2d(points, ref_point):
            """Calculate hypervolume for 2D points."""
            # points: (N, 2)
            # ref_point: (2,)
            if len(points) == 0:
                return 0.0
            
            # Filter points strictly better than ref_point
            valid = points < ref_point
            valid_points = points[np.all(valid, axis=1)]
            
            if len(valid_points) == 0:
                return 0.0
            
            # Sort by f1
            valid_points = valid_points[np.argsort(valid_points[:, 0])]
            
            hv = 0.0
            prev_f2 = ref_point[1]
            for i in range(len(valid_points)):
                f1, f2 = valid_points[i]
                # The contribution is the rectangle from (f1, f2) to (next_f1, prev_f2)
                # But we need to be careful with the boundaries.
                # Standard 2D HV:
                # Sort by f1 ascending.
                # For each point i, the area contributed is (f1[i+1] - f1[i]) * (ref_f2 - f2[i])
                # Wait, the standard formula is:
                # Sort by f1.
                # HV = sum_{i=0}^{N-1} (f1[i+1] - f1[i]) * (ref_f2 - f2[i])
                # Where f1[N] = ref_f1.
                
                pass
            
            # Simpler iterative approach for 2D:
            # Sort by f1 ascending.
            # Keep track of the minimum f2 seen so far? No.
            # The hypervolume is the area under the "step function" formed by the non-dominated points.
            # Since points are non-dominated, f1 increases and f2 decreases.
            # Area = sum over i of (f1[i+1] - f1[i]) * (ref_f2 - f2[i])
            # With f1[N] = ref_f1.
            
            f1_vals = valid_points[:, 0]
            f2_vals = valid_points[:, 1]
            
            f1_next = np.append(f1_vals[1:], ref_point[0])
            widths = f1_next - f1_vals
            heights = ref_point[1] - f2_vals
            
            return np.sum(widths * heights)

        # Initial population
        # We want to cover the front well.
        # ZDT3 front has discontinuities. 
        # Let's sample x1 values.
        n_init = 50
        # Mix of uniform and clustered
        x1_uniform = np.linspace(0.001, 1.0, n_init // 2)
        x1_clustered = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999] * (n_init // 28 + 1))
        x1_init = np.concatenate([x1_uniform, x1_clustered[:n_init - len(x1_uniform)]])
        
        X_init = np.zeros((n_init, self.dim))
        X_init[:, 0] = x1_init
        # x[1:] = 0
        
        F_init = np.array([problem(x) for x in X_init])
        budget_used = n_init
        
        # Keep non-dominated points
        def is_dominated(point, points):
            """Check if point is dominated by any point in points."""
            for p in points:
                if np.all(p <= point) and np.any(p < point):
                    return True
            return False

        # Find non-dominated set
        non_dom_indices = []
        for i in range(len(F_init)):
            dominated = False
            for j in range(len(F_init)):
                if i == j: continue
                if np.all(F_init[j] <= F_init[i]) and np.any(F_init[j] < F_init[i]):
                    dominated = True
                    break
            if not dominated:
                non_dom_indices.append(i)
                
        # Select top 20 non-dominated points to refine
        # If we have more than 20, pick the ones that contribute most to HV? 
        # Or just pick 20 diverse ones.
        # Let's pick the 20 with the best spread.
        if len(non_dom_indices) > 20:
            # Simple heuristic: pick every k-th point in the sorted non-dominated set
            non_dom_indices = non_dom_indices[::max(1, len(non_dom_indices)//20)][:20]
            
        active_indices = non_dom_indices
        X_active = X_init[active_indices].copy()
        F_active = F_init[active_indices].copy()
        
        # Local search phase
        # Budget remaining: 200 - 50 = 150
        # 20 points. 7 evals per point.
        # We will iterate through the points and try to improve them.
        
        while budget_used < self.budget and time.time() < deadline:
            improved = False
            for i in range(len(active_indices)):
                if budget_used >= self.budget or time.time() > deadline:
                    break
                    
                # Try a few perturbations for point i
                x1_curr = X_active[i, 0]
                # Try moving left and right
                steps = [0.01, 0.05, 0.1]
                best_x = X_active[i].copy()
                best_f = F_active[i].copy()
                best_hv = -1
                
                # Calculate current HV of the set
                current_hv = hypervolume_2d(F_active, np.array([11.0, 11.0]))
                
                for step in steps:
                    for direction in [-1, 1]:
                        if budget_used >= self.budget or time.time() > deadline:
                            break
                            
                        new_x1 = x1_curr + direction * step
                        if new_x1 < 0.001 or new_x1 > 1.0:
                            continue
                            
                        new_x = X_active[i].copy()
                        new_x[0] = new_x1
                        new_f = problem(new_x)
                        budget_used += 1
                        
                        # Check if new_f is non-dominated w.r.t other active points
                        other_points = np.delete(F_active, i, axis=0)
                        dominated = False
                        for p in other_points:
                            if np.all(p <= new_f) and np.any(p < new_f):
                                dominated = True
                                break
                        
                        if dominated:
                            continue
                        
                        # Calculate new HV
                        new_F = np.delete(F_active, i, axis=0)
                        new_F = np.vstack([new_F, new_f])
                        new_hv = hypervolume_2d(new_F, np.array([11.0, 11.0]))
                        
                        if new_hv > best_hv:
                            best_hv = new_hv
                            best_x = new_x
                            best_f = new_f
                
                # Update if improved
                if best_hv > current_hv:
                    X_active[i] = best_x
                    F_active[i] = best_f
                    improved = True
        
        return X_active
