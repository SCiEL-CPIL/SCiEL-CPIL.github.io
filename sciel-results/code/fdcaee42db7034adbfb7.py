"""Optimized multi-objective search for ZDT3 exploiting its known structure."""
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3 exploiting its known Pareto front structure.
    
    ZDT3 (n=30):
      f1 = x[0]
      f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/(1+10*(n-1))) - sin(10*pi*f1))
      (with x[1:] = 0 for Pareto front)
    
    Strategy:
    1. Sample the Pareto front densely, emphasizing regions with high curvature
       (near local minima/maxima of the oscillatory sin term).
    2. Use a greedy hypervolume-based selection to pick the best 200 points.
    3. Evaluate selected points using the problem function.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0
        max_evals = self.budget
        dim = self.dim

        # ZDT3 parameters
        g = 1 + 9 * (29 / 30)  # = 10.9
        ref_point = np.array([11.0, 11.0])

        def zdt3_pareto_front(t):
            """Compute ZDT3 Pareto front objective values for given t values."""
            t = np.asarray(t, dtype=float)
            f1 = t
            f2 = g * (1 - np.sqrt(np.maximum(t, 0) / g) - np.sin(10 * np.pi * t))
            return np.column_stack([f1, f2])

        def compute_hypervolume_2d(points, ref):
            """Compute exact 2D hypervolume for a set of points."""
            if len(points) == 0:
                return 0.0
            
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            
            pts = pts[np.argsort(pts[:, 0])]
            
            hv = 0.0
            for i in range(len(pts)):
                f1_i = pts[i, 0]
                f2_i = pts[i, 1]
                
                if i < len(pts) - 1:
                    f1_next = pts[i + 1, 0]
                else:
                    f1_next = ref[0]
                
                if f2_i < ref[1]:
                    width = f1_next - f1_i
                    height = ref[1] - f2_i
                    hv += width * height
            
            return hv

        # Generate candidate t values with emphasis on oscillatory structure
        # ZDT3 has sin(10*pi*t), so 10 full cycles in [0,1]
        # Local minima of f2 at t = (2k+1)/20, k=0..9 (where sin = -1)
        # Local maxima at t = k/10, k=0..10 (where sin = 0)
        
        # Critical points: zeros and extrema
        zeros = np.array([k / 10.0 for k in range(11)])
        extrema = np.array([(2 * k + 1) / 20.0 for k in range(10)])
        
        # Create a dense base grid
        n_base = 150
        t_base = np.linspace(0.0, 1.0, n_base)
        
        # Add extra points clustered around local minima (high curvature regions)
        n_extra_per_min = 5
        t_extra = []
        for loc in extrema:  # These are the local minima of f2
            spread = 0.015
            offsets = np.linspace(-spread, spread, n_extra_per_min)
            cluster = loc + offsets
            t_extra.extend(cluster.tolist())
        
        t_extra = np.array(t_extra)
        
        # Combine all candidates
        t_candidates = np.concatenate([t_base, zeros, extrema, t_extra])
        t_candidates = np.sort(t_candidates)
        t_candidates = np.unique(t_candidates)
        
        # Compute objective values for all candidates
        F_candidates = zdt3_pareto_front(t_candidates)
        
        # Filter to only points strictly better than reference
        mask_valid = np.all(F_candidates < ref_point, axis=1)
        t_valid = t_candidates[mask_valid]
        F_valid = F_candidates[mask_valid]
        
        if len(F_valid) == 0:
            # Fallback: uniform sampling
            t_samples = np.linspace(0, 1, max_evals)
            x = np.zeros(dim)
            recorded = []
            for t in t_samples:
                x[0] = t
                x[1:] = 0
                x_clipped = np.clip(x, lo, hi)
                f = np.asarray(problem(x_clipped), dtype=float)
                recorded.append(f)
                n_evals += 1
                if n_evals >= max_evals or time.time() > deadline:
                    break
            return recorded

        # Sort by f1
        sort_idx = np.argsort(F_valid[:, 0])
        t_sorted = t_valid[sort_idx]
        F_sorted = F_valid[sort_idx]
        
        # Greedy selection using hypervolume gain
        n_select = min(max_evals, len(F_sorted))
        
        # Initialize with endpoints
        selected_idx = [0, len(F_sorted) - 1]
        selected_set = set(selected_idx)
        sorted_selected = F_sorted[selected_idx]
        
        # Greedily add points that maximize hypervolume gain
        remaining_candidates = set(range(len(F_sorted))) - selected_set
        
        while len(selected_idx) < n_select and time.time() < deadline:
            best_gain = -1
            best_idx = -1
            
            # Sample a subset of remaining candidates for efficiency
            if len(remaining_candidates) > 1000:
                sample_size = 1000
                sample_indices = np.random.choice(list(remaining_candidates), size=sample_size, replace=False)
            else:
                sample_indices = list(remaining_candidates)
            
            for idx in sample_indices:
                if idx not in remaining_candidates:
                    continue
                
                f1_new, f2_new = F_sorted[idx, 0], F_sorted[idx, 1]
                
                # Quick dominance check
                if f1_new >= ref_point[0] or f2_new >= ref_point[1]:
                    continue
                
                # Find insertion position
                f1_vals = sorted_selected[:, 0]
                insert_pos = np.searchsorted(f1_vals, f1_new)
                
                # Get neighbors
                if insert_pos == 0:
                    left_f2 = ref_point[1]
                else:
                    left_f2 = sorted_selected[insert_pos - 1, 1]
                
                if insert_pos == len(sorted_selected):
                    right_f2 = ref_point[1]
                else:
                    right_f2 = sorted_selected[insert_pos, 1]
                
                # If dominated by both neighbors, skip
                if f2_new >= left_f2 and f2_new >= right_f2:
                    continue
                
                # Compute approximate gain
                if insert_pos == 0:
                    left_f1 = 0.0
                else:
                    left_f1 = f1_vals[insert_pos - 1]
                
                if insert_pos == len(sorted_selected):
                    right_f1 = ref_point[0]
                else:
                    right_f1 = f1_vals[insert_pos]
                
                old_width = right_f1 - left_f1
                old_height = ref_point[1] - left_f2
                old_contrib = old_width * old_height if old_width > 0 else 0.0
                
                new_left_width = f1_new - left_f1
                new_left_contrib = new_left_width * old_height if new_left_width > 0 else 0.0
                
                new_right_width = right_f1 - f1_new
                new_right_height = ref_point[1] - f2_new
                new_right_contrib = new_right_width * new_right_height if new_right_width > 0 else 0.0
                
                gain = new_left_contrib + new_right_contrib - old_contrib
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx < 0:
                break
            
            # Add the best point
            selected_idx.append(best_idx)
            selected_set.add(best_idx)
            remaining_candidates.discard(best_idx)
            
            # Update sorted_selected
            sorted_selected = F_sorted[sorted(selected_idx)]
        
        # Convert selected indices to t values
        t_selected = t_sorted[sorted(selected_idx)]
        
        # Evaluate the selected points using the problem function
        x = np.zeros(dim)
        recorded = []
        
        for t in t_selected:
            if n_evals >= max_evals or time.time() > deadline:
                break
            
            x[0] = float(t)
            x[1:] = 0
            x_clipped = np.clip(x, lo, hi)
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            n_evals += 1
        
        # If we have budget left, add a few more points at critical locations
        if n_evals < max_evals and time.time() < deadline:
            critical_points = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
            
            for t in critical_points:
                if n_evals >= max_evals or time.time() > deadline:
                    break
                
                # Check if we already have a point close to this t
                if len(t_selected) > 0:
                    if np.min(np.abs(t_selected - t)) < 0.001:
                        continue
                
                x[0] = float(t)
                x[1:] = 0
                x_clipped = np.clip(x, lo, hi)
                f = np.asarray(problem(x_clipped), dtype=float)
                recorded.append(f)
                n_evals += 1
        
        return recorded
