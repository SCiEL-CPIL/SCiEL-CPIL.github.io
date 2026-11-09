"""Multi-objective optimizer for RE21 combining hypervolume archive with local search polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-based multi-objective optimizer for RE21 with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 275.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref_point = np.array([3000.0, 0.06])

        # Use the global numpy random state as the task specifies np.random.seed(seed)
        # We do not create a new RNG; we use np.random directly.

        # Archive: list of (x, f) pairs that are non-dominated
        archive_X = []
        archive_F = []
        used = 0

        def dominates(a, b):
            """Check if a dominates b (a <= b in all objectives, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(f, archive_F_list):
            """Check if f is dominated by any point in archive_F_list."""
            for af in archive_F_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def compute_hypervolume_2d(points, ref_pt):
            """Compute exact hypervolume for 2D points with given reference point.
            points: list of 2-element arrays (objective values)
            Only points strictly better than ref_point in every objective count."""
            valid = [p for p in points if p[0] < ref_pt[0] and p[1] < ref_pt[1]]
            if not valid:
                return 0.0
            
            # Sort by first objective (ascending)
            valid.sort(key=lambda p: p[0])
            
            # Compute hypervolume by sweeping
            hv = 0.0
            prev_y = ref_pt[1]
            for p in valid:
                if p[1] < prev_y:
                    hv += (ref_pt[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        def add_to_archive(x, f):
            """Add a point to the archive if it's not dominated. Remove dominated points."""
            nonlocal archive_X, archive_F
            if f[0] >= ref_point[0] or f[1] >= ref_point[1]:
                return False  # Not useful for scoring
            
            # Check if dominated by existing archive
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return False
            
            # Remove points dominated by f
            new_X = []
            new_F = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_X.append(ax)
                    new_F.append(af)
            
            new_X.append(x.copy())
            new_F.append(f.copy())
            archive_X = new_X
            archive_F = new_F
            return True

        def evaluate_and_add(x):
            """Evaluate x and add to archive if non-dominated."""
            nonlocal used
            f = problem(x)
            f = np.array(f, dtype=np.float64)
            used += 1
            add_to_archive(x.copy(), f)
            return f

        def prune_archive_by_hv(max_size):
            """Prune archive to non-dominated points and limit size by hypervolume contribution."""
            nonlocal archive_X, archive_F
            if len(archive_X) <= max_size:
                return
            
            # Compute contribution of each point
            # For 2D, sort by f[0] ascending. Non-dominated implies f[1] descending.
            # Contribution of point i is roughly (f[i+1][0] - f[i][0]) * (ref[1] - f[i][1])
            # We do an iterative removal of the point with the smallest contribution.
            
            while len(archive_X) > max_size and time.time() < deadline - 5:
                n = len(archive_X)
                if n <= 1:
                    break
                
                # Sort indices by f[0]
                sorted_idx = np.argsort([f[0] for f in archive_F])
                sorted_F = [archive_F[i] for i in sorted_idx]
                
                # Compute contributions
                contribs = np.zeros(n)
                for i in range(n - 1):
                    w = sorted_F[i+1][0] - sorted_F[i][0]
                    h = ref_point[1] - sorted_F[i][1]
                    contribs[i] = max(0, w) * max(0, h)
                
                w = ref_point[0] - sorted_F[-1][0]
                h = ref_point[1] - sorted_F[-1][1]
                contribs[-1] = max(0, w) * max(0, h)
                
                # Find index in sorted order with min contribution
                min_sorted_idx = np.argmin(contribs)
                min_original_idx = sorted_idx[min_sorted_idx]
                
                # Remove that point
                archive_X.pop(min_original_idx)
                archive_F.pop(min_original_idx)

        # Phase 1: Random initialization
        n_init = min(30, self.budget // 5)
        for i in range(n_init):
            x = lo + (hi - lo) * np.random.random(self.dim)
            evaluate_and_add(x)

        # Phase 2: Evolutionary search
        # We use a simple approach:
        # 1. Select a parent from the archive (weighted by hypervolume contribution or just random)
        # 2. Mutate it
        # 3. Evaluate and add to archive
        # 4. Prune archive if it gets too large
        
        max_archive_size = 50
        
        while used < self.budget:
            if time.time() > deadline - 10:
                break
            
            if not archive_F:
                x = lo + (hi - lo) * np.random.random(self.dim)
                evaluate_and_add(x)
                continue
            
            # Select parent
            # Use a simple heuristic: pick randomly from archive
            # Or pick the point with the best hypervolume contribution?
            # For simplicity, random selection from archive works well with mutation.
            idx = np.random.randint(len(archive_F))
            parent = archive_X[idx]
            
            # Mutation
            # Adaptive step size: start larger, decrease over time
            progress = used / self.budget
            # Step size in terms of range
            # Start with 0.5 * range, end with 0.05 * range
            step_scale = 0.5 * (1.0 - progress) + 0.05
            sigma = step_scale * (hi - lo)
            
            child = parent + np.random.standard_normal(self.dim) * sigma
            child = np.clip(child, lo, hi)
            
            evaluate_and_add(child)
            
            # Prune archive if it gets too large
            if len(archive_X) > max_archive_size:
                prune_archive_by_hv(max_archive_size)
            
            # Occasionally add a random point to maintain diversity
            if used % 10 == 0 and used < self.budget - 5:
                x = lo + (hi - lo) * np.random.random(self.dim)
                evaluate_and_add(x)

        # Phase 3: Local search polish
        # For each point in the archive, perform a local search to improve it
        # This is a refinement step.
        
        # We will use a simple local search: try small perturbations in each dimension
        # and keep the best point found.
        
        # To save time, we only polish the top K points by hypervolume contribution
        # or just all points if the archive is small.
        
        # Let's sort the archive by hypervolume contribution (approximate)
        # and polish the top 10 points.
        
        if archive_X and time.time() < deadline - 5:
            # Compute approximate contributions to rank points
            n = len(archive_X)
            sorted_idx = np.argsort([f[0] for f in archive_F])
            sorted_F = [archive_F[i] for i in sorted_idx]
            
            contribs = np.zeros(n)
            for i in range(n - 1):
                w = sorted_F[i+1][0] - sorted_F[i][0]
                h = ref_point[1] - sorted_F[i][1]
                contribs[i] = max(0, w) * max(0, h)
            
            w = ref_point[0] - sorted_F[-1][0]
            h = ref_point[1] - sorted_F[-1][1]
            contribs[-1] = max(0, w) * max(0, h)
            
            # Get the top 10 points by contribution
            top_k = min(10, n)
            top_indices = np.argsort(contribs)[-top_k:]
            top_indices = sorted_idx[top_indices]
            
            # Polish each top point
            for idx in top_indices:
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                best_x = archive_X[idx]
                best_f = archive_F[idx]
                
                # Try local search: perturb each dimension slightly
                # Use a smaller step size
                local_sigma = 0.05 * (hi - lo)
                
                # Limit the number of local search steps per point
                max_local_steps = min(5, self.budget - used)
                
                for step in range(max_local_steps):
                    if used >= self.budget:
                        break
                    if time.time() > deadline - 2:
                        break
                    
                    # Perturb one random dimension
                    dim_to_perturb = np.random.randint(self.dim)
                    perturbation = np.random.standard_normal() * local_sigma[dim_to_perturb]
                    
                    new_x = best_x.copy()
                    new_x[dim_to_perturb] += perturbation
                    new_x = np.clip(new_x, lo, hi)
                    
                    new_f = problem(new_x)
                    new_f = np.array(new_f, dtype=np.float64)
                    used += 1
                    
                    # Check if new point is better
                    if dominates(new_f, best_f):
                        best_x = new_x
                        best_f = new_f
                    elif not dominates(best_f, new_f):
                        # Non-dominated, add to archive
                        add_to_archive(new_x.copy(), new_f)
                        # Keep best_f as is for further local search? 
                        # Or switch to the new one? 
                        # Let's stick with the one that is best in terms of HV contribution, 
                        # but for simplicity, we continue from the last non-dominated point.
                        # Actually, if it's non-dominated, it's a good candidate for further search.
                        # But to keep it simple, we continue from best_x.
                        pass

                # Add the best found point to archive (it might have been added already)
                add_to_archive(best_x.copy(), best_f.copy())

        return archive_X if archive_X else [lo + (hi - lo) * 0.5]
# EVOLVE-BLOCK-END
