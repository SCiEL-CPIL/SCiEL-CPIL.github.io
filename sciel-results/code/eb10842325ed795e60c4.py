import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Archive of all non-dominated points found
        archive_X = []
        archive_F = []
        used = 0
        
        def is_dominated(f, archive_F_list):
            """Check if f is dominated by any point in archive_F_list."""
            for f_arch in archive_F_list:
                if np.all(f_arch <= f) and np.any(f_arch < f):
                    return True
            return False
        
        def update_archive(x, f):
            """Update archive with new point if non-dominated."""
            nonlocal archive_X, archive_F
            # Check if new point is dominated
            if is_dominated(f, archive_F):
                return
            # Remove points dominated by new point
            new_X = []
            new_F = []
            for xi, fi in zip(archive_X, archive_F):
                if not (np.all(f <= fi) and np.any(f < fi)):
                    new_X.append(xi)
                    new_F.append(fi)
            new_X.append(x)
            new_F.append(f)
            archive_X = new_X
            archive_F = new_F
        
        # Phase 1: Systematic grid-based initialization
        # Use a structured sampling approach to cover the space well
        # We'll use a combination of corner points, midpoint, and stratified sampling
        
        N_init = min(60, budget // 3)  # Use about 1/3 of budget for initialization
        if N_init < 10:
            N_init = min(10, budget)
        
        X_init = np.zeros((N_init, dim))
        
        # Add corner points (2^dim is too many for dim=12, so pick key ones)
        # Instead, use a stratified approach: divide each dimension into 3 levels
        # and sample systematically
        
        # Stratified sampling: for each dimension, use 3 levels (low, mid, high)
        # This gives 3^12 = 531441 points which is too many
        # Instead, use a Latin hypercube with stratification
        
        for d in range(dim):
            # Create 3 strata per dimension
            if N_init >= 3:
                # Divide into ceil(N_init/3) groups
                n_groups = (N_init + 2) // 3
                group_size = N_init // n_groups
                remainder = N_init % n_groups
                
                positions = []
                for g in range(n_groups):
                    start = g * group_size + min(g, remainder)
                    end = start + group_size + (1 if g < remainder else 0)
                    # Random positions within this stratum
                    for _ in range(end - start):
                        frac = (g + np.random.random()) / n_groups
                        positions.append(frac)
                
                # Shuffle to mix
                np.random.shuffle(positions)
                X_init[:, d] = lo[d] + (hi[d] - lo[d]) * np.array(positions[:N_init])
            else:
                X_init[:, d] = np.random.uniform(lo[d], hi[d], N_init)
        
        # Evaluate initial points
        for i in range(N_init):
            if time.time() > deadline:
                break
            f = problem(X_init[i])
            update_archive(X_init[i].copy(), np.array(f))
            used += 1
        
        # Phase 2: Local search around promising points
        # Use the archive to guide search
        remaining_budget = budget - used
        
        if remaining_budget > 0 and archive_X:
            # Sort archive by a proxy for quality (e.g., sum of normalized objectives)
            # Normalize objectives to [0,1] for comparison
            F_arr = np.array(archive_F)
            
            # Find min/max for normalization
            f_min = F_arr.min(axis=0)
            f_max = F_arr.max(axis=0)
            f_range = f_max - f_min
            f_range[f_range == 0] = 1.0
            
            F_norm = (F_arr - f_min) / f_range
            scores = F_norm.sum(axis=1)
            
            # Sort by score (lower is better)
            sorted_indices = np.argsort(scores)
            
            # Perform local search on the best points
            n_local = min(len(sorted_indices), max(5, remaining_budget // 10))
            points_to_search = sorted_indices[:n_local]
            
            steps_per_point = max(2, remaining_budget // n_local)
            
            for idx in points_to_search:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                
                x_base = archive_X[idx].copy()
                f_base = archive_F[idx].copy()
                
                # Adaptive step size
                step_scale = 0.1  # Start with 10% of range
                
                for step_iter in range(steps_per_point):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    # Generate a random perturbation
                    direction = np.random.randn(dim)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    
                    x_trial = x_base + direction * step_scale * (hi - lo)
                    x_trial = np.clip(x_trial, lo, hi)
                    
                    f_trial = np.array(problem(x_trial))
                    used += 1
                    
                    if not is_dominated(f_trial, archive_F):
                        update_archive(x_trial.copy(), f_trial)
                        x_base = x_trial.copy()
                        f_base = f_trial.copy()
                        # Reduce step size on success
                        step_scale *= 0.7
                    else:
                        # Increase step size on failure (exploration)
                        step_scale = min(step_scale * 1.3, 0.3)
        
        # Phase 3: If budget remains, do more random sampling with archive-based rejection
        remaining = budget - used
        if remaining > 0:
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                
                x = np.random.uniform(lo, hi)
                f = np.array(problem(x))
                used += 1
                
                if not is_dominated(f, archive_F):
                    update_archive(x.copy(), f)
        
        # Return the archive points (the evaluator records all returned objective vectors)
        if archive_X:
            return np.array(archive_X)
        else:
            return np.array([])
