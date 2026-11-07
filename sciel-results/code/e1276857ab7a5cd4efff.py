import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 290.0  # 10s margin from 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # LHS initialization for better spread
        def lhs_sample(n, dim, lo, hi, rng):
            """Latin Hypercube Sampling"""
            samples = np.zeros((n, dim))
            for d in range(dim):
                # Divide [0,1] into n intervals, pick one point from each
                intervals = np.arange(n) / n
                points = rng.uniform(intervals, intervals + 1.0/n)
                # Map to [lo[d], hi[d]]
                samples[:, d] = lo[d] + points * (hi[d] - lo[d])
            return samples

        # Archive: list of (x, f) tuples, only non-dominated points
        archive_x = []
        archive_f = []
        
        def is_dominated(f_new, archive_f_list):
            """Check if f_new is dominated by any point in archive_f_list"""
            for f in archive_f_list:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        def remove_dominated(archive_f_list, archive_x_list, f_new, x_new):
            """Remove points dominated by f_new, return updated lists"""
            new_f = []
            new_x = []
            for i in range(len(archive_f_list)):
                # If archive point is dominated by f_new, remove it
                if np.all(f_new <= archive_f_list[i]) and np.any(f_new < archive_f_list[i]):
                    continue
                new_f.append(archive_f_list[i])
                new_x.append(archive_x_list[i])
            return new_x, new_f

        # Initialize with LHS
        N_init = min(40, budget // 4)
        X_init = lhs_sample(N_init, dim, lo, hi, np.random)
        
        for i in range(N_init):
            f = problem(X_init[i])
            # Add to archive if not dominated
            if not is_dominated(f, archive_f):
                # Remove dominated points from archive
                archive_x, archive_f = remove_dominated(archive_f, archive_x, f, X_init[i])
                archive_x.append(X_init[i].copy())
                archive_f.append(f)
        
        used = N_init
        
        # DE parameters
        CR = 0.9  # Crossover rate
        F_DE = 0.8  # Differential weight
        
        # Main DE loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Get current archive size
            pop_size = len(archive_x)
            if pop_size < 3:
                # Need at least 3 points for DE
                # Add random points
                while pop_size < 3 and used < budget:
                    x_rand = np.random.uniform(lo, hi)
                    f_rand = problem(x_rand)
                    used += 1
                    if not is_dominated(f_rand, archive_f):
                        archive_x, archive_f = remove_dominated(archive_f, archive_x, f_rand, x_rand)
                        archive_x.append(x_rand.copy())
                        archive_f.append(f_rand)
                    pop_size = len(archive_x)
                continue
            
            # DE: for each point in archive, try to improve
            # Use a subset to avoid O(n^2) if archive is large
            if pop_size > 50:
                # Sample a subset for DE operations
                indices = np.random.choice(pop_size, size=min(30, pop_size), replace=False)
            else:
                indices = np.arange(pop_size)
            
            for idx in indices:
                if used >= budget or time.time() > deadline:
                    break
                
                x_base = archive_x[idx]
                
                # Select two other distinct points
                other_indices = [i for i in range(pop_size) if i != idx]
                if len(other_indices) < 2:
                    continue
                
                i1, i2 = np.random.choice(other_indices, size=2, replace=False)
                
                # DE mutation: x_trial = x_base + F_DE * (x_i1 - x_i2)
                x_trial = x_base + F_DE * (archive_x[i1] - archive_x[i2])
                
                # Binomial crossover
                j_rand = np.random.randint(dim)
                for d in range(dim):
                    if np.random.rand() < CR or d == j_rand:
                        pass  # keep trial value
                    else:
                        x_trial[d] = x_base[d]
                
                # Clip to bounds
                x_trial = np.clip(x_trial, lo, hi)
                
                # Evaluate
                f_trial = problem(x_trial)
                used += 1
                
                # Update archive
                if not is_dominated(f_trial, archive_f):
                    archive_x, archive_f = remove_dominated(archive_f, archive_x, f_trial, x_trial)
                    archive_x.append(x_trial.copy())
                    archive_f.append(f_trial)
        
        return archive_x
