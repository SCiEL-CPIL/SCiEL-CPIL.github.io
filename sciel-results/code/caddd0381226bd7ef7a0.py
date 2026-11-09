import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Phase 1: Latin Hypercube Sampling for better initial coverage
        # We want to use a good chunk of the budget for initialization to get a diverse set.
        # Let's use 100 points if budget is large enough, otherwise half the budget.
        N_init = 100
        if self.budget < 2 * N_init:
            N_init = self.budget // 2
            if N_init < 10:
                N_init = 10 # Minimum viable population

        # Generate LHS points
        X_lhs = np.zeros((N_init, dim))
        for d in range(dim):
            # Divide range into N_init bins
            bin_width = (hi[d] - lo[d]) / N_init
            # Random permutation of bin indices
            perm = np.random.permutation(N_init)
            # Uniform random point within each bin
            X_lhs[:, d] = lo[d] + bin_width * (perm + np.random.random(N_init))
        
        F_lhs = np.array([problem(x) for x in X_lhs])
        used = N_init

        # Maintain a global archive of all non-dominated points found so far
        # This is crucial for hypervolume maximization as we want to keep all
        # potentially useful points, not just a fixed-size population.
        archive_X = []
        archive_F = []
        
        def update_archive(x, f):
            # Check if f is dominated by any in archive
            dominated = False
            for _, f_arch in zip(archive_X, archive_F):
                if np.all(f_arch <= f) and np.any(f_arch < f):
                    dominated = True
                    break
            if dominated:
                return
            
            # Remove points in archive dominated by f
            new_archive_X = []
            new_archive_F = []
            for x_arch, f_arch in zip(archive_X, archive_F):
                if not (np.all(f <= f_arch) and np.any(f < f_arch)):
                    new_archive_X.append(x_arch)
                    new_archive_F.append(f_arch)
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            # Clear and update global lists
            archive_X.clear()
            archive_F.clear()
            archive_X.extend(new_archive_X)
            archive_F.extend(new_archive_F)

        # Initialize archive with LHS points
        for i in range(N_init):
            update_archive(X_lhs[i], F_lhs[i])

        # Phase 2: Local Search / Memetic Evolution
        # Use remaining budget for local refinement of archive points.
        # This combines the coverage of LHS with the exploitation of local search.
        
        # Helper to select a parent from archive using tournament selection on hypervolume contribution
        # But for simplicity and speed, we can just pick randomly or based on simple criteria.
        # Let's pick randomly from the archive.
        
        # Adaptive sigma: start with a larger step size, reduce over time
        initial_sigma = 0.2
        final_sigma = 0.01
        N_search = self.budget - N_init
        
        if N_search <= 0:
            return archive_X

        for i in range(N_search):
            if time.time() > deadline:
                break
            
            # Linearly decrease sigma
            sigma = initial_sigma - (initial_sigma - final_sigma) * (i / max(1, N_search - 1))
            
            # Pick a parent from the archive
            if len(archive_X) == 0:
                # Fallback to random
                x = lo + (hi - lo) * np.random.random(dim)
            else:
                parent_idx = np.random.randint(len(archive_X))
                x_parent = archive_X[parent_idx]
                
                # Gaussian perturbation
                # Scale noise by the range of each dimension
                noise = np.random.normal(0, 1, dim)
                x = x_parent + noise * (hi - lo) * sigma
                
                # Clip to bounds
                x = np.clip(x, lo, hi)
            
            f = problem(x)
            used += 1
            
            # Update archive
            update_archive(x, f)

        return archive_X
