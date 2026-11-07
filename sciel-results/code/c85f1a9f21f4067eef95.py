    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])
        
        archive_X = []
        archive_F = []
        used = 0
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
            
        def try_add(x, f):
            nonlocal used
            # Check if dominated by existing archive
            for af in archive_F:
                if dominates(af, f):
                    return False
            # Remove dominated points from archive
            new_X = []
            new_F = []
            for ax, af in zip(archive_X, archive_F):
                if not dominates(f, af):
                    new_X.append(ax)
                    new_F.append(af)
            archive_X[:] = new_X
            archive_F[:] = new_F
            archive_X.append(x)
            archive_F.append(f)
            return True

        def safe_eval(x):
            nonlocal used
            if used >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            return f

        # Phase 1: Initial Sampling (150 evaluations)
        n_init = min(150, budget)
        for _ in range(n_init):
            x = lo + np.random.random(dim) * (hi - lo)
            f = safe_eval(x)
            if f is None:
                break
            try_add(x, f)
            
        # Add some grid points if budget allows
        if used < budget:
            levels = [0.0, 0.5, 1.0]
            grid_count = 0
            for i in levels:
                for j in levels:
                    for k in levels:
                        for l in levels:
                            if used >= budget:
                                break
                            x = np.array([i, j, k, l])
                            f = safe_eval(x)
                            if f is None:
                                break
                            try_add(x, f)
                            grid_count += 1
                        if used >= budget: break
                    if used >= budget: break
                if used >= budget: break
            if used >= budget: break

        # Phase 2: Local Refinement (Remaining budget)
        # Strategy: Pick random archive points, perturb, evaluate.
        # This helps tighten the front.
        while used < budget and time.time() < deadline:
            if not archive_X:
                # If archive is empty (shouldn't happen), sample random
                x = lo + np.random.random(dim) * (hi - lo)
                f = safe_eval(x)
                if f is not None:
                    try_add(x, f)
                continue
                
            # Pick a random point from archive
            idx = np.random.randint(len(archive_X))
            x_base = archive_X[idx]
            
            # Perturb
            # Scale perturbation by distance to bounds or just fixed small sigma
            sigma = 0.05
            x_new = x_base + np.random.normal(0, sigma, dim)
            
            f = safe_eval(x_new)
            if f is not None:
                try_add(x_new, f)
                
        return archive_X
