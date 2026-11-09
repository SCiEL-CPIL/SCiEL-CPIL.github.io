import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Archive: list of (x, f)
        archive = []
        evaluations = 0
        budget = self.budget
        
        def is_dominated(f, archive_f):
            # Check if f is dominated by any point in archive_f
            # f is dominated if exists f' such that f' <= f and f' != f
            for f_prime in archive_f:
                if np.all(f_prime <= f) and np.any(f_prime < f):
                    return True
            return False

        def add_to_archive(x, f):
            nonlocal archive
            # Remove points dominated by new point
            new_archive = []
            for ax, af in archive:
                # If ax is dominated by new x, remove ax
                if np.all(f <= af) and np.any(f < af):
                    continue
                # If new x is dominated by ax, don't add new x
                # We check this before calling add or inside
                new_archive.append((ax, af))
            
            # Check if new point is dominated by existing
            dominated = False
            for ax, af in new_archive:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            
            if not dominated:
                new_archive.append((x, f))
            
            archive = new_archive

        def normalize_simplex(x):
            x = np.clip(x, 0.0, 1.0)
            s = np.sum(x)
            if s == 0:
                u = np.random.rand(self.dim)
                x = u / np.sum(u)
            else:
                x = x / s
            return x

        # Phase 1: Random sampling on simplex
        # Use half the budget for exploration
        n_explore = budget // 2
        
        for _ in range(n_explore):
            if evaluations >= budget: break
            # Sample uniform on simplex
            u = np.random.rand(self.dim)
            x = u / np.sum(u)
            f = np.array(problem(x))
            evaluations += 1
            add_to_archive(x, f)

        # Phase 2: Local search
        # Use remaining budget
        while evaluations < budget:
            if not archive:
                # Fallback to random if archive empty
                u = np.random.rand(self.dim)
                x = u / np.sum(u)
            else:
                # Select a parent randomly from archive
                parent_x, _ = archive[np.random.randint(len(archive))]
                # Mutate
                # Adaptive sigma? Start with 0.1, maybe decrease?
                # With small budget, fixed 0.1 or 0.2 is reasonable.
                sigma = 0.2
                x = parent_x + sigma * np.random.randn(self.dim)
                x = normalize_simplex(x)
            
            f = np.array(problem(x))
            evaluations += 1
            add_to_archive(x, f)

        # Return the last evaluated point or something? 
        # The task says "return value of __call__ is ignored". 
        # But we must return something. 
        # Let's return the best point (min sum of objectives) or just an array.
        if archive:
             best_x = archive[0][0]
        else:
             best_x = np.zeros(self.dim)
        return best_x
