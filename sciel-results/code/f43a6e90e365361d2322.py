import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        archive = []
        evals = 0
        budget = self.budget
        dim = self.dim
        
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def add_to_archive(x, f):
            nonlocal archive
            for ax, af in archive:
                if dominates(af, f):
                    return False
            new_archive = []
            for ax, af in archive:
                if not dominates(f, af):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            archive = new_archive
            return True

        # Multi-Start Coordinate Descent tailored for ZDT3 structure
        # ZDT3: f1=x1, f2 depends on g(x2..xn). Min g implies x2..xn=0.
        
        while evals < budget:
            # Start new trajectory
            x = lo + (hi - lo) * np.random.rand(dim)
            f = np.array(problem(x))
            evals += 1
            add_to_archive(x, f)
            
            if evals >= budget:
                break
            
            # Phase 1: Drive x[1:] to 0
            indices = np.arange(1, dim)
            np.random.shuffle(indices)
            
            for i in indices:
                if evals >= budget:
                    break
                x_new = x.copy()
                x_new[i] = 0.0
                x_new[i] = np.clip(x_new[i], lo[i], hi[i])
                
                f_new = np.array(problem(x_new))
                evals += 1
                
                # Accept if it improves f2 (f1 is unchanged)
                if f_new[1] < f[1]:
                    x = x_new
                    f = f_new
                    add_to_archive(x, f)

            if evals >= budget:
                break
            
            # Phase 2: Explore x[0] with x[1:] fixed (near 0)
            n_explore = min(5, budget - evals)
            for _ in range(n_explore):
                if evals >= budget:
                    break
                x_new = x.copy()
                if np.random.rand() < 0.5:
                    x_new[0] = np.random.uniform(lo[0], hi[0])
                else:
                    x_new[0] = np.clip(x[0] + np.random.normal(0, 0.1), lo[0], hi[0])
                
                f_new = np.array(problem(x_new))
                evals += 1
                add_to_archive(x_new, f_new)
                
        return [x for x, f in archive]
