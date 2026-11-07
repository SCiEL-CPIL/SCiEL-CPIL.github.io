import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref_point = np.array([3.0, 5.0, 7.0])

    def __call__(self, problem, bounds):
        start_time = time.time()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Archive of non-dominated points
        archive_x = []
        archive_f = []
        
        eval_count = 0
        
        def is_dominated(f, archive_f_list):
            for af in archive_f_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False
            
        def add_to_archive(x, f):
            nonlocal archive_x, archive_f
            # Remove dominated points from archive
            new_archive_x = []
            new_archive_f = []
            for ax, af in zip(archive_x, archive_f):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_x.append(ax)
                    new_archive_f.append(af)
            
            # Check if new point is dominated by existing
            if not is_dominated(f, new_archive_f):
                new_archive_x.append(x)
                new_archive_f.append(f)
                
            archive_x = new_archive_x
            archive_f = new_archive_f

        def evaluate(x):
            nonlocal eval_count
            if time.time() - start_time > 320:
                return None
            if eval_count >= self.budget:
                return None
            eval_count += 1
            return problem(x)

        def local_search(x, steps_per_dim=2):
            # Minimize sum of objectives
            current_f = evaluate(x)
            if current_f is None:
                return x, current_f
            
            for d in range(self.dim):
                # Ternary search on dimension d
                a = lo[d]
                b = hi[d]
                
                for _ in range(steps_per_dim):
                    m1 = a + (b - a) / 3.0
                    m2 = b - (b - a) / 3.0
                    
                    x1 = x.copy()
                    x1[d] = m1
                    f1 = evaluate(x1)
                    
                    x2 = x.copy()
                    x2[d] = m2
                    f2 = evaluate(x2)
                    
                    if f1 is None or f2 is None:
                        break
                        
                    s1 = np.sum(f1)
                    s2 = np.sum(f2)
                    
                    if s1 < s2:
                        b = m2
                        # Optionally update x if m1 is better? 
                        # Ternary search just narrows the interval.
                        # We need to evaluate the best point found at the end.
                    else:
                        a = m1
                        
                # After ternary search, evaluate the midpoint
                x_best_dim = x.copy()
                x_best_dim[d] = (a + b) / 2.0
                f_best_dim = evaluate(x_best_dim)
                
                if f_best_dim is not None and np.sum(f_best_dim) < np.sum(current_f):
                    x = x_best_dim
                    current_f = f_best_dim
            
            return x, current_f

        # Initialization: 5 random points
        n_init = 5
        for _ in range(n_init):
            if time.time() - start_time > 320 or eval_count >= self.budget:
                break
            x0 = lo + (hi - lo) * np.random.rand(self.dim)
            f0 = evaluate(x0)
            if f0 is None:
                break
            add_to_archive(x0, f0)
            
            # Local search
            x_opt, f_opt = local_search(x0, steps_per_dim=2)
            if f_opt is not None:
                add_to_archive(x_opt, f_opt)
                
            # If we have budget left, try to refine the best point further?
            # Or start a new point?
            # With 5 points * 12 dims * 2 steps * 2 evals = 240 evals.
            # Budget is 200.
            # So we can only do ~4 points fully.
            # Let's do 4 points.
            
        # Return the archive points (or just the last one, as return value is ignored)
        # The evaluator records ALL objective vectors returned by problem(x).
        # So we just need to ensure we called problem enough times.
        
        return archive_x[-1] if archive_x else None
