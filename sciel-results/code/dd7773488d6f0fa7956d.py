import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        mid = (lo + hi) / 2.0
        
        archive_x = []
        archive_f = []
        evals = 0
        ref_point = np.array([3000.0, 0.06])
        
        def time_ok():
            return time.time() < deadline

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_dominated(f_new, archive_f_list):
            if len(archive_f_list) == 0:
                return False
            for f in archive_f_list:
                if dominates(f, f_new):
                    return True
            return False

        def add_to_archive(x, f):
            if is_dominated(f, archive_f):
                return False
            
            new_archive_x = []
            new_archive_f = []
            for xi, fi in zip(archive_x, archive_f):
                if dominates(f, fi):
                    continue
                new_archive_x.append(xi)
                new_archive_f.append(fi)
                
            new_archive_x.append(x)
            new_archive_f.append(f)
            
            archive_x[:] = new_archive_x
            archive_f[:] = new_archive_f
            return True

        def compute_hv_2d(f_list, ref_point):
            if not f_list:
                return 0.0
            pts = [f for f in f_list if np.all(f < ref_point)]
            if not pts:
                return 0.0
            pts = sorted(pts, key=lambda x: x[0])
            hv = 0.0
            for i in range(len(pts)):
                f1, f2 = pts[i]
                next_f1 = pts[i+1][0] if i < len(pts) - 1 else ref_point[0]
                width = next_f1 - f1
                height = ref_point[1] - f2
                if width > 0 and height > 0:
                    hv += width * height
            return hv

        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(n, dim):
            p = np.zeros(dim)
            primes = [2, 3, 5, 7]
            for d in range(dim):
                p[d] = halton_sequence(n, primes[d])
            return p

        # 1. Initialization: Use Halton sequence for better coverage
        init_size = 40
        for i in range(1, init_size + 1):
            if not time_ok() or evals >= self.budget:
                break
            p = halton_point(i, self.dim)
            x = lo + range_ * p
            f = problem(x)
            evals += 1
            add_to_archive(x, f)
        
        if not time_ok() or evals >= self.budget:
            return archive_x

        # 2. Main search: Hybrid of Random Sampling and Local Optimization
        # We will use a mix of:
        # - Random points (exploration)
        # - Local optimization from archive points (exploitation)
        
        while evals < self.budget and time_ok():
            # Decide strategy: 50% random, 50% local optimization
            if np.random.rand() < 0.5:
                # Exploration: Random point
                x = lo + range_ * np.random.rand(self.dim)
                f_child = problem(x)
                evals += 1
                add_to_archive(x, f_child)
            else:
                # Exploitation: Local optimization from a random archive point
                if not archive_x:
                    x = lo + range_ * np.random.rand(self.dim)
                    f_child = problem(x)
                    evals += 1
                    add_to_archive(x, f_child)
                    continue
                
                idx = np.random.randint(len(archive_x))
                x0 = archive_x[idx].copy()
                
                # Define a scalarized objective for local optimization
                # Use a random weighting vector to explore different parts of the Pareto front
                w = np.random.dirichlet(np.ones(self.n_obj))
                
                def scalarized_objective(x):
                    if evals >= self.budget or not time_ok():
                        return float('inf')
                    fx = problem(x)
                    evals += 1
                    return np.dot(w, fx)
                
                try:
                    # Use Nelder-Mead for derivative-free optimization
                    res = minimize(
                        scalarized_objective, 
                        x0, 
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-4}
                    )
                    
                    if evals < self.budget and time_ok():
                        x_opt = res.x
                        f_opt = problem(x_opt)
                        evals += 1
                        add_to_archive(x_opt, f_opt)
                        
                except Exception:
                    pass

        return archive_x
