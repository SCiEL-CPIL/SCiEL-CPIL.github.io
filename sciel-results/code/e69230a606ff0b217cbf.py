import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Strategy: ZDT3 has Pareto front where g(x)=0
        # g depends on x_2..x_30. We need to find points with small g.
        # Use a combination of:
        # 1. Boundary-biased sampling (ZDT3 front is often near boundaries)
        # 2. Local search to minimize f2 (proxy for g)
        # 3. Diverse x_1 to cover the front
        
        evaluated = []
        eval_count = 0
        
        def eval_point(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            f = problem(x)
            eval_count += 1
            evaluated.append((x.copy(), np.array(f, dtype=float)))
            return f
        
        # Phase 1: Diverse initialization with boundary bias
        # ZDT3's front regions are at specific x_2..x_30 values
        # Try combinations of 0 and 1 for x_2..x_30, and vary x_1
        
        n_phase1 = min(80, budget // 2)
        
        # Generate diverse points: x_1 uniform, x_2..x_30 biased to {0, 1}
        for i in range(n_phase1):
            if time.time() > deadline:
                break
            if eval_count >= budget:
                break
            
            x = np.zeros(dim)
            # x_1: spread across [0,1] to cover front
            x[0] = (i + 0.5) / n_phase1 * (hi[0] - lo[0]) + lo[0]
            
            # x_2..x_30: mix of boundary values with some randomness
            for j in range(1, dim):
                r = np.random.random()
                if r < 0.3:
                    x[j] = lo[j]
                elif r < 0.6:
                    x[j] = hi[j]
                elif r < 0.8:
                    x[j] = lo[j] + np.random.random() * 0.1 * range_[j]
                elif r < 0.9:
                    x[j] = hi[j] - np.random.random() * 0.1 * range_[j]
                else:
                    x[j] = lo[j] + np.random.random() * range_[j]
            
            x = np.clip(x, lo, hi)
            eval_point(x)
        
        # Phase 2: Local refinement of best points
        # Sort by f2 (second objective) and refine top candidates
        if evaluated and eval_count < budget:
            # Sort by second objective (f2) - smaller is better for being on front
            evaluated.sort(key=lambda xf: xf[1][1])
            
            n_refine = min(10, len(evaluated))
            step = 0.05
            
            for i in range(n_refine):
                if eval_count >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x_best = evaluated[i][0]
                f_best = evaluated[i][1]
                
                # Try small perturbations
                for trial in range(3):
                    if eval_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Perturb x_2..x_30 slightly (keep x_1 for front coverage)
                    x_new = x_best.copy()
                    for j in range(1, dim):
                        if np.random.random() < 0.3:
                            x_new[j] += np.random.choice([-1, 1]) * step * range_[j] * np.random.random()
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = eval_point(x_new)
                    if f_new is not None and f_new[1] < f_best[1]:
                        x_best = x_new
                        f_best = f_new
        
        # Phase 3: If budget remains, try more targeted search
        # Focus on points with very small f2
        if evaluated and eval_count < budget:
            evaluated.sort(key=lambda xf: xf[1][1])
            # Take the best few and try even finer local search
            n_fine = min(5, len(evaluated))
            step_fine = 0.01
            
            for i in range(n_fine):
                if eval_count >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x_best = evaluated[i][0]
                f_best = evaluated[i][1]
                
                for trial in range(4):
                    if eval_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    x_new = x_best.copy()
                    for j in range(1, dim):
                        if np.random.random() < 0.2:
                            x_new[j] += np.random.choice([-1, 1]) * step_fine * range_[j] * np.random.random()
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = eval_point(x_new)
                    if f_new is not None and f_new[1] < f_best[1]:
                        x_best = x_new
                        f_best = f_new
        
        # Return the best point (will be ignored, but keep interface)
        if evaluated:
            evaluated.sort(key=lambda xf: xf[1][0] + xf[1][1])
            return evaluated[0][0]
        return np.zeros(dim)
