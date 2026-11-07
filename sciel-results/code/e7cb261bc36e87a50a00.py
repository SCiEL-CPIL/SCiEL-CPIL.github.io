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
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        # State
        evals_used = 0
        archive_X = []
        archive_F = []
        
        def evaluate(x):
            nonlocal evals_used
            if evals_used >= budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            evals_used += 1
            return f
            
        def dominates(f1, f2):
            # f1 dominates f2 if f1 <= f2 and f1 != f2
            return np.all(f1 <= f2) and np.any(f1 < f2)
            
        def update_archive(x, f):
            if f is None:
                return
            # Check if f is dominated by any in archive
            # If yes, discard
            for ax, af in zip(archive_X, archive_F):
                if dominates(af, f):
                    return
                    
            # Remove points from archive dominated by f
            to_remove = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if dominates(f, af):
                    to_remove.append(i)
                    
            for idx in sorted(to_remove, reverse=True):
                archive_X.pop(idx)
                archive_F.pop(idx)
                
            archive_X.append(x)
            archive_F.append(f)

        # 1. Initialization
        n_init = 30
        X_init = lo + span * np.random.random((n_init, dim))
        for x in X_init:
            f = evaluate(x)
            if f is not None:
                update_archive(x, f)
        
        # 2. Differential Evolution for global search
        # DE uses population differences for mutation, which is more effective than simple random mutation
        # for small-dimensional continuous problems. We use a population of non-dominated points.
        pop_size = 20
        n_global = budget - 60 - n_init  # Reserve 60 for local polish
        
        # Maintain a population of size pop_size, initialized from archive + random
        population_X = []
        population_F = []
        
        # Initialize population from archive (non-dominated) + random fill
        for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
            if len(population_X) >= pop_size:
                break
            population_X.append(np.array(ax))
            population_F.append(np.array(af))
        
        while len(population_X) < pop_size and evals_used < budget - 60:
            x = np.random.uniform(lo, hi)
            f = evaluate(x)
            if f is not None:
                population_X.append(x)
                population_F.append(f)
        
        # DE loop
        F_de = 0.8  # Differential weight
        CR = 0.9    # Crossover rate
        
        while evals_used < budget - 60 and time.time() < deadline:
            if len(population_X) < 3:
                # Not enough population, add random points
                x = np.random.uniform(lo, hi)
                f = evaluate(x)
                if f is not None:
                    population_X.append(x)
                    population_F.append(f)
                continue
            
            # Generate trial vector using DE/rand/1
            idx = np.random.randint(len(population_X))
            i1 = np.random.randint(len(population_X))
            i2 = np.random.randint(len(population_X))
            while i1 == i2 or i1 == idx or i2 == idx:
                i1 = np.random.randint(len(population_X))
                i2 = np.random.randint(len(population_X))
            
            # Differential mutation
            v = population_X[i1] + F_de * (population_X[i2] - population_X[idx])
            
            # Binomial crossover
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    x_trial = v[j]
                else:
                    x_trial = population_X[idx][j]
                if j < dim - 1:
                    x_trial_j = x_trial
                else:
                    x_trial_j = x_trial
            
            # Build trial vector
            x_trial = np.zeros(dim)
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    x_trial[j] = v[j]
                else:
                    x_trial[j] = population_X[idx][j]
            
            x_trial = np.clip(x_trial, lo, hi)
            f_trial = evaluate(x_trial)
            
            if f_trial is None:
                continue
            
            # Greedy selection: keep if it dominates or has better HV contribution
            # Simple: keep if it dominates the target, or if target doesn't dominate it and it's better in at least one objective
            dominates_target = np.all(f_trial <= population_F[idx]) and np.any(f_trial < population_F[idx])
            target_dominates_trial = np.all(population_F[idx] <= f_trial) and np.any(population_F[idx] < f_trial)
            
            if dominates_target or not target_dominates_trial:
                population_X[idx] = x_trial
                population_F[idx] = f_trial
                update_archive(x_trial, f_trial)
                
        # 3. Local Polish with smooth Tchebycheff decomposition
        # Use log-sum-exp for smooth max to help Nelder-Mead converge
        if len(archive_X) > 0:
            F_arr = np.array(archive_F)
            # Sort by first objective to ensure we pick across the front
            order = np.argsort(F_arr[:, 0])
            
            # Select up to 8 points, evenly spaced in index
            n_points_to_refine = min(8, len(archive_X))
            # Pick indices
            indices = np.linspace(0, len(order)-1, n_points_to_refine).astype(int)
            
            # Define diverse weight vectors for Tchebycheff decomposition
            weight_vectors = [
                np.array([0.1, 0.9]),
                np.array([0.3, 0.7]),
                np.array([0.5, 0.5]),
                np.array([0.7, 0.3]),
                np.array([0.9, 0.1]),
            ]
            
            # Smoothness parameter for log-sum-exp approximation of max
            beta = 50.0  # Higher = closer to exact max, lower = smoother
            
            for i, idx in enumerate(indices):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                x0 = np.array(archive_X[order[idx]])
                f0 = np.array(archive_F[order[idx]])
                
                w = weight_vectors[i % len(weight_vectors)]
                
                # Nadir point estimate
                z_max = np.max(F_arr, axis=0) if len(F_arr) > 0 else np.ones(n_obj)
                z_max = z_max + 1e-6 * np.ones(n_obj)
                
                def obj_smooth(x, w=w, z_max=z_max, beta=beta):
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped), dtype=np.float64)
                    # Smooth Tchebycheff: use log-sum-exp to approximate max(w * (f - z_max))
                    vals = w * (f - z_max)
                    # Log-sum-exp trick for numerical stability
                    m = np.max(vals)
                    if m > 1e10:
                        return m
                    log_sum = m + np.log(np.sum(np.exp(vals - m)))
                    return log_sum / beta
                    # As beta -> inf, this approaches max(vals)
                    # For finite beta, it's a smooth approximation
                
                try:
                    # Use more iterations for better convergence with smooth objective
                    res = minimize(obj_smooth, x0, method='Nelder-Mead', 
                                   options={'maxiter': 30, 'maxfev': 7, 'xatol': 1e-6, 'fatol': 1e-6})
                    if evals_used < budget:
                        x_best = np.clip(res.x, lo, hi)
                        f_best = evaluate(x_best)
                        if f_best is not None:
                            update_archive(x_best, f_best)
                except Exception:
                    pass
                    
        return archive_X
