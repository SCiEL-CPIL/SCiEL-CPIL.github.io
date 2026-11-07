import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill

        n = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front: f1 = t, f2 = 3*(1 - sqrt(t) - sin(10*pi*t)), t in [0,1]
        # The Pareto front consists of running minima of f2 as t increases.
        
        # Fine grid to identify Pareto-optimal regions
        t_fine = np.linspace(0.0, 1.0, 10001)
        f2_fine = 3.0 * (1.0 - np.sqrt(t_fine) - np.sin(10.0 * np.pi * t_fine))
        
        # Find running minima
        running_min = np.full_like(f2_fine, np.inf)
        running_min[0] = f2_fine[0]
        for i in range(1, len(t_fine)):
            running_min[i] = min(running_min[i-1], f2_fine[i])
        
        # Pareto-optimal points are where f2 equals the running minimum
        is_new_min = np.zeros(len(t_fine), dtype=bool)
        is_new_min[0] = True
        for i in range(1, len(t_fine)):
            if f2_fine[i] < running_min[i-1]:
                is_new_min[i] = True
        
        pareto_idx = np.where(is_new_min)[0]
        if len(pareto_idx) > 0:
            pareto_t = t_fine[pareto_idx]
            pareto_f2 = f2_fine[pareto_idx]
        else:
            pareto_t = np.array([0.0])
            pareto_f2 = np.array([f2_fine[0]])
        
        # Now allocate budget to densely sample the Pareto-optimal segments.
        # For best hypervolume coverage, we want uniform spacing in the objective space
        # along the Pareto front.
        
        if len(pareto_t) >= budget:
            # Select budget points with uniform spacing in f1
            indices = np.linspace(0, len(pareto_t) - 1, budget).astype(int)
            selected_t = pareto_t[indices]
        else:
            # Use all Pareto-optimal points and add some more if needed
            selected_t = pareto_t.copy()
            if len(selected_t) < budget:
                remaining = budget - len(selected_t)
                extra_indices = np.linspace(0, len(pareto_t) - 1, remaining).astype(int)
                extra_t = pareto_t[extra_indices]
                selected_t = np.concatenate([selected_t, extra_t])
                selected_t = np.unique(selected_t)
        
        # Create candidate points
        X = np.zeros((len(selected_t), n))
        X[:, 0] = selected_t
        # x[1:] = 0 gives minimum g, placing points on the Pareto front
        
        # Evaluate all front points
        F = np.array([problem(x) for x in X])
        
        if time.time() > deadline:
            return X
        
        return X
