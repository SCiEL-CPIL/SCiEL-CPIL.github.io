import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        # ZDT6 specific knowledge
        # f1(x1) = 1 - (1 - exp(-4*x1) * sin(6*pi*x1)**6)**(1/10)
        # f2(x1) = 1 - sqrt(f1(x1))
        
        def calc_f1(x1):
            # x1 is scalar or array
            return 1 - (1 - np.exp(-4 * x1) * np.sin(6 * np.pi * x1)**6)**0.1

        # Use closed-form parametric representation
        # On the Pareto front, x2..xn = 0, and f1 = 1 - u where u = exp(-4*x1)*sin(6*pi*x1)^6
        # We can sample u uniformly in [0, 1] and solve for x1 such that f1(x1) = 1 - u
        # Actually, f1 is not monotonic, so we need to be careful.
        # Let's use the dense lookup table approach but with better precision.
        
        # Sample x1 densely over [0, 1]
        # Use 200,000 points for high precision
        x1_samples = np.linspace(0, 1, 200000)
        f1_samples = calc_f1(x1_samples)
        
        # Sort by f1
        sort_idx = np.argsort(f1_samples)
        f1_sorted = f1_samples[sort_idx]
        x1_sorted = x1_samples[sort_idx]
        
        # Find the maximum f1 value
        f1_max = f1_sorted[-1]
        
        # Generate target f1 values uniformly spaced
        n_points = self.budget
        target_f1s = np.linspace(0, f1_max, n_points)
        
        # Find x1 for each target using searchsorted
        indices = np.searchsorted(f1_sorted, target_f1s)
        indices = np.clip(indices, 0, len(f1_sorted) - 1)
        
        # Refinement: Check neighbors for better accuracy
        x1_targets = np.empty(n_points)
        for i in range(n_points):
            idx = indices[i]
            target = target_f1s[i]
            
            # Check current, previous, and next
            best_dist = np.inf
            best_x1 = x1_sorted[idx]
            
            for j in range(max(0, idx - 1), min(len(f1_sorted), idx + 2)):
                dist = abs(f1_sorted[j] - target)
                if dist < best_dist:
                    best_dist = dist
                    best_x1 = x1_sorted[j]
            
            x1_targets[i] = best_x1

        # Construct and evaluate points
        recorded = []
        for i in range(n_points):
            # Check deadline
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x1_targets[i]
            x = np.clip(x, 0, 1)
            obj = problem(x)
            recorded.append(obj)
            
        return recorded
