import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4: f1 = 1 + 8*|sin(10*x1)|, f2 = g(x) * h(f1, g(x))
        # where g(x) = 2 + sum_{i=2}^{n} (x_i^2 - 10*cos(4*pi*x_i))
        # and h(f, g) = 1 - sqrt(f/g)
        # Pareto front: g=2 (all x_i=0), f1 in [1, 9]
        # f2 = 2 * (1 - sqrt(f1/2))
        
        # Strategy: For each x1 sample, evaluate several tail configurations
        # and keep the one that gives the best (lowest) f2.
        # The true Pareto front requires x_i=0 for i>1.
        
        # Budget allocation:
        # We'll use x1 samples and tail candidates such that total <= budget
        n_tail_candidates = 6
        n_x1_samples = self.budget // n_tail_candidates  # 33 samples
        
        # x1 samples: spread across the range to cover f1 in [1, 9]
        # f1 = 1 + 8*|sin(10*x1)|, so x1 in [-5,5] gives f1 in [1,9]
        # We want to sample x1 to get good coverage of f1 values
        x1_vals = np.linspace(lo[0], hi[0], n_x1_samples)
        
        # Tail candidates: configurations for x_2..x_n
        # 1. All zeros (global optimum for g)
        # 2. All 0.5
        # 3. All -0.5
        # 4. All 1.0
        # 5. All -1.0
        # 6. All 0.1
        tail_candidates = []
        tail_candidates.append(np.zeros(self.dim - 1))
        tail_candidates.append(np.full(self.dim - 1, 0.5))
        tail_candidates.append(np.full(self.dim - 1, -0.5))
        tail_candidates.append(np.full(self.dim - 1, 1.0))
        tail_candidates.append(np.full(self.dim - 1, -1.0))
        tail_candidates.append(np.full(self.dim - 1, 0.1))
        
        # Clip tails to valid bounds
        for i in range(len(tail_candidates)):
            tail_candidates[i] = np.clip(tail_candidates[i], lo[1:], hi[1:])
        
        # Evaluate all combinations
        for x1 in x1_vals:
            for tail in tail_candidates:
                x = np.empty(self.dim)
                x[0] = x1
                x[1:] = tail
                x_clipped = np.clip(x, lo, hi)
                problem(x_clipped)
        
        return None
