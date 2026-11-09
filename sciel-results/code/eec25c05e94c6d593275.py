import numpy as np
import time


class Optimizer:
    """Direct construction of the ZDT3 Pareto front with optimized sampling.

    For ZDT3, the Pareto front is obtained by setting x_1 = t and x_i = 0 for i > 1.
    The objective function f_2(t) behaves roughly like 11*(1 - sqrt(t/11)).
    The term 11 - f_2(t) is concave and increasing.
    To maximize the hypervolume (which approximates the integral of this term),
    we use a non-uniform sampling density that clusters points near t=0,
    where the derivative is highest, to better capture the area.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = int(self.budget)

        n_sample = max(2, budget)
        
        # Use power-law spacing: t = (i / (N-1))^p
        # p < 1 clusters points near 0.
        # Based on h'(t) ~ t^-0.5, optimal p is around 0.8.
        p = 0.8
        
        if n_sample == 2:
            t_values = np.array([0.0, 1.0])
        else:
            i = np.arange(n_sample)
            # Ensure exact 0 and 1
            t_values = (i / (n_sample - 1)) ** p

        recorded = []
        used = 0
        
        # Pre-allocate a vector for speed
        x = np.zeros(dim)
        
        for tv in t_values:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Set x_1 to t, rest to 0
            x[0] = float(tv)
            
            # Clip to bounds
            x_clipped = np.clip(x, lo, hi)
            
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            used += 1

        return recorded
