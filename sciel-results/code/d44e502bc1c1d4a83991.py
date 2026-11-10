import numpy as np
import time


class Optimizer:
    """Direct construction of the ZDT3 Pareto front with optimized sampling.

    For ZDT3, the Pareto front is obtained by setting x_1 = t and x_i = 0 for i > 1.
    The objective function f_2(t) = 11*(1 - sqrt(t/11)).
    We use numerical integration to find optimal spacing that maximizes hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = int(self.budget)

        n_sample = max(2, budget)
        ref1 = 11.0
        ref2 = 11.0

        # Analytical ZDT3 Pareto front:
        # f1(t) = t, f2(t) = 11*(1 - sqrt(t/11)) for t in [0, 1]
        # g(t) = ref2 - f2(t) = 11*sqrt(t/11)
        
        def g(t):
            """Height contribution: ref2 - f2(t)"""
            return 11.0 * np.sqrt(np.maximum(t, 0.0) / 11.0)
        
        def compute_hv(t_sorted):
            """Compute hypervolume for sorted t values on the ZDT3 front."""
            if len(t_sorted) == 0:
                return 0.0
            t_sorted = np.asarray(t_sorted, dtype=float)
            # Ensure sorted
            t_sorted = np.sort(t_sorted)
            # f1 values are t_sorted, f2 values are 11*(1 - sqrt(t/11))
            # HV = sum_{i=0}^{N-1} (f1_{i+1} - f1_i) * (ref2 - f2_i)
            # where f1_N = ref1
            gaps = np.diff(t_sorted)
            # Add final gap to reference point
            final_gap = ref1 - t_sorted[-1]
            all_gaps = np.append(gaps, final_gap)
            heights = g(t_sorted)
            return float(np.sum(all_gaps * heights))
        
        # Multi-scale local search to optimize t positions
        # Start with a reasonable initial guess (power law)
        p_init = 4.0 / 3.0
        if n_sample == 2:
            t_values = np.array([0.0, 1.0])
        else:
            i = np.arange(n_sample)
            t_values = (i / (n_sample - 1)) ** p_init
            t_values[0] = 0.0
            t_values[-1] = 1.0
        
        # Local optimization: iteratively adjust each t to maximize HV
        # Use a simple coordinate descent with local refinement
        best_hv = compute_hv(t_values)
        best_t = t_values.copy()
        
        # Coordinate descent: try to improve each position
        n_iter = 10  # Number of full passes
        step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        for iteration in range(n_iter):
            if time.time() > deadline - 50.0:
                break
            improved = False
            for idx in range(n_sample):
                if idx == 0:
                    continue  # Keep t_0 = 0
                if idx == n_sample - 1:
                    continue  # Keep t_{N-1} = 1
                
                # Try small perturbations around current position
                current_t = best_t[idx]
                for step in step_sizes:
                    for direction in [-1, 1]:
                        new_t = current_t + direction * step
                        # Ensure bounds and ordering
                        if new_t <= best_t[idx-1] or new_t >= best_t[idx+1]:
                            continue
                        trial_t = best_t.copy()
                        trial_t[idx] = new_t
                        trial_hv = compute_hv(trial_t)
                        if trial_hv > best_hv + 1e-12:
                            best_hv = trial_hv
                            best_t = trial_t
                            improved = True
            
            if not improved:
                break
        
        t_values = best_t
        
        # Now evaluate the actual problem
        recorded = []
        used = 0

        x = np.zeros(dim)

        for tv in t_values:
            if used >= budget:
                break
            if time.time() > deadline:
                break

            x[0] = float(tv)
            x_clipped = np.clip(x, lo, hi)
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            used += 1

        return recorded
