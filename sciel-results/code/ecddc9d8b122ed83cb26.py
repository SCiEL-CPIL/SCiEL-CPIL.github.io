"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Adaptive Simulated Annealing (ASA) with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget

        # Wrap func to count calls and enforce budget
        call_count = [0]
        best_x_global = [None]
        best_f_global = [np.inf]

        def wrapped(x):
            if call_count[0] >= budget:
                return np.inf
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            if f < best_f_global[0]:
                best_f_global[0] = f
                best_x_global[0] = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Multi-Start Adaptive Simulated Annealing (ASA)
        # We allocate 40% of the budget to ASA to find good basins.
        asa_total_budget = int(budget * 0.4)
        n_starts = 5
        asa_budget_per_start = max(10, int(asa_total_budget / n_starts))

        # Initial points: Center, corners of a smaller box, random
        starts = [np.zeros(dim)]
        for i in range(dim):
            for sign in [1, -1]:
                p = np.zeros(dim)
                p[i] = sign * 4.0
                starts.append(p)
        # Add random starts to fill up to n_starts
        while len(starts) < n_starts:
            starts.append(rng.uniform(-4.5, 4.5, dim))
        # If we have more than n_starts, pick n_starts diverse ones
        # For simplicity, just take the first n_starts
        starts = starts[:n_starts]

        for x0 in starts:
            if call_count[0] >= asa_total_budget:
                break
            if time.time() > deadline - 100: # Leave time for NM
                break

            # ASA Parameters
            T = 1.0
            cooling_rate = 0.95
            alpha = 0.5
            sigma = 1.0
            
            x_curr = x0.copy()
            f_curr = wrapped(x_curr)
            x_best_local = x_curr.copy()
            f_best_local = f_curr
            
            # Run ASA for this start
            for _ in range(asa_budget_per_start):
                if call_count[0] >= asa_total_budget:
                    break
                if time.time() > deadline - 100:
                    break
                
                # ASA Update Rule: x_new = x_curr + alpha * (x_best - x_curr) + sigma * T * noise
                # This pulls towards the best found so far (memetic) and adds noise
                noise = rng.standard_normal(dim)
                x_new = x_curr + alpha * (x_best_local - x_curr) + sigma * T * noise
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = wrapped(x_new)
                
                # Acceptance Criterion
                if f_new < f_curr:
                    x_curr = x_new
                    f_curr = f_new
                    if f_new < f_best_local:
                        x_best_local = x_new
                        f_best_local = f_new
                else:
                    # Boltzmann acceptance
                    delta_f = f_new - f_curr
                    if delta_f < 0:
                        accept = True
                    else:
                        accept_prob = np.exp(-delta_f / (T + 1e-8))
                        accept = (rng.random() < accept_prob)
                    
                    if accept:
                        x_curr = x_new
                        f_curr = f_new
                        if f_new < f_best_local:
                            x_best_local = x_new
                            f_best_local = f_new
                
                # Cool down
                T *= cooling_rate

        # Phase 2: Nelder-Mead Polish on the Global Best
        if best_x_global[0] is not None and time.time() < deadline - 20 and call_count[0] < budget:
            x0 = best_x_global[0].copy()
            remaining_budget = budget - call_count[0]
            
            # NM options
            # maxfev is the max function evaluations. 
            # We want to use the remaining budget.
            nm_maxfev = remaining_budget
            
            try:
                res = minimize(
                    wrapped,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000, 
                        'maxfev': nm_maxfev, 
                        'xatol': 1e-12, 
                        'fatol': 1e-14, 
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Phase 3: Final Coordinate Descent Polish (High Precision)
        if best_x_global[0] is not None and time.time() < deadline - 5 and call_count[0] < budget:
            x = best_x_global[0].copy()
            f = best_f_global[0]
            step = 1.0
            # Use a small fraction of remaining calls for CD
            cd_budget = min(50, budget - call_count[0])
            
            for _ in range(cd_budget):
                if call_count[0] >= budget:
                    break
                if time.time() > deadline - 3:
                    break
                improved = False
                for d in range(dim):
                    if call_count[0] >= budget:
                        break
                    for sign in [1, -1]:
                        trial = x.copy()
                        trial[d] = np.clip(trial[d] + sign * step, -5.0, 5.0)
                        tf = wrapped(trial)
                        if tf < f:
                            x = trial
                            f = tf
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-10:
                        break
                else:
                    step *= 0.8

        return best_x_global[0], best_f_global[0]
# EVOLVE-BLOCK-END
