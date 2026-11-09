"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Augmented Epsilon-Constraint Method for multi-objective optimization on RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # 5s safety margin before 300s kill
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # Phase 1: Initial sampling to estimate objective ranges
        n_init = min(20, budget // 4)
        rng = np.random.default_rng(42)
        X_init = lo + span * rng.random((n_init, dim))
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Estimate ranges for normalization
        f1_min, f1_max = F_init[:, 0].min(), F_init[:, 0].max()
        f2_min, f2_max = F_init[:, 1].min(), F_init[:, 1].max()
        
        # Ensure ranges are valid
        if f1_max <= f1_min:
            f1_max = f1_min + 1.0
        if f2_max <= f2_min:
            f2_max = f2_min + 1.0

        # Store all evaluated points
        all_X = list(X_init)
        all_F = list(F_init)

        # Phase 2: Epsilon-Constraint Method
        # We optimize f1 subject to f2 <= epsilon
        # Sweep epsilon from f2_max down to f2_min
        
        # Number of epsilon levels we can afford
        # Each minimize call uses ~10-30 evaluations depending on convergence
        # Let's target ~15-20 epsilon levels
        n_eps = min(15, (budget - used) // 10)
        
        # Generate epsilon values
        # Use a log-spaced or linear spread. Linear is safer for unknown landscapes.
        epsilons = np.linspace(f2_max, f2_min, n_eps)
        
        # To avoid duplicates and ensure coverage, we can also add the extreme points
        # found in initialization if they are good.
        
        for eps in epsilons:
            if not time_ok() or used >= budget:
                break
            
            # If epsilon is too tight (below current best f2), skip
            current_min_f2 = min(f[1] for f in all_F)
            if eps < current_min_f2 - 1e-9:
                continue
            
            # Define the scalarized objective with penalty for constraint violation
            # Minimize f1 + mu * max(0, f2 - eps)
            mu = 10.0  # Penalty weight
            
            def obj_func(x):
                fx = problem(x)
                # Clip to ensure finite
                fx = np.clip(fx, -1e6, 1e6)
                penalty = max(0.0, fx[1] - eps)
                return fx[0] + mu * penalty
            
            # Initial guess: pick a point from all_X that is closest to satisfying f2 <= eps
            # or just use the point with best f1 among those with f2 <= eps + margin
            best_x0 = None
            best_score = np.inf
            for x, f in zip(all_X, all_F):
                # Prefer points that satisfy the constraint or are close
                viol = max(0.0, f[1] - eps)
                score = f[0] + mu * viol
                if score < best_score:
                    best_score = score
                    best_x0 = x
            
            if best_x0 is None:
                best_x0 = (lo + hi) / 2.0
            
            # Run local optimization
            try:
                res = minimize(
                    obj_func,
                    best_x0,
                    method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 20, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if used < budget:
                    x_new = np.clip(res.x, lo, hi)
                    f_new = problem(x_new)
                    used += 1
                    all_X.append(x_new)
                    all_F.append(f_new)
            except Exception:
                pass

        # Phase 3: Refine with remaining budget using NSGA-II-like selection or more epsilon sweeps
        remaining = budget - used
        if remaining > 10 and time_ok():
            # Do a second pass with finer epsilon resolution if budget allows
            # Or use the current front to guide further searches
            current_F = np.array(all_F)
            
            # Find non-dominated points
            nd_mask = np.ones(len(current_F), dtype=bool)
            for i in range(len(current_F)):
                if not nd_mask[i]:
                    continue
                for j in range(len(current_F)):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(current_F[j] <= current_F[i]) and np.any(current_F[j] < current_F[i]):
                        nd_mask[i] = False
                        break
            
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) > 0:
                # Perform local refinement on a few non-dominated points
                # Select up to 5 diverse points
                if len(nd_indices) > 5:
                    # Simple diversity: pick points that are far apart in objective space
                    # For simplicity, just pick 5 evenly spaced in index after sorting by f1
                    nd_indices = nd_indices[np.argsort(current_F[nd_indices, 0])]
                    step = max(1, len(nd_indices) // 5)
                    selected = nd_indices[::step][:5]
                else:
                    selected = nd_indices
                
                for idx in selected:
                    if not time_ok() or used >= budget:
                        break
                    
                    x0 = all_X[idx]
                    
                    # Try improving f1 while keeping f2 low
                    # And improving f2 while keeping f1 low
                    for target_obj in [0, 1]:
                        if used >= budget or not time_ok():
                            break
                        
                        def refine_func(x):
                            fx = problem(x)
                            # Penalize deviation from the current point's other objective
                            if target_obj == 0:
                                # Minimize f1, penalize if f2 increases
                                base_f2 = all_F[idx][1]
                                penalty = max(0.0, fx[1] - base_f2)
                                return fx[0] + 5.0 * penalty
                            else:
                                # Minimize f2, penalize if f1 increases
                                base_f1 = all_F[idx][0]
                                penalty = max(0.0, fx[0] - base_f1)
                                return fx[1] + 5.0 * penalty
                        
                        try:
                            res = minimize(
                                refine_func,
                                x0,
                                method='L-BFGS-B',
                                bounds=list(zip(lo, hi)),
                                options={'maxiter': 15, 'ftol': 1e-10}
                            )
                            if used < budget:
                                x_new = np.clip(res.x, lo, hi)
                                f_new = problem(x_new)
                                used += 1
                                all_X.append(x_new)
                                all_F.append(f_new)
                        except Exception:
                            pass

        # Return all evaluated points (the evaluator records all returned objective vectors)
        # The contract says "Every returned objective vector is recorded".
        # The return value of __call__ is ignored, but we must return something.
        # Wait, the contract says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This is slightly ambiguous. Usually in these tasks, the `problem` function is called, and the evaluator tracks those calls.
        # "Every returned objective vector" likely refers to the output of `problem(x)`.
        # The return value of `__call__` itself is ignored.
        # So we just need to ensure we make the calls.
        
        return all_X
# EVOLVE-BLOCK-END
