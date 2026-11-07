"""Multi-objective optimizer for WFG4 using constrained Chebyshev scalarization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Constrained Chebyshev scalarization optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 295.0  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        span = hi - lo

        used = 0
        archive_X = []
        archive_F = []

        def time_left():
            return deadline - (time.time() - start_time)

        def evaluate(x):
            nonlocal used
            f = problem(x)
            used += 1
            return np.array(f, dtype=float)

        def is_dominated(f_new, F_list):
            """Check if f_new is dominated by any point in F_list."""
            for f_old in F_list:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def add_to_archive(x, f):
            """Add point to archive if non-dominated; remove dominated points."""
            nonlocal archive_X, archive_F
            if is_dominated(f, archive_F):
                return
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_X)):
                f_old = archive_F[i]
                if not (np.all(f <= f_old) and np.any(f < f_old)):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(f_old)
            new_archive_X.append(np.array(x, dtype=float))
            new_archive_F.append(np.array(f, dtype=float))
            archive_X = new_archive_X
            archive_F = new_archive_F

        def chebyshev_scalarization(x, weights, z_ideal):
            """
            Constrained Chebyshev scalarization objective.
            minimizes max_i { w_i * (f_i(x) - z_ideal_i) }
            subject to f_i(x) <= z_ideal_i + epsilon for all i
            """
            f = evaluate(x)
            # Check constraint: all objectives must be within epsilon of z_ideal
            # If any objective is too large, return a large penalty
            max_dev = np.max(weights * (f - z_ideal))
            
            # Constraint: f_i <= z_ideal_i + 0.5 for all i
            constraint_violation = np.max(f - (z_ideal + 0.5))
            if constraint_violation > 0:
                return max_dev + 100.0 * constraint_violation
            
            return max_dev

        # Generate weight vectors along the simplex
        # For 3 objectives, weights are (w1, w2, w3) with w1+w2+w3=1, wi >= 0
        n_weights = 15
        weights_list = []
        # Use a grid approach for 3 objectives
        for w1 in np.linspace(0.05, 0.9, n_weights):
            for w2 in np.linspace(0.05, 0.9 - w1, max(2, int((0.9 - w1) / 0.05))):
                w3 = 1.0 - w1 - w2
                if w3 >= 0.05:
                    weights_list.append(np.array([w1, w2, w3]))
        
        # Shuffle weights to get diverse exploration early
        rng = np.random.RandomState(42)
        rng.shuffle(weights_list)
        
        # If we have too many weights, subsample
        if len(weights_list) > 40:
            indices = rng.choice(len(weights_list), 40, replace=False)
            weights_list = [weights_list[i] for i in indices]

        # Initial evaluation to get z_ideal estimate
        init_x = np.random.uniform(lo, hi, dim)
        init_f = evaluate(init_x)
        add_to_archive(init_x, init_f)
        
        z_ideal = init_f.copy()
        for f in archive_F:
            z_ideal = np.minimum(z_ideal, f)

        # Solve each scalar subproblem
        for w_idx, weights in enumerate(weights_list):
            if used >= budget or time_left() < 10.0:
                break
            
            # Multiple restarts for each weight vector
            n_restarts = 3
            best_f_for_w = None
            best_x_for_w = None
            
            for restart in range(n_restarts):
                if used >= budget or time_left() < 5.0:
                    break
                
                # Choose initial point: mix of random and archive-based
                if archive_X and restart > 0:
                    # Perturb an archive point
                    base_idx = rng.randint(0, len(archive_X))
                    x0 = archive_X[base_idx].copy()
                    noise = rng.normal(0, 0.1, dim) * span
                    x0 = np.clip(x0 + noise, lo, hi)
                else:
                    x0 = np.random.uniform(lo, hi, dim)
                
                # Define objective function for scipy
                def objective(x):
                    return chebyshev_scalarization(x, weights, z_ideal)
                
                # Bounds for scipy
                scipy_bounds = list(zip(lo, hi))
                
                try:
                    result = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=scipy_bounds,
                        options={'maxiter': 50, 'maxfun': 20}
                    )
                    
                    if result.success and used < budget:
                        f_result = evaluate(result.x)
                        if best_f_for_w is None or np.max(weights * (f_result - z_ideal)) < np.max(weights * (best_f_for_w - z_ideal)):
                            best_f_for_w = f_result
                            best_x_for_w = result.x.copy()
                            add_to_archive(result.x, f_result)
                            # Update z_ideal
                            z_ideal = np.minimum(z_ideal, f_result)
                            
                except Exception:
                    pass

        # Local polish: refine archive points with small perturbations
        if used < budget and time_left() > 5.0 and len(archive_X) > 0:
            n_polish = min(15, len(archive_X))
            if n_polish < len(archive_X):
                indices = np.linspace(0, len(archive_X) - 1, n_polish, dtype=int)
            else:
                indices = np.arange(len(archive_X))
            
            for idx in indices:
                if used >= budget or time_left() < 3.0:
                    break
                base_x = archive_X[idx].copy()
                base_f = archive_F[idx].copy()
                
                for attempt in range(8):
                    if used >= budget or time_left() < 3.0:
                        break
                    
                    # Small random perturbation
                    direction = rng.randn(dim)
                    direction /= np.linalg.norm(direction) + 1e-10
                    step_size = 0.05 * span
                    new_x = base_x + direction * step_size * rng.random(dim)
                    new_x = np.clip(new_x, lo, hi)
                    
                    new_f = evaluate(new_x)
                    
                    # Check if non-dominated
                    if not is_dominated(new_f, archive_F):
                        add_to_archive(new_x, new_f)
                        base_x = new_x.copy()
                        base_f = new_f.copy()
                        z_ideal = np.minimum(z_ideal, new_f)

        # Return the archive
        if archive_X:
            return np.array(archive_X)
        else:
            return np.zeros((1, dim))

# EVOLVE-BLOCK-END
