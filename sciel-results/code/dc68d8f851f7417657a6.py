import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """
    Hybrid optimizer for BBOB f12 (Bent Cigar).
    
    Combines:
    1. Gaussian Process (Matérn kernel) surrogate with Expected Improvement
       acquisition for global exploration and navigating the rotated valley.
    2. Hooke-Jeeves pattern search for efficient local exploitation along
       the narrow valley structure.
    
    Strategy:
    - Phase 1: Brief random initialization to seed the GP.
    - Phase 2: GP-guided acquisition to find promising regions.
    - Phase 3: Hooke-Jeeves local refinement around the best point.
    - Phase 4: Final GP polish if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # === Phase 1: Random initialization ===
        n_init = min(12, budget // 10)
        rng = np.random.default_rng(42)
        for i in range(n_init):
            if eval_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, d)
            eval_func(x)

        if eval_count >= budget or time.time() > deadline:
            return best_x, best_f

        # === Phase 2: GP-guided acquisition ===
        # Build GP surrogate with Matérn kernel
        # Collect data points
        X_data = []
        y_data = []
        # Re-evaluate best point and collect all evaluated points
        # We need to track all evaluations. Let's modify approach:
        # Keep a list of all evaluated points
        all_points = []
        all_values = []

        # Re-evaluate: we need to track points. Let's restart tracking.
        # Actually, let's just use the best point and do GP from scratch
        # with a small number of points.

        # For GP, we need a set of observed points. Let's collect them
        # by doing a small number of random evaluations and tracking.
        
        # Since we already did n_init evaluations, let's assume we tracked them.
        # Let me refactor to track all points.

        # Restart with proper tracking
        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')
        all_points = []
        all_values = []

        def eval_func2(x):
            nonlocal eval_count, best_x, best_f, all_points, all_values
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            eval_count += 1
            all_points.append(x.copy())
            all_values.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Random initialization
        n_init = min(12, budget // 10)
        rng = np.random.default_rng(42)
        for i in range(n_init):
            if eval_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, d)
            eval_func2(x)

        if eval_count >= budget or time.time() > deadline:
            return best_x, best_f

        # GP parameters
        length_scale = 1.0
        signal_var = 1.0
        noise_var = 1e-6
        nu = 2.5  # Matérn kernel order

        def matern_kernel(X1, X2, ls, sv):
            """Matérn kernel with nu=2.5"""
            D = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2))
            sqrt_5 = np.sqrt(5.0)
            K = (1 + sqrt_5 * D / ls + (5 * D**2) / (3 * ls**2)) * np.exp(-sqrt_5 * D / ls)
            K *= sv**2
            return K

        def gp_predict(X, X_new, ls, sv, nv):
            """GP prediction with Matérn kernel"""
            K = matern_kernel(X, X, ls, sv) + np.eye(len(X)) * nv
            K_star = matern_kernel(X, X_new, ls, sv)
            y = np.array(all_values)
            try:
                K_inv = np.linalg.inv(K)
                mean = K_star.T @ K_inv @ y
                v = K_inv @ K_star
                K_star_star = matern_kernel(X_new, X_new, ls, sv)
                var = K_star_star - np.sum(v * K_star.T, axis=1)
                var = np.maximum(var, 1e-10)
            except:
                mean = np.zeros(len(X_new))
                var = np.ones(len(X_new)) * sv**2
            return mean, var

        def expected_improvement(X_new, mean, var, f_best):
            """Expected Improvement acquisition function"""
            sigma = np.sqrt(var)
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = (f_best - mean) / sigma
                EI = (f_best - mean) * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI = np.where(sigma > 1e-8, EI, 0.0)
            return EI

        # GP optimization loop
        gp_iters = min(200, budget - eval_count - 50)
        for it in range(gp_iters):
            if eval_count >= budget - 50 or time.time() > deadline:
                break

            # Generate candidate points
            n_cand = 500
            X_cand = rng.uniform(-5, 5, (n_cand, d))

            # Predict with GP
            mean, var = gp_predict(np.array(all_points), X_cand, length_scale, signal_var, noise_var)

            # Compute EI
            ei = expected_improvement(X_cand, mean, var, best_f)

            # Select best candidate
            idx = np.argmax(ei)
            x_new = X_cand[idx]
            f_new = eval_func2(x_new)

            if eval_count >= budget - 50 or time.time() > deadline:
                break

            # Adapt length scale based on progress
            if it % 20 == 0 and it > 0:
                # Shrink length scale as we converge
                length_scale *= 0.8
                length_scale = max(length_scale, 0.1)

        if eval_count >= budget or time.time() > deadline:
            return best_x, best_f

        # === Phase 3: Hooke-Jeeves local refinement ===
        # Use remaining budget for local search
        x_hj = best_x.copy()
        f_hj = best_f
        step_sizes = np.ones(d) * 0.5
        min_step = 1e-8
        max_step = 5.0
        step_shrink = 0.5
        step_expand = 1.5

        while eval_count < budget and time.time() < deadline:
            progress = False
            for i in range(d):
                if eval_count >= budget or time.time() > deadline:
                    break

                # Try positive direction
                trial = x_hj.copy()
                trial[i] += step_sizes[i]
                trial = np.clip(trial, -5, 5)
                trial_f = eval_func2(trial)
                if eval_count >= budget:
                    break

                if trial_f < f_hj:
                    x_hj = trial
                    f_hj = trial_f
                    progress = True
                    step_sizes[i] *= step_expand
                    step_sizes[i] = min(step_sizes[i], max_step)
                    # Pattern move: continue in same direction
                    while eval_count < budget and time.time() < deadline:
                        trial = x_hj.copy()
                        trial[i] += step_sizes[i]
                        trial = np.clip(trial, -5, 5)
                        trial_f = eval_func2(trial)
                        if eval_count >= budget:
                            break
                        if trial_f < f_hj:
                            x_hj = trial
                            f_hj = trial_f
                        else:
                            break
                else:
                    # Try negative direction
                    trial = x_hj.copy()
                    trial[i] -= step_sizes[i]
                    trial = np.clip(trial, -5, 5)
                    trial_f = eval_func2(trial)
                    if eval_count >= budget:
                        break

                    if trial_f < f_hj:
                        x_hj = trial
                        f_hj = trial_f
                        progress = True
                        step_sizes[i] *= step_expand
                        step_sizes[i] = min(step_sizes[i], max_step)
                        # Pattern move: continue in same direction
                        while eval_count < budget and time.time() < deadline:
                            trial = x_hj.copy()
                            trial[i] -= step_sizes[i]
                            trial = np.clip(trial, -5, 5)
                            trial_f = eval_func2(trial)
                            if eval_count >= budget:
                                break
                            if trial_f < f_hj:
                                x_hj = trial
                                f_hj = trial_f
                            else:
                                break
                    else:
                        # No improvement, shrink step
                        step_sizes[i] *= step_shrink
                        step_sizes[i] = max(step_sizes[i], min_step)

            if not progress:
                if np.all(step_sizes <= min_step * 1.1):
                    break
                # Random perturbation to escape
                if eval_count < budget and time.time() < deadline:
                    trial = x_hj + rng.uniform(-0.5, 0.5, d)
                    trial = np.clip(trial, -5, 5)
                    trial_f = eval_func2(trial)
                    if eval_count >= budget:
                        break
                    if trial_f < f_hj:
                        x_hj = trial
                        f_hj = trial_f
                        step_sizes = np.ones(d) * 0.3
                    else:
                        step_sizes *= step_shrink
                        step_sizes = np.maximum(step_sizes, min_step)

        if f_hj < best_f:
            best_x = x_hj
            best_f = f_hj

        return best_x, best_f
