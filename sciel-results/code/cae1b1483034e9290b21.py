"""Sparse Grid Interpolation + Adaptive Refinement optimizer for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Sparse Grid Interpolation + Adaptive Refinement optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Data structure for adaptive refinement
        # Store evaluated points and their values
        X_eval = []
        Y_eval = []

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = float(func(x))
            X_eval.append(x.copy())
            Y_eval.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Initial Sparse Grid (Sobol) ---
        # Evaluate a small set of space-filling points to initialize the surrogate
        n_init = 32
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            init_points_unit = sampler.random(n_init)
            init_points = init_points_unit * 10.0 - 5.0
            for x in init_points:
                if eval_count >= budget or time.time() > deadline:
                    break
                wrapped_func(x)
        except Exception:
            for _ in range(n_init):
                if eval_count >= budget or time.time() > deadline:
                    break
                x = rng.uniform(-5.0, 5.0, dim)
                wrapped_func(x)

        if eval_count >= budget or time.time() > deadline:
            return best_x, best_f

        # --- Phase 2: Adaptive Refinement Loop ---
        # Strategy:
        # 1. Build a simple surrogate model (e.g., IDW or Kriging-like)
        # 2. Identify regions of high uncertainty or low predicted value
        # 3. Select the next point to evaluate
        # 4. Refine locally if a new best is found
        
        # We will use a simple Inverse Distance Weighting (IDW) surrogate.
        # To find the next point, we can use a grid of candidate points and
        # select the one that minimizes the predicted value + uncertainty penalty.
        
        # Since IDW doesn't give a direct variance, we can approximate uncertainty
        # by the distance to the nearest neighbors. Points far from existing evaluations
        # are "uncertain".
        
        # We'll maintain a list of "promising" candidate points.
        # Instead of a full grid, we'll use a batch of random points and a local grid around best.
        
        while eval_count < budget and time.time() < deadline:
            # 1. Generate a batch of candidate points
            # Mix of:
            # - Global random points (exploration)
            # - Local perturbations of best (exploitation)
            # - Points in "uncertain" regions (surrogate-guided)
            
            n_candidates = 50
            candidates = []
            
            # 50% global random
            n_global = n_candidates // 2
            global_cands = rng.uniform(-5.0, 5.0, (n_global, dim))
            candidates.extend(global_cands)
            
            # 30% local around best
            n_local = n_candidates // 3
            local_cands = best_x + rng.normal(0, 1.0, (n_local, dim))
            local_cands = np.clip(local_cands, -5.0, 5.0)
            candidates.extend(local_cands)
            
            # 20% surrogate-guided
            n_surr = n_candidates - n_global - n_local
            # For surrogate-guided, we need to evaluate uncertainty.
            # Let's pick random points and score them by IDW prediction + distance penalty.
            if n_surr > 0:
                surr_cands = rng.uniform(-5.0, 5.0, (n_surr * 5, dim)) # Over-sample
                # Score each candidate
                scores = []
                for x_c in surr_cands:
                    # IDW Prediction
                    if len(X_eval) == 0:
                        pred = 0
                        dist = 10
                    else:
                        X_arr = np.array(X_eval)
                        Y_arr = np.array(Y_eval)
                        dists = np.linalg.norm(X_arr - x_c, axis=1)
                        # Avoid division by zero
                        dists = np.maximum(dists, 1e-8)
                        # IDW weights
                        p = 2  # power
                        weights = 1.0 / (dists ** p)
                        pred = np.sum(weights * Y_arr) / np.sum(weights)
                        # Uncertainty ~ distance to nearest neighbor
                        dist = np.min(dists)
                    
                    # Objective: Minimize predicted value + uncertainty penalty
                    # If pred is low, it's good. If dist is high, it's uncertain (could be better).
                    # We want to explore uncertainty, so we add a penalty for being far?
                    # Actually, for minimization, high uncertainty means we don't know if it's good.
                    # Let's use Expected Improvement (EI) approximation or just UCB (Upper Confidence Bound).
                    # UCB = pred - beta * dist (for minimization, we want low pred, but high dist means potential for lower).
                    # Wait, if dist is high, pred is unreliable. 
                    # Let's just pick the point with the lowest predicted value, but with a bias towards unexplored areas.
                    # Simple heuristic: score = pred - 0.1 * dist
                    # If dist is large, score decreases (more attractive).
                    score = pred - 0.1 * dist
                    scores.append(score)
                
                scores = np.array(scores)
                # Pick the best few surrogate candidates
                best_indices = np.argsort(scores)[:n_surr]
                for idx in best_indices:
                    candidates.append(surr_cands[idx])
            
            # 2. Select the best candidate
            # We evaluate all candidates? No, that's too expensive.
            # We should select ONE candidate to evaluate.
            # But we don't have a surrogate for the candidates themselves easily without looping.
            # Let's just pick the candidate with the lowest "heuristic score".
            # For global/local, we don't have a surrogate score.
            # Let's just evaluate the first one? No.
            
            # Better approach:
            # Use the surrogate to rank ALL candidates.
            # If len(X_eval) < 10, just use random.
            
            if len(X_eval) < 10:
                # Not enough data for surrogate, just pick random
                x_next = candidates[0]
            else:
                X_arr = np.array(X_eval)
                Y_arr = np.array(Y_eval)
                cands_arr = np.array(candidates)
                
                # Compute IDW prediction and distance for all candidates
                # This is O(N_candidates * N_eval)
                # N_candidates ~ 50, N_eval up to 500. 25k ops. Fast.
                
                # Distance matrix: (N_cands, N_eval)
                dists = np.linalg.norm(cands_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :], axis=2)
                dists = np.maximum(dists, 1e-8)
                
                p = 2
                weights = 1.0 / (dists ** p) # (N_cands, N_eval)
                
                # Prediction: (N_cands,)
                sum_weights = np.sum(weights, axis=1)
                preds = np.sum(weights * Y_arr[np.newaxis, :], axis=1) / sum_weights
                
                # Uncertainty: min distance (N_cands,)
                min_dists = np.min(dists, axis=1)
                
                # Score: pred - 0.1 * min_dist
                scores = preds - 0.1 * min_dists
                
                # Pick the best candidate
                idx_best = np.argmin(scores)
                x_next = cands_arr[idx_best]
            
            # 3. Evaluate the selected point
            f_next = wrapped_func(x_next)
            
            # 4. If new best, do a quick local refinement
            if f_next < best_f + 1e-6:
                # Local refinement around x_next
                # Use Nelder-Mead or simple random search
                # We have budget left?
                if eval_count < budget and time.time() < deadline:
                    # Try a few steps of local search
                    # Simple: evaluate 5 points in a small ball
                    for _ in range(5):
                        if eval_count >= budget or time.time() > deadline:
                            break
                        x_local = x_next + rng.normal(0, 0.5, dim)
                        x_local = np.clip(x_local, -5.0, 5.0)
                        wrapped_func(x_local)

        # Phase 3: Final Polish
        # If we have budget left, do a few more random samples near best
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.1, dim)
            x = np.clip(x, -5.0, 5.0)
            wrapped_func(x)

        return best_x, best_f

# EVOLVE-BLOCK-END
