"""Surrogate-assisted derivative-free optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Surrogate-assisted optimizer using RBF model and expected improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        d = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                return 1e10
            calls += 1
            try:
                val = float(func(x))
                if np.isnan(val) or np.isinf(val):
                    return 1e10
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val
            except Exception:
                return 1e10

        # Phase 1: Initial sampling for diverse coverage
        rng = np.random.default_rng(42)
        
        # Generate initial points: center, axis-aligned, and random
        init_points = []
        init_points.append(np.zeros(d))
        
        # Axis endpoints
        for i in range(d):
            p = np.zeros(d)
            p[i] = 4.0
            init_points.append(p.copy())
            p[i] = -4.0
            init_points.append(p.copy())
        
        # Random points to fill the space
        n_init = 30
        for _ in range(n_init):
            init_points.append(rng.uniform(-5, 5, d))
        
        # Evaluate initial points
        X = []
        Y = []
        for x in init_points:
            if not time_ok() or calls >= budget:
                break
            fx = eval_func(x)
            if fx < 1e9:
                X.append(x)
                Y.append(fx)
        
        if not X:
            return best_x, best_f
            
        X = np.array(X)
        Y = np.array(Y)
        
        # Phase 2: Surrogate-assisted search
        # We use a simple RBF-like model or just a nearest-neighbor based heuristic 
        # combined with local search since full GP might be overkill or slow to implement 
        # correctly without a library. However, the directive suggests a surrogate.
        
        # Let's implement a simple Kriging-like surrogate using distance-weighted interpolation
        # or simply use a population-based method that leverages the data collected so far.
        
        # Actually, implementing a robust GP from scratch is risky. 
        # Let's try a different structural change: CMA-ES is a very strong derivative-free 
        # method for this type of problem (unimodal-ish, rotated). 
        # But the directive specifically asked for "surrogate model".
        
        # Let's implement a simple surrogate: 
        # Model: f(x) ~ mu(x) + sigma(x)
        # mu(x) = weighted average of known Y
        # sigma(x) = weighted variance or distance to nearest points
        
        # Given the budget of 500, we can afford to evaluate ~20-50 points per iteration.
        # Let's do a loop:
        # 1. Build surrogate from X, Y
        # 2. Generate a set of candidates (e.g., random + perturbations of best)
        # 3. Score candidates using Expected Improvement (EI)
        # 4. Evaluate the top candidate
        # 5. Update X, Y, best
        
        # However, EI with a simple surrogate might not be strong enough.
        # Let's combine: Use the surrogate to identify promising regions, 
        # then do a local Nelder-Mead or coordinate search around the best point.
        
        # Actually, let's stick to the directive: "Replace gradient-based local search with a global derivative-free surrogate model".
        
        # We will implement a simple RBF surrogate.
        # f(x) = sum(w_i * phi(||x - x_i||))
        # We solve for w_i using least squares.
        
        def build_surrogate(X, Y):
            # Simple linear RBF: phi(r) = r^2 or exp(-r^2)
            # Let's use linear RBF for simplicity and robustness
            n = len(X)
            d = X.shape[1]
            # Kernel matrix K
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_sq = np.sum((X[i] - X[j])**2)
                    K[i, j] = dist_sq  # Linear RBF
            # Add a small regularization for stability
            K += 1e-6 * np.eye(n)
            
            try:
                # Solve K * w = Y
                w = np.linalg.solve(K, Y)
            except np.linalg.LinAlgError:
                # Fallback to least squares
                w = np.linalg.lstsq(K, Y, rcond=None)[0]
            
            def predict(x):
                dist_sq = np.sum((X - x)**2, axis=1)
                return np.sum(w * dist_sq)
            
            return predict
        
        # Expected Improvement
        def expected_improvement(x, current_best, predict_fn, X, Y):
            # Simple EI: E[max(best - f(x), 0)]
            # We need a variance estimate. 
            # Let's use the distance to the nearest point in X as a proxy for uncertainty.
            dists = np.sum((X - x)**2, axis=1)
            min_dist = np.min(dists)
            # Scale uncertainty
            sigma = 0.1 * min_dist + 0.01
            mu = predict_fn(x)
            # Normalized improvement
            if mu >= current_best:
                return 0.0
            z = (current_best - mu) / sigma
            # Approximate EI for normal distribution
            # EI = (mu - best) * Phi(z) + sigma * phi(z)  -- wait, sign conventions
            # Let's use a simpler heuristic: Score = (current_best - mu) / (sigma + 1e-6)
            # This favors points with high predicted improvement and low uncertainty.
            return (current_best - mu) / (sigma + 1e-6)
        
        # Main loop
        max_iters = 100
        for _ in range(max_iters):
            if not time_ok() or calls >= budget - 10:
                break
                
            # Build surrogate
            if len(X) < 5:
                # Not enough data, just sample randomly
                x_new = rng.uniform(-5, 5, d)
                fx = eval_func(x_new)
                if fx < 1e9:
                    X = np.vstack([X, x_new])
                    Y = np.append(Y, fx)
                continue
                
            predict_fn = build_surrogate(X, Y)
            
            # Generate candidates
            n_candidates = 50
            candidates = []
            
            # 1. Random points
            for _ in range(n_candidates // 2):
                candidates.append(rng.uniform(-5, 5, d))
                
            # 2. Perturbations of best point
            for _ in range(n_candidates // 4):
                step_size = 0.5
                x_cand = best_x + rng.normal(0, step_size, d)
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)
                
            # 3. Perturbations of other good points
            if len(X) > 1:
                # Pick a few random points from X
                indices = rng.choice(len(X), size=min(5, len(X)), replace=False)
                for idx in indices:
                    step_size = 0.3
                    x_cand = X[idx] + rng.normal(0, step_size, d)
                    x_cand = np.clip(x_cand, -5, 5)
                    candidates.append(x_cand)
            
            # Score candidates
            scores = []
            for x_cand in candidates:
                ei = expected_improvement(x_cand, best_f, predict_fn, X, Y)
                scores.append(ei)
            
            scores = np.array(scores)
            # Select the best candidate
            best_idx = np.argmax(scores)
            x_new = candidates[best_idx]
            
            # Evaluate
            fx = eval_func(x_new)
            
            # Update data
            if fx < 1e9:
                X = np.vstack([X, x_new])
                Y = np.append(Y, fx)
                
                # Periodically do a local search around the best point to exploit
                if calls % 20 == 0 and calls < budget - 20:
                    # Simple coordinate descent or Nelder-Mead
                    # Let's do a few steps of coordinate descent
                    for _ in range(5):
                        if not time_ok() or calls >= budget - 5:
                            break
                        for i in range(d):
                            for step in [0.1, -0.1, 0.01, -0.01]:
                                x_try = best_x.copy()
                                x_try[i] += step
                                if np.any(x_try < -5) or np.any(x_try > 5):
                                    continue
                                f_try = eval_func(x_try)
                                if f_try < best_f:
                                    best_x = x_try
                                    break
                                    
        return best_x, best_f
# EVOLVE-BLOCK-END
