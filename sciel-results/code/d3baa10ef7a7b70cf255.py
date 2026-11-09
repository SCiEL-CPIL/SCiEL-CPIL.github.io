"""Adaptive Model-Driven Sampling with Sobol Sequence for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer using a pre-computed Sobol sequence and adaptive k-NN guided selection.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        X_eval = []
        Y_eval = []

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            X_eval.append(x.copy())
            Y_eval.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return calls_made < budget and (time.time() - start_time) < (deadline - start_time)

        # 1. Generate Sobol points
        # We generate more points than the budget to have a large pool to choose from
        n_sobol = 800 
        sobol = qmc.Sobol(d=dim, scramble=True)
        # Generate points in [0, 1]^dim
        sobol_pts_unit = sobol.random(n_sobol)
        # Scale to [-5, 5]^dim
        sobol_pts = sobol_pts_unit * 10.0 - 5.0
        
        # Add center point explicitly
        sobol_pts = np.vstack([np.zeros(dim), sobol_pts])
        
        # Track which points have been evaluated
        evaluated_mask = np.zeros(len(sobol_pts), dtype=bool)
        
        # Map from index to point for easy lookup
        # We will pick indices from sobol_pts

        # 2. Initial Exploration
        # Evaluate the center and a few well-spread points to initialize the model
        init_indices = [0] # Center
        # Add a few random indices from the Sobol set to start
        init_indices += list(rng.choice(len(sobol_pts), size=15, replace=False))
        
        for idx in init_indices:
            if not time_ok():
                break
            if not evaluated_mask[idx]:
                evaluate(sobol_pts[idx])
                evaluated_mask[idx] = True

        # 3. Adaptive Selection Loop
        K = 8 # k-NN neighbors
        beta = 2.0 # Exploration weight
        
        # Helper to compute k-NN prediction and uncertainty
        def knn_predict_and_uncertainty(x_query, X_train, Y_train, k=K):
            if len(X_train) == 0:
                return 0.0, 10.0
            X_arr = np.array(X_train)
            Y_arr = np.array(Y_train)
            n = len(X_arr)
            if n <= k:
                d2 = np.sum((X_arr - x_query) ** 2, axis=1)
                # Uncertainty is distance to farthest neighbor
                unc = np.sqrt(np.max(d2))
                # Prediction is weighted mean
                w = 1.0 / (d2 + 1e-6)
                pred = float(np.dot(w, Y_arr) / np.sum(w))
                return pred, float(unc)
            
            d2 = np.sum((X_arr - x_query) ** 2, axis=1)
            idx = np.argpartition(d2, k)[:k]
            d2_k = d2[idx]
            Y_k = Y_arr[idx]
            # Uncertainty: distance to k-th nearest neighbor
            unc = float(np.sqrt(np.max(d2_k)))
            # Prediction: weighted mean
            w = 1.0 / (d2_k + 1e-6)
            pred = float(np.dot(w, Y_k) / np.sum(w))
            return pred, unc

        while time_ok():
            # Get indices of unevaluated points
            unevaluated_indices = np.where(~evaluated_mask)[0]
            if len(unevaluated_indices) == 0:
                # If all Sobol points are evaluated, break and go to polish
                break
            
            # Compute acquisition scores for all unevaluated points
            # To save time, we can subsample if the pool is huge, but 800 is small.
            # However, computing k-NN for 800 points * 500 steps = 400k distance calcs.
            # This is fast in NumPy if vectorized, but here we loop.
            # Let's optimize: only compute for a subset if needed, or vectorize.
            
            # Vectorized k-NN is complex for variable K, so we loop but it's fast enough for 5D.
            # 800 points * 500 steps = 400,000 evaluations of knn_predict.
            # Each knn_predict does a distance calc to ~100 points. 40M ops. < 1s.
            
            scores = np.zeros(len(unevaluated_indices))
            for i, idx in enumerate(unevaluated_indices):
                x_q = sobol_pts[idx]
                pred, unc = knn_predict_and_uncertainty(x_q, X_eval, Y_eval, K)
                # Acquisition: Minimize pred, maximize unc
                # Score = -pred + beta * unc
                scores[i] = -pred + beta * unc
            
            # Select the best score
            best_local_idx = np.argmax(scores)
            selected_idx = unevaluated_indices[best_local_idx]
            
            # Evaluate
            evaluate(sobol_pts[selected_idx])
            evaluated_mask[selected_idx] = True
            
            # Decay beta
            # Linear decay from 2.0 to 0.0 over the budget
            progress = calls_made / budget
            beta = 2.0 * (1.0 - progress)
            
            # If beta is very small, we are mostly exploiting.
            # We can also check if the best point has improved recently to adjust beta.
            # For now, simple linear decay.

        # 4. Local Polish (Nelder-Mead)
        # Use remaining budget to refine the best point
        if time_ok() and best_x is not None:
            nm_x = best_x.copy()
            nm_f = best_f
            nm_sigma = 0.05
            
            # Build simplex
            simplex = [nm_x.copy()]
            for d in range(dim):
                p = nm_x.copy()
                p[d] += nm_sigma
                simplex.append(p)
            simplex = np.array(simplex)
            fvals = [nm_f]
            for i in range(1, dim + 1):
                if not time_ok() or calls_made >= budget:
                    break
                fvals.append(evaluate(simplex[i]))
            
            # Nelder-Mead iterations
            for it in range(50):
                if not time_ok() or calls_made >= budget:
                    break
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = [fvals[i] for i in order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                if calls_made >= budget:
                    break
                fr = evaluate(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    if calls_made >= budget:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        break
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    if calls_made >= budget:
                        break
                    fc = evaluate(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if calls_made >= budget:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = evaluate(simplex[i])
                
                if max(fvals) - min(fvals) < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
