"""Surrogate-assisted multi-start optimizer with immediate local exploitation for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Surrogate-assisted optimizer for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        X_all = []
        Y_all = []

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            X_all.append(x.copy())
            Y_all.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return calls_made < budget and (time.time() - start_time) < (deadline - start_time)

        # ---- Phase 1: Initial exploration ----
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = int(index)
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_sequence(n, dim):
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
            pts = []
            for i in range(1, n + 1):
                pts.append(np.array([halton(i, b) for b in bases]))
            return np.array(pts) * 10.0 - 5.0

        # Structured points
        structured = [np.zeros(dim)]
        for d in range(dim):
            for v in [-3.0, 0.0, 3.0]:
                p = np.zeros(dim)
                p[d] = v
                structured.append(p)
        for mask in [(1, 0, 1, 0, 1), (0, 1, 0, 1, 0), (1, 1, 0, 0, 1), (0, 0, 1, 1, 0),
                     (1, 0, 0, 1, 0), (0, 1, 1, 0, 1), (1, 1, 1, 0, 0), (0, 0, 1, 1, 1)]:
            p = np.array([(-2.5 if m else 2.5) for m in mask])
            structured.append(p)

        n_halton = 40
        init_pts = halton_sequence(n_halton, dim)

        all_init = np.vstack([np.array(structured), init_pts])
        all_init = np.clip(all_init, -5.0, 5.0)

        unique_init = [all_init[0]]
        for p in all_init[1:]:
            if np.all([np.linalg.norm(p - u) > 0.15 for u in unique_init]):
                unique_init.append(p)
                if len(unique_init) >= 50:
                    break

        for p in unique_init:
            if not time_ok():
                break
            evaluate(p)

        # ---- k-NN Surrogate ----
        K = 5

        def knn_predict(x):
            if len(X_all) == 0:
                return 0.0
            X_arr = np.array(X_all)
            Y_arr = np.array(Y_all)
            n = len(X_arr)
            d2 = np.sum((X_arr - x) ** 2, axis=1)
            if n <= K:
                w = 1.0 / (d2 + 1e-6)
                return float(np.dot(w, Y_arr) / np.sum(w))
            idx = np.argpartition(d2, K)[:K]
            d2_k = d2[idx]
            Y_k = Y_arr[idx]
            w = 1.0 / (d2_k + 1e-6)
            return float(np.dot(w, Y_k) / np.sum(w))

        def max_min_distance(x):
            if len(X_all) == 0:
                return 10.0
            X_arr = np.array(X_all)
            d2 = np.sum((X_arr - x) ** 2, axis=1)
            if len(X_arr) <= K:
                return float(np.sqrt(np.max(d2)))
            idx = np.argpartition(d2, K)[:K]
            return float(np.sqrt(np.max(d2[idx])))

        # ---- Local Search Helper: Nelder-Mead ----
        def nelder_mead(start_x, max_iters=5, shrink_factor=0.5, expand_factor=2.0):
            """
            Runs a short Nelder-Mead from start_x.
            Returns the best point and value found during this local search.
            """
            if not time_ok() or calls_made >= budget - max_iters:
                return start_x, func(start_x) if False else 0.0 # Placeholder, we use existing best_f if no calls made? 
                # Actually, we should just return the input if we can't run.
            
            # Initialize simplex
            simplex = [start_x.copy()]
            fvals = [evaluate(start_x)]
            
            # If the first eval failed or we ran out of budget
            if calls_made >= budget:
                return start_x, fvals[0]

            step_size = 0.5
            for d in range(dim):
                if not time_ok() or calls_made >= budget:
                    break
                p = start_x.copy()
                p[d] += step_size
                p = np.clip(p, -5.0, 5.0)
                simplex.append(p)
                fvals.append(evaluate(p))
            
            # If we couldn't even build the simplex
            if len(simplex) < dim + 1:
                return start_x, min(fvals) if fvals else 0.0

            for _ in range(max_iters):
                if not time_ok() or calls_made >= budget:
                    break
                
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = evaluate(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    if not time_ok() or calls_made >= budget:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        break
                    xe = centroid + expand_factor * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
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
                    if not time_ok() or calls_made >= budget:
                        break
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = evaluate(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if not time_ok() or calls_made >= budget:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = evaluate(simplex[i])
                
                # Check convergence
                if max(fvals) - min(fvals) < 1e-8:
                    break
            
            # Find best in simplex
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # ---- Phase 2: Main Optimization Loop ----
        x_current = best_x.copy() if best_x is not None else np.zeros(dim)
        f_current = best_f
        epsilon = 0.5
        n_trials = 0
        stall_count = 0
        last_best_f = best_f

        while time_ok():
            n_trials += 1
            # Decay epsilon
            progress = min(1.0, n_trials / max(1, (budget - calls_made) * 0.7))
            epsilon = 0.6 * (1.0 - progress) + 0.1 * progress

            # Generate candidates
            candidates = []
            
            # 1. Local candidates around current best
            for _ in range(3):
                c = x_current + 0.5 * rng.standard_normal(dim)
                c = np.clip(c, -5.0, 5.0)
                candidates.append(c)
            
            # 2. Candidate near global best
            if best_x is not None:
                c = best_x + 0.3 * rng.standard_normal(dim)
                c = np.clip(c, -5.0, 5.0)
                candidates.append(c)
                
            # 3. Random exploration
            for _ in range(2):
                c = rng.uniform(-5.0, 5.0, size=dim)
                candidates.append(c)

            # 4. Surrogate-guided
            if len(X_all) >= K + 2:
                for _ in range(3):
                    # Sample around best or random
                    base = best_x if best_x is not None else rng.uniform(-5, 5, dim)
                    c = base + 1.0 * rng.standard_normal(dim)
                    c = np.clip(c, -5.0, 5.0)
                    candidates.append(c)

            if len(candidates) == 0:
                break

            Xq = np.array(candidates)
            
            # Compute scores
            preds = np.array([knn_predict(c) for c in Xq])
            expl_scores = np.array([max_min_distance(c) for c in Xq])
            
            # Normalize
            if np.max(preds) - np.min(preds) > 1e-10:
                preds_norm = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))
            else:
                preds_norm = np.zeros_like(preds)
            
            if np.max(expl_scores) - np.min(expl_scores) > 1e-10:
                expl_norm = (expl_scores - np.min(expl_scores)) / (np.max(expl_scores) - np.min(expl_scores))
            else:
                expl_norm = np.zeros_like(expl_scores)

            # Epsilon-greedy selection
            if rng.random() < epsilon:
                idx = int(np.argmax(expl_norm))
            else:
                idx = int(np.argmin(preds_norm))

            x_candidate = Xq[idx]
            
            # Evaluate candidate
            f_cand = evaluate(x_candidate)
            
            # Check if we found a new best
            if f_cand < best_f:
                best_f = f_cand
                best_x = x_candidate.copy()
                stall_count = 0
                # **Structural Change**: Immediately exploit the new basin
                # Run a short Nelder-Mead to polish this point
                if time_ok() and calls_made < budget - 5:
                    nm_x, nm_f = nelder_mead(best_x, max_iters=5)
                    # Update best if NM found something better
                    # Note: nelder_mead updates global best_x/best_f via evaluate()
                    # So we just need to ensure our local variables are synced if needed
                    # But since evaluate() updates nonlocals, best_x is already updated.
                    pass
            else:
                stall_count += 1
                # Update current point if it's better than current local best
                if f_cand < f_current:
                    x_current = x_candidate.copy()
                    f_current = f_cand

            # Restart logic: If stalled for a long time, jump to a random point
            if stall_count > 30:
                x_current = rng.uniform(-4, 4, size=dim)
                f_current = np.inf
                stall_count = 0
                # Optional: Evaluate the jump point
                if time_ok():
                    evaluate(x_current)

        # ---- Phase 3: Final Polish on the Global Best ----
        if time_ok() and best_x is not None and calls_made < budget - 10:
            # Run a slightly longer NM on the final best
            nm_x, nm_f = nelder_mead(best_x, max_iters=15)
            
            # Coordinate Descent on the result
            if time_ok() and best_x is not None:
                x_polish = best_x.copy()
                f_polish = best_f
                sigma_p = 0.05
                
                for _ in range(20):
                    if not time_ok() or calls_made >= budget:
                        break
                    improved = False
                    for d in range(dim):
                        if not time_ok() or calls_made >= budget:
                            break
                        for step in [sigma_p, -sigma_p]:
                            y = x_polish.copy()
                            y[d] += step
                            y = np.clip(y, -5.0, 5.0)
                            fy = evaluate(y)
                            if fy < f_polish - 1e-12:
                                x_polish = y.copy()
                                f_polish = fy
                                improved = True
                                break
                    if improved:
                        sigma_p = max(0.002, sigma_p * 0.8)
                    else:
                        sigma_p = max(0.001, sigma_p * 0.7)
                        # If no improvement, try random perturbation
                        y = x_polish + sigma_p * rng.standard_normal(dim)
                        y = np.clip(y, -5.0, 5.0)
                        fy = evaluate(y)
                        if fy < f_polish:
                            x_polish = y
                            f_polish = fy

        return best_x, best_f
# EVOLVE-BLOCK-END
