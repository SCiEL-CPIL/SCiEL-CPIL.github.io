import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            return func(x)

        def check_time():
            return calls_made < budget and time.time() < deadline - 3

        def coordinate_descent(x, fx, max_evals=30):
            """Efficient adaptive coordinate descent with bidirectional search."""
            nonlocal best_f, best_x
            step = np.ones(dim) * 0.5
            for _ in range(max_evals):
                if not check_time():
                    break
                improved = False
                for d in range(dim):
                    if not check_time():
                        break
                    # Try positive and negative, pick the better one
                    x_pos = x.copy()
                    x_pos[d] = np.clip(x_pos[d] + step[d], -5, 5)
                    x_neg = x.copy()
                    x_neg[d] = np.clip(x_neg[d] - step[d], -5, 5)
                    
                    if np.allclose(x_pos, x) and np.allclose(x_neg, x):
                        continue
                    
                    f_pos = safe_func(x_pos) if not np.allclose(x_pos, x) else np.inf
                    if not check_time():
                        break
                    f_neg = safe_func(x_neg) if not np.allclose(x_neg, x) else np.inf
                    if not check_time():
                        break
                    
                    if f_pos < fx:
                        x = x_pos
                        fx = f_pos
                        step[d] = min(step[d] * 1.3, 3.0)
                        improved = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    elif f_neg < fx:
                        x = x_neg
                        fx = f_neg
                        step[d] = min(step[d] * 1.3, 3.0)
                        improved = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                if not improved:
                    step *= 0.4
                    if np.max(step) < 1e-4:
                        break
            return x, fx

        def nelder_mead(x0, fx0, max_evals=40):
            """Nelder-Mead simplex method for local refinement."""
            nonlocal best_f, best_x
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [fx0]
            for d in range(dim):
                x_new = x0.copy()
                x_new[d] = np.clip(x_new[d] + 0.5, -5, 5)
                simplex.append(x_new)
                fvals.append(safe_func(x_new))
                if not check_time():
                    return x0, fx0

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            max_simplex_iters = max(1, max_evals // (2 * dim + 1))

            for _ in range(max_simplex_iters):
                if not check_time():
                    break
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                centroid = np.clip(centroid, -5, 5)

                # Reflection
                x_ref = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                f_ref = safe_func(x_ref)
                if not check_time():
                    break

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = np.clip(centroid + gamma * (x_ref - centroid), -5, 5)
                    f_exp = safe_func(x_exp)
                    if not check_time():
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    x_con = np.clip(centroid + rho * (simplex[-1] - centroid), -5, 5)
                    f_con = safe_func(x_con)
                    if not check_time():
                        break
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                            fvals[i] = safe_func(simplex[i])
                            if not check_time():
                                break

                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

            # Return best point from simplex
            min_idx = np.argmin(fvals)
            return simplex[min_idx], fvals[min_idx]

        def fine_polish(x, fx, max_evals=20):
            """High-precision polish using adaptive step sizes and bidirectional search."""
            nonlocal best_f, best_x
            for scale in [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]:
                if not check_time() or calls_made >= budget - 5:
                    break
                step = np.ones(dim) * scale
                for _ in range(4):
                    if not check_time():
                        break
                    improved = False
                    for d in range(dim):
                        if not check_time():
                            break
                        x_pos = x.copy()
                        x_pos[d] = np.clip(x_pos[d] + step[d], -5, 5)
                        x_neg = x.copy()
                        x_neg[d] = np.clip(x_neg[d] - step[d], -5, 5)
                        
                        if np.allclose(x_pos, x) and np.allclose(x_neg, x):
                            continue
                        
                        f_pos = safe_func(x_pos) if not np.allclose(x_pos, x) else np.inf
                        if not check_time():
                            break
                        f_neg = safe_func(x_neg) if not np.allclose(x_neg, x) else np.inf
                        if not check_time():
                            break
                        
                        if f_pos < fx:
                            x = x_pos
                            fx = f_pos
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        elif f_neg < fx:
                            x = x_neg
                            fx = f_neg
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                    if not improved:
                        break
            return x, fx

        # Phase 1: Generate diverse starting points - more starts for Rastrigin
        # Rastrigin has local minima at integer points, so sample near integers
        n_starts = 20
        starts = [np.zeros(dim)]
        # Integer grid points (Rastrigin minima locations)
        for i in range(-4, 5):
            for j in range(-4, 5):
                pt = np.zeros(dim)
                pt[0] = i
                pt[1] = j
                if np.all(np.abs(pt) <= 4) and len(starts) < 10:
                    starts.append(pt)
        # Add points offset from integers to catch rotated minima
        for i in range(-3, 4):
            for j in range(-3, 4):
                pt = np.zeros(dim)
                pt[0] = i + 0.5
                pt[1] = j + 0.5
                if np.all(np.abs(pt) <= 4) and len(starts) < 15:
                    starts.append(pt)
        # Fill remaining with random
        while len(starts) < n_starts:
            starts.append(rng.uniform(-4, 4, dim))

        # Phase 2: For each start, do coordinate descent then Nelder-Mead
        # Track top performers for deeper refinement
        local_results = []
        
        for s_idx, x0 in enumerate(starts):
            if not check_time():
                break

            x = np.clip(x0.copy(), -5, 5)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive coordinate descent - shorter for more starts
            remaining_evals = budget - calls_made
            cd_evals = min(12, max(4, remaining_evals // (n_starts - s_idx + 1)))
            x, fx = coordinate_descent(x, fx, max_evals=cd_evals)

            # Nelder-Mead refinement
            if check_time():
                nm_evals = min(15, budget - calls_made)
                x, fx = nelder_mead(x, fx, max_evals=nm_evals)
            
            local_results.append((fx, x.copy()))

        # Phase 3: Refine the top 3 local minima more deeply
        if local_results and check_time():
            local_results.sort(key=lambda t: t[0])
            for rank, (fx_r, x_r) in enumerate(local_results[:3]):
                if not check_time():
                    break
                # Additional coordinate descent with smaller steps
                x, fx = coordinate_descent(x_r, fx_r, max_evals=10)
                # Nelder-Mead for finer convergence
                if check_time():
                    x, fx = nelder_mead(x, fx, max_evals=10)

        # Phase 4: Final fine-grained polish around the best point
        if best_x is not None and check_time():
            x = best_x.copy()
            fx = best_f
            x, fx = fine_polish(x, fx, max_evals=30)

        return best_x, best_f
