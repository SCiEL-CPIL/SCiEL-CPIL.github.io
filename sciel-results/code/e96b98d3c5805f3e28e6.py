import numpy as np
import time

class Optimizer:
    """
    Nelder-Mead Simplex optimizer for BBOB f12 (Bent Cigar).
    
    Strategy:
    1. Multi-start initialization to handle the unknown shift/rotation.
    2. Nelder-Mead Simplex method for local refinement.
       - Adapts simplex shape to the anisotropic valley of Bent Cigar.
       - More efficient than coordinate-wise search for elongated minima.
    3. Budget management with explicit deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 330s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            
            x = np.clip(np.asarray(x, dtype=float).flatten(), -5, 5)
            
            try:
                f = float(func(x))
            except Exception:
                return float('inf')

            if not np.isfinite(f):
                return float('inf')

            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
            return f

        def nelder_mead(start_x, initial_f=None, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            """
            Nelder-Mead Simplex method.
            """
            nonlocal eval_count
            
            if eval_count >= budget - 2:
                return

            x = np.clip(start_x, -5, 5).copy()
            if initial_f is not None:
                f = initial_f
            else:
                f = eval_func(x)
                if eval_count >= budget:
                    return

            # Initialize simplex
            simplex = [x]
            fvals = [f]
            
            # Create other vertices by perturbing each coordinate
            for i in range(d):
                xi = x.copy()
                # Initial step size: 5% of the range, or 0.05 if at origin
                step = 0.05 if abs(x[i]) < 1e-10 else 0.05 * abs(x[i])
                # Ensure step is not too small
                step = max(step, 0.01)
                xi[i] += step
                simplex.append(xi)
                fi = eval_func(xi)
                if eval_count >= budget:
                    return
                fvals.append(fi)

            # Main loop
            max_iter = 1000
            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                f_min = fvals[0]
                f_max = fvals[-1]
                if abs(f_max - f_min) < 1e-10 * max(1.0, abs(f_min)):
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = eval_func(x_ref)
                if eval_count >= budget:
                    break

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = eval_func(x_exp)
                    if eval_count >= budget:
                        break

                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + 0.5 * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - 0.5 * (simplex[-1] - centroid)
                    
                    x_con = np.clip(x_con, -5, 5)
                    f_con = eval_func(x_con)
                    if eval_count >= budget:
                        break

                    if f_con <= min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = eval_func(simplex[i])
                            if eval_count >= budget:
                                break
                        if eval_count >= budget:
                            break

        # --- Phase 1: Multi-Start Initialization ---
        rng = np.random.default_rng(42)
        
        # Generate initial candidates
        init_points = [np.zeros(d)]
        for _ in range(9):
            init_points.append(rng.uniform(-4, 4, d))
            
        # Evaluate all initial points
        point_scores = []
        for p in init_points:
            if eval_count >= budget:
                break
            fp = eval_func(p)
            point_scores.append((fp, p))
            
        # Sort by function value
        point_scores.sort(key=lambda x: x[0])
        
        # --- Phase 2: Local Refinement on the best starting points ---
        top_k = min(3, len(point_scores))
        
        for i in range(top_k):
            if eval_count >= budget - 20:
                break
            
            nelder_mead(point_scores[i][1], point_scores[i][0])
            
        return best_x, best_f
