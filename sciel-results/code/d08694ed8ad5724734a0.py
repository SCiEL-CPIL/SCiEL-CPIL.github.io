import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 40s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_f = np.inf
        eval_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            if eval_count >= budget or not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global random sampling to find good basins
        # Use 20 points for good coverage without wasting too much budget
        n_initial = 20
        candidates = []
        for i in range(n_initial):
            if eval_count >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x.copy()))

        if not candidates:
            x = rng.uniform(-5, 5, n)
            evaluate(x)
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])

        # Phase 2: Multi-start Nelder-Mead local refinement on top candidates
        # Use top 3 candidates for multi-start
        top_k = min(3, len(candidates))
        remaining = budget - eval_count
        per_candidate = max(30, remaining // max(1, top_k))

        for rank in range(top_k):
            if eval_count >= budget or not time_ok():
                break

            f0, x0 = candidates[rank]
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            # Build simplex with moderate initial steps
            simplex = [x0.copy()]
            for j in range(n):
                point = x0.copy()
                delta = 0.3 if x0[j] + 0.3 <= 5 else -0.3
                point[j] += delta
                point = np.clip(point, -5, 5)
                simplex.append(point)

            fvals = [f0]
            for pt in simplex[1:]:
                if eval_count >= budget or not time_ok():
                    break
                fx = evaluate(pt)
                if fx is None:
                    break
                fvals.append(fx)

            if len(fvals) != len(simplex):
                continue

            max_nm_iter = per_candidate - (len(fvals) - 1)
            for _ in range(max(0, max_nm_iter)):
                if eval_count >= budget or not time_ok():
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if abs(fvals[-1] - fvals[0]) < 1e-10:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                if eval_count >= budget or not time_ok():
                    break
                fr = evaluate(xr)
                if fr is None:
                    break

                if fr < fvals[0]:
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    if eval_count >= budget or not time_ok():
                        break
                    fe = evaluate(xe)
                    if fe is None:
                        simplex[-1], fvals[-1] = xr, fr
                        break
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    if fr < fvals[-1]:
                        xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                    else:
                        xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                    if eval_count >= budget or not time_ok():
                        break
                    fc = evaluate(xc)
                    if fc is None:
                        break
                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = np.clip(simplex[0] + sigma_shrink * (simplex[i] - simplex[0]), -5, 5)
                            if eval_count >= budget or not time_ok():
                                break
                        if eval_count >= budget or not time_ok():
                            break
                        fvals_new = []
                        for pt in simplex:
                            if pt is simplex[0]:
                                fvals_new.append(fvals[0])
                            else:
                                fx = evaluate(pt)
                                if fx is None:
                                    break
                                fvals_new.append(fx)
                        if len(fvals_new) != len(simplex):
                            break
                        fvals = fvals_new

        # Phase 3: Coordinate descent polish on the global best
        if best_x is not None and eval_count < budget and time_ok():
            x = best_x.copy()
            f = best_f
            step = 0.1
            min_step = 1e-6
            max_cd_iter = budget - eval_count
            
            for _ in range(max_cd_iter):
                if not time_ok() or eval_count >= budget:
                    break
                
                improved = False
                for j in range(n):
                    if eval_count >= budget or not time_ok():
                        break
                    
                    # Try positive direction
                    x_plus = x.copy()
                    x_plus[j] += step
                    x_plus = np.clip(x_plus, -5, 5)
                    if eval_count >= budget or not time_ok():
                        break
                    f_plus = evaluate(x_plus)
                    if f_plus is None:
                        break
                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        improved = True
                    else:
                        # Try negative direction
                        x_minus = x.copy()
                        x_minus[j] -= step
                        x_minus = np.clip(x_minus, -5, 5)
                        if eval_count >= budget or not time_ok():
                            break
                        f_minus = evaluate(x_minus)
                        if f_minus is None:
                            break
                        if f_minus < f:
                            x = x_minus
                            f = f_minus
                            improved = True
                
                if not improved:
                    step *= 0.5
                    if step < min_step:
                        break
                else:
                    # Keep step or increase slightly if improving
                    step = min(step * 1.1, 0.5)

        # Phase 4: Final aggressive local polish with adaptive step size
        if best_x is not None and eval_count < budget and time_ok():
            x = best_x.copy()
            f = best_f
            sigma = 0.1
            min_sigma = 1e-6
            for _ in range(budget - eval_count):
                if not time_ok() or eval_count >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = evaluate(y)
                if fy is None:
                    break
                if fy < f:
                    x, f = y, fy
                    sigma = min(sigma * 1.3, 2.0)
                else:
                    sigma *= 0.8
                    if sigma < min_sigma:
                        break

        return best_x, best_f
