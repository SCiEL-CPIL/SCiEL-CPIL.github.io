import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    def check_validity(xs, ys, rs, tol=1e-9):
        """Check if circles are valid (inside square, non-overlapping)."""
        for i in range(N):
            if xs[i] < rs[i] - tol or xs[i] > 1.0 - rs[i] + tol:
                return False
            if ys[i] < rs[i] - tol or ys[i] > 1.0 - rs[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist_sq = dx * dx + dy * dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist * min_dist - tol * (min_dist * min_dist):
                    return False
        return True

    def relax(xs, ys, rs, max_iter=100, damp=0.9):
        """
        Position-based dynamics relaxation.
        Returns (xs, ys, converged).
        """
        xs = xs.copy()
        ys = ys.copy()
        for _ in range(max_iter):
            moved = False

            # Boundary constraints
            for i in range(N):
                r = rs[i]
                if xs[i] < r:
                    xs[i] = r
                    moved = True
                elif xs[i] > 1.0 - r:
                    xs[i] = 1.0 - r
                    moved = True
                if ys[i] < r:
                    ys[i] = r
                    moved = True
                elif ys[i] > 1.0 - r:
                    ys[i] = 1.0 - r
                    moved = True

            # Pairwise non-overlap constraints
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    dist_sq = dx * dx + dy * dy
                    min_dist = rs[i] + rs[j]

                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-15 else 1e-8
                        overlap = min_dist - dist
                        if dist < 1e-9:
                            angle = np.random.rand() * 2 * np.pi
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist

                        push = overlap * 0.5 * damp
                        xs[i] -= dx * push
                        ys[i] -= dy * push
                        xs[j] += dx * push
                        ys[j] += dy * push
                        moved = True

            if not moved:
                return xs, ys, True

        return xs, ys, False

    def initial_grid(n):
        """Create initial grid placement."""
        # 5x5 grid + 1 in center
        xs = np.zeros(n)
        ys = np.zeros(n)
        rs = np.zeros(n)
        
        # 5x5 grid
        for i in range(25):
            row = i // 5
            col = i % 5
            xs[i] = (col + 0.5) / 5.0
            ys[i] = (row + 0.5) / 5.0
            rs[i] = 0.5 / 5.0 - 1e-4
        
        # 26th circle in the center
        xs[25] = 0.5
        ys[25] = 0.5
        rs[25] = 0.5 / 5.0 - 1e-4
        
        return xs, ys, rs

    def initial_random(n):
        """Create random initial placement."""
        xs = np.random.rand(n) * 0.8 + 0.1
        ys = np.random.rand(n) * 0.8 + 0.1
        rs = np.full(n, 0.05)
        return xs, ys, rs

    def grow_and_relax(xs, ys, rs, growth_factor, max_iter=50):
        """
        Try to grow all radii by growth_factor, then relax.
        Returns (xs, ys, rs, success).
        """
        new_rs = rs * growth_factor
        if np.max(new_rs) > 0.5:
            return xs, ys, rs, False

        xs_new, ys_new, converged = relax(xs, ys, new_rs, max_iter)

        if converged and check_validity(xs_new, ys_new, new_rs):
            return xs_new, ys_new, new_rs, True

        return xs, ys, rs, False

    def polish(xs, ys, rs):
        """
        Final polish using scipy.optimize.minimize with penalty method.
        """
        if time.time() > deadline - 10:
            return xs, ys, rs

        def penalty_func(params):
            xs_ = params[:N]
            ys_ = params[N:2*N]
            rs_ = params[2*N:3*N]
            
            # Objective: minimize -sum(rs) + penalty
            score = -np.sum(rs_)
            
            # Boundary penalty
            pen = 0.0
            for i in range(N):
                if xs_[i] < rs_[i]:
                    pen += (rs_[i] - xs_[i])**2
                if xs_[i] > 1.0 - rs_[i]:
                    pen += (xs_[i] - (1.0 - rs_[i]))**2
                if ys_[i] < rs_[i]:
                    pen += (rs_[i] - ys_[i])**2
                if ys_[i] > 1.0 - rs_[i]:
                    pen += (ys_[i] - (1.0 - rs_[i]))**2
            
            # Overlap penalty
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs_[j] - xs_[i]
                    dy = ys_[j] - ys_[i]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs_[i] + rs_[j]
                    if dist < min_dist:
                        pen += (min_dist - dist)**2
            
            return score + 100.0 * pen

        # Initial params
        params0 = np.concatenate([xs, ys, rs])
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
        for i in range(N):
            bounds.append((0.0, 1.0)) # y
        for i in range(N):
            bounds.append((0.0, 0.5)) # r

        try:
            res = minimize(penalty_func, params0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
            if res.success:
                new_xs = res.x[:N]
                new_ys = res.x[N:2*N]
                new_rs = res.x[2*N:3*N]
                if check_validity(new_xs, new_ys, new_rs):
                    if np.sum(new_rs) > np.sum(rs):
                        return new_xs, new_ys, new_rs
        except Exception:
            pass
            
        return xs, ys, rs

    best_score = -1.0
    best_xs = None
    best_ys = None
    best_rs = None

    # Try a few initializations
    inits = [
        initial_grid(N),
        initial_random(N),
        initial_random(N)
    ]

    for xs_init, ys_init, rs_init in inits:
        if time.time() > deadline - 20:
            break

        xs, ys, rs = xs_init.copy(), ys_init.copy(), rs_init.copy()
        
        # Initial relax
        xs, ys, _ = relax(xs, ys, rs, max_iter=100)
        
        # Growth loop
        growth_factor = 1.005
        min_growth = 1.0001
        
        while time.time() < deadline - 30:
            xs, ys, rs, success = grow_and_relax(xs, ys, rs, growth_factor, max_iter=50)
            if success:
                current_score = np.sum(rs)
                if current_score > best_score:
                    best_score = current_score
                    best_xs = xs.copy()
                    best_ys = ys.copy()
                    best_rs = rs.copy()
                growth_factor = min(1.005, growth_factor * 1.01)
            else:
                growth_factor *= 0.9
                if growth_factor < min_growth:
                    break

    # Final polish on the best result
    if best_xs is not None and time.time() < deadline - 5:
        best_xs, best_ys, best_rs = polish(best_xs, best_ys, best_rs)

    # Final validation
    if not check_validity(best_xs, best_ys, best_rs):
        # Fallback
        xs_f, ys_f, rs_f = initial_grid(N)
        xs_f, ys_f, _ = relax(xs_f, ys_f, rs_f, max_iter=200)
        best_xs, best_ys, best_rs = xs_f, ys_f, rs_f

    return np.column_stack((best_xs, best_ys, best_rs))
