import numpy as np
import time
from scipy.optimize import minimize

N = 32
TIME_LIMIT = 145.0  # Evaluator kills at 150s, keep 5s margin

def construct():
    start_time = time.time()
    
    def _elapsed():
        return time.time() - start_time

    def _is_valid(x, y, r, eps=1e-7):
        if np.any(x < -eps) or np.any(x > 1.0 + eps):
            return False
        if np.any(y < -eps) or np.any(y > 1.0 + eps):
            return False
        if np.any(r < -eps):
            return False
        if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
            return False
        if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                if d < (r[i] + r[j]) - eps:
                    return False
        return True

    def _sum_r(r):
        return float(np.sum(r))

    def _make_valid(x, y, r, max_iter=500):
        """Iteratively shrink radii to satisfy all constraints strictly."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        # Initial clamp to boundaries
        for i in range(N):
            r[i] = min(r[i], x[i] * 0.9999, (1.0 - x[i]) * 0.9999,
                       y[i] * 0.9999, (1.0 - y[i]) * 0.9999)
        
        for _ in range(max_iter):
            if _elapsed() > TIME_LIMIT - 2.0:
                break
            changed = False
            # Pairwise overlap check and shrink
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.hypot(dx, dy)
                    min_d = r[i] + r[j]
                    if d < min_d - 1e-10:
                        # Shrink both proportionally to fit
                        if min_d > 0:
                            shrink_factor = (d / min_d) * 0.9999
                            r[i] *= shrink_factor
                            r[j] *= shrink_factor
                        else:
                            r[i] = 1e-6
                            r[j] = 1e-6
                        changed = True
            if not changed:
                break
                
        # Final safety clamp
        for i in range(N):
            r[i] = min(r[i], x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            if r[i] < 1e-9:
                r[i] = 1e-9
        return x, y, r

    def _hex_init():
        """
        Hexagonal packing initialization tuned for N=32.
        Rows: 6, 5, 6, 5, 6, 4 = 32 circles.
        """
        rows = [6, 5, 6, 5, 6, 4]
        x = []
        y = []
        r = []
        total_rows = len(rows)
        row_h = 1.0 / total_rows
        
        for row_idx, count in enumerate(rows):
            ry = (row_idx + 0.5) * row_h
            # Offset odd rows to create hex pattern
            offset = 0.0
            if row_idx % 2 == 1:
                offset = 0.5 / count
            
            for c in range(count):
                cx = (c + 0.5 + offset) / count
                # Initial radius estimate: half the row height, clamped by boundaries
                cr = 0.48 * row_h
                cr = min(cr, cx * 0.99, (1.0 - cx) * 0.99, ry * 0.99, (1.0 - ry) * 0.99)
                x.append(cx)
                y.append(ry)
                r.append(cr)
        
        # Ensure we have exactly N
        x = x[:N]
        y = y[:N]
        r = r[:N]
        while len(x) < N:
            x.append(0.5)
            y.append(0.5)
            r.append(0.02)
            
        return np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

    def _rect_init():
        """Rectangular grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        x = []
        y = []
        r = []
        for i in range(side):
            for j in range(side):
                if len(x) < N:
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    cr = 0.48 / side
                    cr = min(cr, cx * 0.99, (1.0 - cx) * 0.99, cy * 0.99, (1.0 - cy) * 0.99)
                    x.append(cx)
                    y.append(cy)
                    r.append(cr)
        return np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

    def _random_init(seed):
        """Random initialization with fixed seed for reproducibility."""
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = np.full(N, 0.06)
        return x, y, r

    def _objective(params, soft_eps=1e-4, mu=50.0):
        """
        Penalty-based objective.
        Minimizes: -sum(r) + mu * (overlap_pen + bound_pen)
        Uses smooth max for differentiability.
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        obj = -np.sum(r)
        
        # Pairwise overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = r[:, None] + r[None, :]
        diff = min_dist - dist
        smooth_max = 0.5 * (diff + np.sqrt(diff*diff + soft_eps*soft_eps))
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlap_pen = np.sum(smooth_max[mask])
        
        # Boundary penalties
        pen = np.zeros(4*N)
        pen[0:N] = r - x          # left
        pen[N:2*N] = x - (1.0 - r)  # right
        pen[2*N:3*N] = r - y       # bottom
        pen[3*N:4*N] = y - (1.0 - r)  # top
        bound_pen = np.sum(0.5 * (pen + np.sqrt(pen*pen + soft_eps*soft_eps)))
        
        return obj + mu * (overlap_pen + bound_pen)

    def _objective_tight(params):
        """High precision objective for final polish."""
        return _objective(params, soft_eps=1e-6, mu=200.0)

    def _push_apart(x, y, r, iterations=50):
        """
        Physics-based push apart to resolve overlaps quickly.
        Moves centers, not radii.
        """
        x = x.copy()
        y = y.copy()
        for _ in range(iterations):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.hypot(dx, dy)
                    min_d = r[i] + r[j]
                    if d < min_d - 1e-10:
                        if d < 1e-10:
                            dx, dy = 1e-3, 0.0
                            d = np.hypot(dx, dy)
                        # Push apart
                        overlap = min_d - d
                        nx, ny = dx/d, dy/d
                        push = overlap * 0.5 * 0.8
                        x[i] += nx * push
                        y[i] += ny * push
                        x[j] -= nx * push
                        y[j] -= ny * push
                        moved = True
            # Clamp to bounds
            for i in range(N):
                x[i] = np.clip(x[i], r[i]*0.99, 1.0 - r[i]*0.99)
                y[i] = np.clip(y[i], r[i]*0.99, 1.0 - r[i]*0.99)
            if not moved:
                break
        return x, y

    def _optimize(x0, time_budget=20):
        """Run L-BFGS-B optimization with time guard."""
        try:
            res = minimize(_objective, x0, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxfun': 50000})
            return res.x
        except Exception:
            return x0

    def _tight_polish(params):
        """Final high-precision polish."""
        try:
            res = minimize(_objective_tight, params, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12,
                                    'maxfun': 100000})
            return res.x
        except Exception:
            return params

    def _local_search(params, max_iter=200):
        """
        Local search with random perturbations to escape local minima.
        """
        best_params = params.copy()
        best_score = -_sum_r(best_params[2*N:3*N])
        rng = np.random.RandomState(12345)
        
        for i in range(max_iter):
            if _elapsed() > TIME_LIMIT - 10:
                break
            
            # Annealing scale
            scale = 0.05 * (1 - i / max_iter)
            perturbed = best_params + rng.normal(0, scale, size=3*N)
            
            # Push apart to resolve overlaps before optimizing
            px = perturbed[:N].copy()
            py = perturbed[N:2*N].copy()
            pr = perturbed[2*N:3*N].copy()
            px, py = _push_apart(px, py, pr, iterations=20)
            p_opt = np.concatenate([px, py, pr])
            
            # Quick optimize
            try:
                res = minimize(_objective, p_opt, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8,
                                        'maxfun': 5000})
                p_new = res.x
                score = -_sum_r(p_new[2*N:3*N])
                
                # Check validity roughly before accepting
                if _is_valid(p_new[:N], p_new[N:2*N], p_new[2*N:3*N], eps=1e-5):
                    if score < best_score:
                        best_score = score
                        best_params = p_new
            except Exception:
                pass
                
        return best_params

    # --- Main Execution ---
    
    # 1. Generate Initial Guesses
    x_hex, y_hex, r_hex = _hex_init()
    x_rect, y_rect, r_rect = _rect_init()
    
    guesses = []
    guesses.append(np.concatenate([x_hex, y_hex, r_hex]))
    guesses.append(np.concatenate([x_rect, y_rect, r_rect]))
    
    # Add a few random starts
    for s in [1, 2, 3]:
        x_r, y_r, r_r = _random_init(s)
        # Pre-process with push apart
        x_r, y_r = _push_apart(x_r, y_r, r_r, iterations=50)
        guesses.append(np.concatenate([x_r, y_r, r_r]))

    best_solution = None
    best_score = 0.0

    # 2. Phase 1: Coarse Optimization from each start
    for v0 in guesses:
        if _elapsed() > TIME_LIMIT - 30:
            break
            
        v_opt = _optimize(v0)
        
        # Make valid
        x_v, y_v, r_v = _make_valid(v_opt[:N], v_opt[N:2*N], v_opt[2*N:3*N])
        v_valid = np.concatenate([x_v, y_v, r_v])
        
        if _is_valid(x_v, y_v, r_v):
            score = _sum_r(r_v)
            if score > best_score:
                best_score = score
                best_solution = v_valid

    if best_solution is None:
        # Fallback: return a valid trivial solution
        x_f, y_f, r_f = _hex_init()
        x_f, y_f, r_f = _make_valid(x_f, y_f, r_f)
        return np.column_stack((x_f, y_f, r_f))

    # 3. Phase 2: Local Search to escape local minima
    if _elapsed() < TIME_LIMIT - 20:
        best_solution = _local_search(best_solution, max_iter=150)
        
        # Re-validate and update score
        x_v, y_v, r_v = _make_valid(best_solution[:N], best_solution[N:2*N], best_solution[2*N:3*N])
        v_valid = np.concatenate([x_v, y_v, r_v])
        if _is_valid(x_v, y_v, r_v):
             score = _sum_r(r_v)
             if score > best_score:
                 best_score = score
                 best_solution = v_valid

    # 4. Phase 3: High-precision Polish
    if _elapsed() < TIME_LIMIT - 5:
        v_polished = _tight_polish(best_solution)
        x_v, y_v, r_v = _make_valid(v_polished[:N], v_polished[N:2*N], v_polished[2*N:3*N])
        v_valid = np.concatenate([x_v, y_v, r_v])
        
        if _is_valid(x_v, y_v, r_v):
            score = _sum_r(r_v)
            if score > best_score:
                best_score = score
                best_solution = v_valid

    # Final Output
    x_final, y_final, r_final = best_solution[:N], best_solution[N:2*N], best_solution[2*N:3*N]
    return np.column_stack((x_final, y_final, r_final))

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Sum of radii: {np.sum(res[:, 2]):.6f}")
    print(f"Valid: {_is_valid(res[:, 0], res[:, 1], res[:, 2])}")
