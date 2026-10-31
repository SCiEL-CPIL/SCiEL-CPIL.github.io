"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0

    def unpack(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return x, y, r

    def validate(x, y, r, w, h):
        if np.any(r < 1e-10):
            return False
        if np.any(x < r - 1e-9) or np.any(x > w - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > h - r + 1e-9):
            return False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    def pebble_growth(w, h, xs, ys, max_iter=200):
        """
        Pebble growth: iteratively grow each circle's radius by the minimum
        distance to nearest neighbor or boundary. This naturally enforces
        non-overlap and boundary constraints.
        """
        r = np.zeros(N)
        for _ in range(max_iter):
            # For each circle, compute min distance to boundary and to other circles
            for i in range(N):
                # Distance to boundaries
                d_bound = min(xs[i] - r[i], w - xs[i] - r[i], ys[i] - r[i], h - ys[i] - r[i])
                # Distance to other circles
                d_min = d_bound
                for j in range(N):
                    if i == j:
                        continue
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    d_min = min(d_min, (dist - r[i] - r[j]) / 2.0)
                
                # Grow by half the min distance (symmetric growth)
                growth = max(0, d_min) * 0.5
                r[i] += growth
            
            # Check convergence
            max_growth = max([max(0, min(xs[i] - r[i], w - xs[i] - r[i], ys[i] - r[i], h - ys[i] - r[i])) for i in range(N)])
            if max_growth < 1e-10:
                break
        
        return r

    def pack_and_score(w, h, xs, ys):
        """Run pebble growth and return the sum of radii if valid."""
        r = pebble_growth(w, h, xs, ys)
        if validate(xs, ys, r, w, h):
            return np.sum(r), r
        return 0.0, None

    # Aspect ratio search
    aspect_ratios = np.array([0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25])
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        # Generate multiple initial configurations
        np.random.seed(42)
        for restart in range(12):
            if time.time() > deadline - 5:
                break

            # Different initialization strategies
            if restart < 4:
                # Hexagonal grid with jitter
                ar_wh = w / h
                if ar_wh >= 1.0:
                    n_cols = int(np.round(np.sqrt(N * ar_wh)))
                    n_rows = int(np.ceil(N / n_cols))
                else:
                    n_rows = int(np.round(np.sqrt(N / ar_wh)))
                    n_cols = int(np.ceil(N / n_rows))
                
                while n_cols * n_rows < N:
                    if n_cols <= n_rows:
                        n_cols += 1
                    else:
                        n_rows += 1

                xs = np.zeros(N)
                ys = np.zeros(N)
                cell_w = w / n_cols
                cell_h = h / n_rows
                
                for i in range(N):
                    col = i % n_cols
                    row = i // n_cols
                    xs[i] = (col + 0.5) * cell_w
                    ys[i] = (row + 0.5) * cell_h
                    if row % 2 == 1:
                        xs[i] += 0.5 * cell_w
                        xs[i] = np.clip(xs[i], 0.05, w - 0.05)

                jitter_scale = 0.05 * min(cell_w, cell_h)
                xs += np.random.normal(0, jitter_scale, N)
                ys += np.random.normal(0, jitter_scale, N)
                xs = np.clip(xs, 0.02, w - 0.02)
                ys = np.clip(ys, 0.02, h - 0.02)

            elif restart < 8:
                # Random positions
                xs = np.random.uniform(0.05, w - 0.05, N)
                ys = np.random.uniform(0.05, h - 0.05, N)
            else:
                # Grid with random perturbation
                n_cols = int(np.ceil(np.sqrt(N * w / h)))
                n_rows = int(np.ceil(N / n_cols))
                cell_w = w / n_cols
                cell_h = h / n_rows
                
                xs = np.zeros(N)
                ys = np.zeros(N)
                for i in range(N):
                    col = i % n_cols
                    row = i // n_cols
                    xs[i] = (col + 0.5) * cell_w + np.random.uniform(-0.2*cell_w, 0.2*cell_w)
                    ys[i] = (row + 0.5) * cell_h + np.random.uniform(-0.2*cell_h, 0.2*cell_h)
                xs = np.clip(xs, 0.02, w - 0.02)
                ys = np.clip(ys, 0.02, h - 0.02)

            score, r = pack_and_score(w, h, xs, ys)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([xs, ys, r])

                # Try SLSQP polish on the best configuration
                if time.time() < deadline - 8:
                    try:
                        def objective_polish(vars_):
                            x, y, r_ = unpack(vars_)
                            # Penalty for constraint violations
                            pen = 0.0
                            for i in range(N):
                                pen += max(0, r_[i] - x[i]) + max(0, r_[i] - (w - x[i]))
                                pen += max(0, r_[i] - y[i]) + max(0, r_[i] - (h - y[i]))
                            for i in range(N):
                                for j in range(i+1, N):
                                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                                    pen += max(0, r_[i] + r_[j] - d)
                            return -np.sum(r_) + 1000.0 * pen

                        x0 = np.column_stack([xs, ys, r]).flatten()
                        result = minimize(objective_polish, x0, method='Nelder-Mead',
                                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
                        x2, y2, r2 = unpack(result.x)
                        if validate(x2, y2, r2, w, h):
                            score2 = np.sum(r2)
                            if score2 > best_score:
                                best_score = score2
                                best_result = np.column_stack([x2, y2, r2])
                    except Exception:
                        pass

    # Fallback
    if best_result is None:
        w = h = 1.0
        n_cols = 5
        n_rows = 5
        cells = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(cells) < N:
                    cx = (j + 0.5) * w / n_cols
                    cy = (i + 0.5) * h / n_rows
                    r = 0.45 * min(w / n_cols, h / n_rows)
                    cells.append((cx, cy, r))
        return np.array(cells)

    return best_result
# EVOLVE-BLOCK-END
