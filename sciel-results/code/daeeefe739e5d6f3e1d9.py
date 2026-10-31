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

    # Focused aspect ratio search around square and slight deviations
    # The square (1.0) is often optimal, but slight deviations can help fit 21 circles better.
    aspect_ratios = np.array([0.8, 0.9, 1.0, 1.1, 1.25, 1.5])
    
    # Number of restarts per aspect ratio
    # We have ~115s. 6 ARs * 5 restarts = 30 optimizations.
    # Each SLSQP run takes ~1-3s. 30 * 3 = 90s. Fits within budget.
    max_restarts_per_ar = 5

    def unpack(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return x, y, r

    def validate(x, y, r, w, h):
        # Check if circles are inside rectangle and non-overlapping
        if np.any(r < 0):
            return False
        if np.any(x < r - 1e-9) or np.any(x > w - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > h - r + 1e-9):
            return False
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    def objective(vars_):
        x, y, r = unpack(vars_)
        return -np.sum(r)

    def make_constraints(w, h):
        def cons_inside(vars_):
            x, y, r = unpack(vars_)
            return np.concatenate([
                x - r,
                w - x - r,
                y - r,
                h - y - r
            ])

        def cons_non_overlap(vars_):
            x, y, r = unpack(vars_)
            cons = []
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    cons.append(dist - r[i] - r[j])
            return np.array(cons)

        def cons_positive_r(vars_):
            x, y, r = unpack(vars_)
            return r

        return [
            {'type': 'ineq', 'fun': cons_inside},
            {'type': 'ineq', 'fun': cons_non_overlap},
            {'type': 'ineq', 'fun': cons_positive_r}
        ]

    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        constraints = make_constraints(w, h)

        # Try multiple restarts
        for restart in range(max_restarts_per_ar):
            if time.time() > deadline - 5:
                break

            # Generate hexagonal initial layout
            # Estimate grid size
            # For N=21, 5 cols x 4 rows = 20, plus 1.
            # Let's try to fit N circles in a hexagonal grid.
            n_cols = int(np.sqrt(N * w / h)) + 1
            n_rows = int(np.ceil(N / n_cols))
            
            # Adjust to ensure we have enough space
            while n_cols * n_rows < N:
                if n_cols <= n_rows:
                    n_cols += 1
                else:
                    n_rows += 1

            xs = np.zeros(N)
            ys = np.zeros(N)
            
            # Hexagonal packing coordinates
            # Horizontal spacing: 2*r
            # Vertical spacing: sqrt(3)*r
            # We don't know r yet, so we estimate based on grid density.
            # Assume r ~ min(w/n_cols, h/n_rows) / 1.1
            est_r = min(w / n_cols, h / n_rows) * 0.45
            
            for i in range(N):
                col = i % n_cols
                row = i // n_cols
                
                # Center of cell
                x_center = (col + 0.5) * w / n_cols
                y_center = (row + 0.5) * h / n_rows
                
                # Stagger odd rows
                if row % 2 == 1:
                    x_center += 0.5 * w / n_cols
                    # Wrap around or clamp?
                    # If it goes out of bounds, we might need to adjust the grid.
                    # For simplicity, let's just shift and let the optimizer fix it,
                    # or clamp to stay inside.
                    if x_center + est_r > w:
                        x_center = w - est_r
                    elif x_center - est_r < 0:
                        x_center = est_r

                xs[i] = x_center
                ys[i] = y_center

            # Add random jitter to break symmetry
            if restart > 0:
                # Jitter amount should be small, relative to cell size
                jitter_scale = 0.1 * min(w / n_cols, h / n_rows)
                xs += np.random.normal(0, jitter_scale, N)
                ys += np.random.normal(0, jitter_scale, N)
                
                # Ensure initial points are inside rectangle (roughly)
                # We'll let the optimizer handle fine tuning, but start inside
                # to help SLSQP.
                # Clip to [r0, w-r0] etc? We don't have r0 yet.
                # Let's just clip to [0.01, w-0.01]
                xs = np.clip(xs, 0.01, w - 0.01)
                ys = np.clip(ys, 0.01, h - 0.01)

            # Initial radii
            # Calculate min distance to neighbors and walls
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, xs[i], w - xs[i], ys[i], h - ys[i])
            
            r0 = 0.45 * min_dist
            radii = np.full(N, r0)

            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = xs[i]
                x0[3*i+1] = ys[i]
                x0[3*i+2] = radii[i]

            try:
                # Use SLSQP
                # maxiter: allow enough iterations to converge
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )

                # Check if result is valid and better
                # Even if result.success is False, it might be a good candidate
                x, y, r = unpack(result.x)
                
                if validate(x, y, r, w, h):
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x, y, r])

            except Exception:
                continue

    # Fallback: simple grid in a square of perimeter 4 (w=h=1)
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)

    return best_result
# EVOLVE-BLOCK-END
