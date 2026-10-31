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

    # Aspect ratios to try.
    # w + h = 2. ar = h/w.
    # Optimal aspect ratios for circle packing are often near 1.0.
    # We will use a mix of square-ish and slightly rectangular.
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.75, 1.4, 0.7]
    
    max_restarts_per_ar = 3 # Reduced to allow more time for relaxation/polish

    def unpack(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return x, y, r

    def validate(x, y, r, w, h):
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

    # Simple force-directed relaxation to improve initial positions
    def relax_positions(xs, ys, radii, w, h, iterations=50):
        """
        Applies repulsive forces to overlapping circles to separate them.
        This helps escape local minima of the initial grid.
        """
        xs = xs.copy()
        ys = ys.copy()
        
        for _ in range(iterations):
            fx = np.zeros(N)
            fy = np.zeros(N)
            overlap_exists = False
            
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist and dist > 1e-6:
                        overlap_exists = True
                        # Force magnitude proportional to overlap
                        force = (min_dist - dist)
                        # Normalize direction
                        fx[i] += force * (dx / dist)
                        fy[i] += force * (dy / dist)
                        fx[j] -= force * (dx / dist)
                        fy[j] -= force * (dy / dist)
                    elif dist <= 1e-6:
                        # If exactly on top, push randomly
                        overlap_exists = True
                        fx[i] += np.random.normal(0, 0.01)
                        fy[i] += np.random.normal(0, 0.01)
                        fx[j] -= np.random.normal(0, 0.01)
                        fy[j] -= np.random.normal(0, 0.01)
            
            if not overlap_exists:
                break
            
            # Update positions
            step_size = 0.5
            xs += step_size * fx
            ys += step_size * fy
            
            # Keep inside rectangle (simple clipping, forces will handle fine-tuning)
            xs = np.clip(xs, radii, w - radii)
            ys = np.clip(ys, radii, h - radii)

        return xs, ys

    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w

        constraints = make_constraints(w, h)

        for restart in range(max_restarts_per_ar):
            if time.time() > deadline - 5:
                break

            # Generate hexagonal initial layout
            n_cols = int(np.sqrt(N * w / h)) + 1
            n_rows = int(np.ceil(N / n_cols))
            
            while n_cols * n_rows < N:
                if n_cols <= n_rows:
                    n_cols += 1
                else:
                    n_rows += 1

            xs = np.zeros(N)
            ys = np.zeros(N)
            
            est_r = min(w / n_cols, h / n_rows) * 0.45
            
            for i in range(N):
                col = i % n_cols
                row = i // n_cols
                
                x_center = (col + 0.5) * w / n_cols
                y_center = (row + 0.5) * h / n_rows
                
                if row % 2 == 1:
                    x_center += 0.5 * w / n_cols
                    if x_center + est_r > w:
                        x_center = w - est_r
                    elif x_center - est_r < 0:
                        x_center = est_r

                xs[i] = x_center
                ys[i] = y_center

            # Add random jitter
            if restart > 0:
                jitter_scale = 0.1 * min(w / n_cols, h / n_rows)
                xs += np.random.normal(0, jitter_scale, N)
                ys += np.random.normal(0, jitter_scale, N)
                xs = np.clip(xs, 0.01, w - 0.01)
                ys = np.clip(ys, 0.01, h - 0.01)

            # Initial radii
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, xs[i], w - xs[i], ys[i], h - ys[i])
            
            r0 = 0.45 * min_dist
            radii = np.full(N, r0)

            # --- NEW: Relaxation Step ---
            # Use relaxation to separate circles that might be overlapping due to jitter or grid constraints
            # This provides a much better starting point for SLSQP
            if time.time() < deadline - 10: # Only relax if we have time
                xs, ys = relax_positions(xs, ys, radii, w, h, iterations=30)
            
            # Re-calculate min_dist after relaxation to adjust r0 if needed?
            # No, keep r0 fixed for the optimization start, SLSQP will adjust.
            # But ensure r0 is not too large.
            
            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = xs[i]
                x0[3*i+1] = ys[i]
                x0[3*i+2] = radii[i]

            try:
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-8}
                )

                x, y, r = unpack(result.x)
                
                if validate(x, y, r, w, h):
                    score = np.sum(r)
                    if score > best_score:
                        # --- NEW: Polish Step ---
                        # If we found a good solution, try to polish it
                        if time.time() < deadline - 5:
                            try:
                                result_polish = minimize(
                                    objective, result.x, method='SLSQP',
                                    constraints=constraints,
                                    options={'maxiter': 100, 'ftol': 1e-10}
                                )
                                xp, yp, rp = unpack(result_polish.x)
                                if validate(xp, yp, rp, w, h):
                                    score_polish = np.sum(rp)
                                    if score_polish > score:
                                        x, y, r = xp, yp, rp
                                        score = score_polish
                            except:
                                pass
                        
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
