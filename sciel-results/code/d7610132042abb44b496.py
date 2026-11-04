import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    best_val = 1e10
    best_x = None

    def generate_grid_init(w_init, strategy=0):
        """Generate initial grid layout with different strategies"""
        h_init = 2.0 - w_init
        side = int(np.ceil(np.sqrt(N)))
        xs = []
        ys = []
        rs = []

        for i in range(side):
            for j in range(side):
                if len(xs) >= N:
                    break
                if strategy == 0:
                    # Standard grid
                    cx = (i + 0.5) / side * w_init
                    cy = (j + 0.5) / side * h_init
                    r = min(cx, w_init - cx, cy, h_init - cy) * 0.9
                elif strategy == 1:
                    # Offset grid for better packing
                    offset_x = 0.5 / side * (i % 2) * w_init
                    offset_y = 0.5 / side * (j % 2) * h_init
                    cx = (i + 0.5) / side * w_init + offset_x * 0.3
                    cy = (j + 0.5) / side * h_init + offset_y * 0.3
                    cx = np.clip(cx, 0.1, w_init - 0.1)
                    cy = np.clip(cy, 0.1, h_init - 0.1)
                    r = min(cx, w_init - cx, cy, h_init - cy) * 0.85
                else:
                    # Randomized grid
                    cx = (i + np.random.uniform(0.2, 0.8)) / side * w_init
                    cy = (j + np.random.uniform(0.2, 0.8)) / side * h_init
                    cx = np.clip(cx, 0.05, w_init - 0.05)
                    cy = np.clip(cy, 0.05, h_init - 0.05)
                    r = min(cx, w_init - cx, cy, h_init - cy) * 0.8

                xs.append(cx)
                ys.append(cy)
                rs.append(r)

        while len(xs) < N:
            xs.append(w_init/2)
            ys.append(h_init/2)
            rs.append(0.05)

        return np.array(xs[:N]), np.array(ys[:N]), np.array(rs[:N])

    def objective(v):
        w = v[0]
        if w <= 0 or w >= 2:
            return 1e10
        h = 2.0 - w

        xs = v[1:N+1]
        ys = v[N+1:2*N+1]
        rs = v[2*N+1:3*N+1]

        # Penalty for negative radii
        if np.any(rs < 0):
            return 1e10

        # Boundary penalties
        pen = 0.0
        pen += np.sum(np.maximum(0, rs - xs)**2)
        pen += np.sum(np.maximum(0, rs - (w - xs))**2)
        pen += np.sum(np.maximum(0, rs - ys)**2)
        pen += np.sum(np.maximum(0, rs - (h - ys))**2)

        # Vectorized overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = rs[i] + rs[j]
                if d < min_d:
                    pen += (min_d - d)**2

        return -np.sum(rs) + 100.0 * pen

    def validate_solution(pos_r, w, h, n_circles, tol=1e-7):
        """
        Check if circles are valid: disjoint interiors, within bounds.
        pos_r: (n, 3) array of [x, y, r]
        """
        if pos_r.shape[0] != n_circles:
            return False

        xs = pos_r[:, 0]
        ys = pos_r[:, 1]
        rs = pos_r[:, 2]

        # Check radii are positive
        if np.any(rs < 1e-9):
            return False

        # Check bounds: circles must be inside [0, w] x [0, h]
        if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
            return False

        # Check non-overlap
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < (rs[i] + rs[j]) - tol:
                    return False
        return True

    def resolve_collisions(circles, w, h, max_iters=2000, tol=1e-8):
        """
        Relaxation algorithm to resolve overlaps and boundary violations.
        circles: (n, 3) array [x, y, r]
        Returns modified circles array.
        """
        n = circles.shape[0]
        pos = circles[:, :2].copy()
        rad = circles[:, 2].copy()

        for _ in range(max_iters):
            moved = False

            # 1. Boundary constraints: push circles inside [0, w] x [0, h]
            for i in range(n):
                x, y, r = pos[i, 0], pos[i, 1], rad[i]
                if x - r < 0:
                    pos[i, 0] = r
                    moved = True
                if x + r > w:
                    pos[i, 0] = w - r
                    moved = True
                if y - r < 0:
                    pos[i, 1] = r
                    moved = True
                if y + r > h:
                    pos[i, 1] = h - r
                    moved = True

            # 2. Pairwise collisions: push apart
            for i in range(n):
                for j in range(i+1, n):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rad[i] + rad[j]

                    if dist < min_dist:
                        if dist < 1e-9:
                            # Random perturbation if exactly overlapping
                            angle = np.random.uniform(0, 2*np.pi)
                            push = (min_dist - dist) * 0.5
                            pos[i, 0] -= np.cos(angle) * push
                            pos[i, 1] -= np.sin(angle) * push
                            pos[j, 0] += np.cos(angle) * push
                            pos[j, 1] += np.sin(angle) * push
                            moved = True
                        else:
                            overlap = min_dist - dist
                            nx = dx / dist
                            ny = dy / dist
                            # Push apart by half the overlap each
                            pos[i, 0] -= nx * overlap * 0.5
                            pos[i, 1] -= ny * overlap * 0.5
                            pos[j, 0] += nx * overlap * 0.5
                            pos[j, 1] += ny * overlap * 0.5
                            moved = True

            if not moved:
                break

        # Final clamp to ensure boundary validity
        for i in range(n):
            pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
            pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])

        return np.column_stack((pos, rad))

    def slsqp_optimize(circles, w, h, deadline):
        """
        Optimize radii and positions using SLSQP with hard constraints.
        This is the primary refinement engine, adapted from Inspiration 1.
        """
        if time.time() > deadline - 5.0:
            return None

        n = len(circles)
        x0 = circles.flatten()

        # Objective: maximize sum of radii => minimize negative sum
        def objective_sl(x):
            if time.time() > deadline:
                return 0.0
            r = x[2::3]
            return -np.sum(r)

        # Constraints
        constraints = []

        # Boundary constraints
        for i in range(n):
            # x >= r  => x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
            # x + r <= w => w - x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
            # y >= r  => y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
            # y + r <= h => h - y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
            # r > 0 => r >= 1e-6
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})

        # Overlap constraints: dist >= r_i + r_j
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
                })

        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((1e-6, w)) # x
            bounds.append((1e-6, h)) # y
            bounds.append((1e-6, min(w, h)/2)) # r

        try:
            res = minimize(
                objective_sl,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-9}
            )

            if res.success:
                new_circles = res.x.reshape(-1, 3)
                # Validate and fix any tiny numerical issues
                if validate_solution(new_circles, w, h, N):
                    return new_circles
                else:
                    # Try to fix with relaxation
                    fixed = resolve_collisions(new_circles, w, h, max_iters=100)
                    if validate_solution(fixed, w, h, N):
                        return fixed
        except Exception:
            pass

        return None

    def polish_solution(x_sol):
        """Final high-precision polish pass"""
        w_best = x_sol[0]
        h_best = 2.0 - w_best
        xs_best = x_sol[1:N+1].copy()
        ys_best = x_sol[N+1:2*N+1].copy()
        rs_best = x_sol[2*N+1:3*N+1].copy()

        # Multiple passes to resolve collisions and ensure validity
        for _ in range(50):
            changed = False

            # Ensure within bounds
            for i in range(N):
                max_r = min(xs_best[i], w_best - xs_best[i], ys_best[i], h_best - ys_best[i])
                if rs_best[i] > max_r:
                    rs_best[i] = max_r
                    changed = True

            # Resolve overlaps iteratively
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs_best[i] - xs_best[j]
                    dy = ys_best[i] - ys_best[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = rs_best[i] + rs_best[j]
                    if d < min_d - 1e-12:
                        # Shrink both radii proportionally
                        if d > 1e-10:
                            factor = d / (min_d * 1.001)
                            rs_best[i] *= factor
                            rs_best[j] *= factor
                        else:
                            # Move apart if very close
                            if dx == 0 and dy == 0:
                                xs_best[i] += 0.01
                                xs_best[j] -= 0.01
                        changed = True

            if not changed:
                break

        # Final validation
        for i in range(N):
            rs_best[i] = min(rs_best[i], xs_best[i], w_best - xs_best[i], ys_best[i], h_best - ys_best[i])
            if rs_best[i] < 0:
                rs_best[i] = 0

        return np.column_stack((xs_best, ys_best, rs_best))

    # Try diverse aspect ratios with multiple initialization strategies
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4]

    for w_init in aspect_ratios:
        if time.time() > deadline:
            break

        for strategy in range(3):
            if time.time() > deadline:
                break

            xs, ys, rs = generate_grid_init(w_init, strategy)
            x0_curr = np.concatenate(([w_init], xs, ys, rs))

            # Use L-BFGS-B with tighter tolerances
            try:
                res = minimize(objective, x0_curr, method='L-BFGS-B', 
                             options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})

                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()

                    # Second pass with tighter tolerance from improved starting point
                    if time.time() < deadline - 15:
                        res2 = minimize(objective, best_x, method='L-BFGS-B', 
                                     options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
                        if res2.fun < best_val:
                            best_val = res2.fun
                            best_x = res2.x.copy()
            except:
                pass

    if best_x is None:
        # Fallback to simple grid
        w_init = 1.0
        xs, ys, rs = generate_grid_init(w_init, 0)
        best_x = np.concatenate(([w_init], xs, ys, rs))

    # Extract current best solution
    w_best = best_x[0]
    h_best = 2.0 - w_best
    circles_best = np.column_stack((best_x[1:N+1], best_x[N+1:2*N+1], best_x[2*N+1:3*N+1]))

    # Apply collision resolution to ensure validity
    circles_resolved = resolve_collisions(circles_best, w_best, h_best, max_iters=500)

    # Try SLSQP refinement on the resolved solution
    if time.time() < deadline - 20:
        circles_slsqp = slsqp_optimize(circles_resolved, w_best, h_best, deadline)
        if circles_slsqp is not None:
            # Compare sum of radii
            if np.sum(circles_slsqp[:, 2]) > np.sum(circles_resolved[:, 2]):
                circles_best = circles_slsqp
            else:
                circles_best = circles_resolved
        else:
            circles_best = circles_resolved

    # Final polish
    result = polish_solution(np.concatenate(([w_best], circles_best[:, 0], circles_best[:, 1], circles_best[:, 2])))

    # Additional optimization pass on polished solution
    if time.time() < deadline - 10:
        x_polished = np.concatenate(([w_best], result[:, 0], result[:, 1], result[:, 2]))
        try:
            res_final = minimize(objective, x_polished, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_x = res_final.x.copy()
                result = polish_solution(best_x)
        except:
            pass

    return result
