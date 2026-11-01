"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    best_sum = 0.0
    best_result = None

    # We'll use a grid-based initialization with varying aspect ratios
    # and then refine with scipy SLSQP or a custom optimizer.

    from scipy.optimize import minimize

    def pack_with_aspect(w, h, positions, radii):
        """
        Given aspect ratio (w, h with 2(w+h)=4 => w+h=2),
        positions (N,2) in [0,w]x[0,h], radii (N,),
        check validity and return sum of radii.
        """
        # Check circles are inside rectangle
        for i in range(N):
            if positions[i, 0] < radii[i] - 1e-10:
                return -1.0
            if positions[i, 0] > w - radii[i] + 1e-10:
                return -1.0
            if positions[i, 1] < radii[i] - 1e-10:
                return -1.0
            if positions[i, 1] > h - radii[i] + 1e-10:
                return -1.0

        # Check disjoint interiors
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - 1e-10:
                    return -1.0

        return np.sum(radii)

    def objective(x, w, h):
        """
        x: [x0, y0, r0, x1, y1, r1, ..., x_{N-1}, y_{N-1}, r_{N-1}]
        Minimize negative sum of radii.
        """
        positions = x[:2 * N].reshape(N, 2)
        radii = x[2 * N:]

        # Bounds: circles must be inside rectangle
        for i in range(N):
            if positions[i, 0] < radii[i]:
                return 1e10
            if positions[i, 0] > w - radii[i]:
                return 1e10
            if positions[i, 1] < radii[i]:
                return 1e10
            if positions[i, 1] > h - radii[i]:
                return 1e10

        # Disjoint interiors constraint
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j]:
                    return 1e10

        return -np.sum(radii)

    def constraints_func(w, h):
        """Generate constraint functions for SLSQP."""
        cons = []
        # Bounds are handled by bounds parameter, but we add explicit ones
        for i in range(N):
            # x_i >= r_i
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, w=w, h=h: x[2 * i] - x[2 * N + i]
            })
            # x_i + r_i <= w
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, w=w, h=h: w - x[2 * i] - x[2 * N + i]
            })
            # y_i >= r_i
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, w=w, h=h: x[2 * i + 1] - x[2 * N + i]
            })
            # y_i + r_i <= h
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, w=w, h=h: h - x[2 * i + 1] - x[2 * N + i]
            })

        # Disjoint interiors: dist_ij >= r_i + r_j
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sqrt(
                        (x[2 * i] - x[2 * j]) ** 2 +
                        (x[2 * i + 1] - x[2 * j + 1]) ** 2
                    ) - x[2 * N + i] - x[2 * N + j]
                })

        return cons

    def grid_init(w, h, rows, cols):
        """Initialize circles in a grid pattern within w x h rectangle."""
        positions = np.zeros((N, 2))
        radii = np.zeros(N)

        # Place N circles in a grid of rows x cols
        # Fill row by row
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= N:
                    break
                x = (c + 0.5) * w / cols
                y = (r + 0.5) * h / rows
                positions[idx, 0] = x
                positions[idx, 1] = y
                # Initial radius: half the minimum cell dimension, slightly reduced
                cell_w = w / cols
                cell_h = h / rows
                r_init = min(cell_w, cell_h) / 2 - 0.01
                radii[idx] = max(r_init, 0.01)
                idx += 1

        return positions, radii

    def run_slsqp(w, h, positions, radii):
        """Run SLSQP optimization for given aspect ratio."""
        x0 = np.concatenate([positions.flatten(), radii])

        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, w))  # x_i
            bounds.append((0, h))  # y_i
        for i in range(N):
            bounds.append((0.01, min(w, h) / 2))  # r_i

        cons = constraints_func(w, h)

        try:
            result = minimize(
                lambda x: objective(x, w, h),
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-8}
            )

            if result.success and result.fun < 0:
                positions_opt = result.x[:2 * N].reshape(N, 2)
                radii_opt = result.x[2 * N:]
                total = -result.fun
                if total > best_sum:
                    # Validate
                    score = pack_with_aspect(w, h, positions_opt, radii_opt)
                    if score > 0:
                        return positions_opt, radii_opt, score
        except Exception:
            pass

        return None, None, 0.0

    # Try various aspect ratios and grid configurations
    aspect_ratios = [
        (1.0, 1.0),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.5, 0.5),
        (0.5, 1.5),
        (1.3, 0.7),
        (0.7, 1.3),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.1, 0.9),
        (0.9, 1.1),
        (1.6, 0.4),
        (0.4, 1.6),
    ]

    grid_configs = [
        (3, 7),
        (4, 5),
        (5, 4),
        (6, 3),
        (3, 3),  # will only use 9
        (4, 6),
        (6, 4),
        (7, 3),
    ]

    for w, h in aspect_ratios:
        if time.time() > deadline:
            break

        for rows, cols in grid_configs:
            if time.time() > deadline:
                break

            if rows * cols < N:
                continue

            positions, radii = grid_init(w, h, rows, cols)
            positions_opt, radii_opt, score = run_slsqp(w, h, positions, radii)

            if radii_opt is not None and score > best_sum:
                best_sum = score
                best_result = np.column_stack([positions_opt, radii_opt])

    # If we haven't found anything good, return a simple grid solution
    if best_result is None:
        w, h = 1.0, 1.0
        positions, radii = grid_init(w, h, 5, 4)
        best_result = np.column_stack([positions, radii])

    return best_result
# EVOLVE-BLOCK-END
