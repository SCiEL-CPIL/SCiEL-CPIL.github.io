"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 21
DEADLINE = 140.0  # safety margin before 150s kill


def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter at most 4.
    Maximize sum of radii.
    Returns array of shape (21, 3): [x, y, r] for each circle.
    """
    t_start = time.time()

    def remaining_time():
        return DEADLINE - (time.time() - t_start)

    # Helper: compute the bounding box and perimeter
    def compute_perimeter(xs, ys, rs):
        x_min = np.minimum(xs - rs).min()
        x_max = np.maximum(xs + rs).max()
        y_min = np.minimum(ys - rs).min()
        y_max = np.maximum(ys + rs).max()
        w = x_max - x_min
        h = y_max - y_min
        return 2 * (w + h)

    # Helper: check if circles are valid (no overlaps, within some bounds)
    def is_valid(xs, ys, rs, tol=1e-6):
        if any(r < -tol for r in rs):
            return False
        # Check pairwise distances
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        # Check perimeter
        P = compute_perimeter(xs, ys, rs)
        if P > 4.0 + tol:
            return False
        return True

    # Helper: compute score (sum of radii), only if valid
    def score_if_valid(xs, ys, rs):
        if is_valid(xs, ys, rs):
            return np.sum(rs)
        return -1.0

    # Generate multiple initial configurations with different aspect ratios
    def make_initial_config(w, h, seed=42):
        """Place N circles in a grid within a w x h box, centered at origin."""
        rng = np.random.RandomState(seed)
        # Grid layout
        n_cols = max(1, int(np.ceil(np.sqrt(N * w / h))))
        n_rows = max(1, int(np.ceil(N / n_cols)))
        actual_w = w / n_cols
        actual_h = h / n_rows
        r = min(actual_w, actual_h) * 0.49  # slightly less than half to avoid overlap

        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.full(N, r)
        count = 0
        for i in range(n_cols):
            for j in range(n_rows):
                if count >= N:
                    break
                xs[count] = -w / 2 + (i + 0.5) * actual_w
                ys[count] = -h / 2 + (j + 0.5) * actual_h
                count += 1
        return xs, ys, rs

    # Relaxation algorithm: iterative overlap resolution and boundary pushing
    def relax(xs, ys, rs, max_iter=5000):
        """
        Iteratively resolve overlaps and push circles toward boundaries.
        Returns improved xs, ys, rs.
        """
        xs = xs.copy()
        ys = ys.copy()
        rs = rs.copy()

        for iteration in range(max_iter):
            if remaining_time() < 10.0:
                break

            # Compute current bounding box
            x_min = np.minimum(xs - rs).min()
            x_max = np.maximum(xs + rs).max()
            y_min = np.minimum(ys - rs).max() if N == 0 else np.minimum(ys - rs).min()
            y_max = np.maximum(ys + rs).max()
            w = x_max - x_min
            h = y_max - y_min
            P = 2 * (w + h)

            # If perimeter is too large, shrink radii proportionally
            if P > 4.0:
                scale = 4.0 / P
                rs *= scale ** 0.5  # gentle shrink
                # Adjust positions to maintain relative layout
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                xs = center_x + (xs - center_x) * scale ** 0.5
                ys = center_y + (ys - center_y) * scale ** 0.5

            # Resolve overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist and dist > 1e-12:
                        # Push apart along the line connecting centers
                        overlap = min_dist - dist
                        # Distribute push proportional to radii
                        total_r = rs[i] + rs[j]
                        push_i = overlap * rs[i] / total_r
                        push_j = overlap * rs[j] / total_r
                        ux = dx / dist
                        uy = dy / dist
                        xs[i] -= ux * push_i
                        ys[i] -= uy * push_i
                        xs[j] += ux * push_j
                        ys[j] += uy * push_j
                    elif dist <= 1e-12:
                        # Perfectly overlapping, push randomly
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = min_dist / 2
                        xs[i] += np.cos(angle) * push
                        ys[i] += np.sin(angle) * push
                        xs[j] -= np.cos(angle) * push
                        ys[j] -= np.sin(angle) * push

            # Push circles toward boundaries to fill space
            # For each circle, if it's not touching a boundary, nudge it toward the nearest one
            for i in range(N):
                # Left boundary
                if xs[i] - rs[i] > x_min + 1e-4:
                    xs[i] -= 0.01
                # Right boundary
                if xs[i] + rs[i] < x_max - 1e-4:
                    xs[i] += 0.01
                # Bottom boundary
                if ys[i] - rs[i] > y_min + 1e-4:
                    ys[i] -= 0.01
                # Top boundary
                if ys[i] + rs[i] < y_max - 1e-4:
                    ys[i] += 0.01

            # Try to grow radii slightly if there's room
            # Check if we can uniformly scale up
            P_check = 2 * (w + h)
            if P_check < 3.95:
                rs *= 1.001

        return xs, ys, rs

    # Main algorithm: try multiple aspect ratios and pick the best
    best_score = -1.0
    best_result = None

    # Try various aspect ratios
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4, 2.5, 0.33, 3.0]

    for aspect in aspect_ratios:
        if remaining_time() < 15.0:
            break

        # w * h * 2 = 4, so w * h = 2
        # aspect = w / h, so w = aspect * h
        # aspect * h * h = 2 => h = sqrt(2 / aspect)
        h = np.sqrt(2.0 / aspect)
        w = aspect * h

        # Generate initial configuration
        xs, ys, rs = make_initial_config(w, h, seed=int(aspect * 100))

        # Relax
        xs, ys, rs = relax(xs, ys, rs, max_iter=2000)

        # Evaluate
        s = score_if_valid(xs, ys, rs)
        if s > best_score:
            best_score = s
            best_result = np.column_stack((xs, ys, rs))

        # Also try a perturbed version
        if remaining_time() > 20.0:
            rng = np.random.RandomState(123)
            xs_p = xs + rng.normal(0, 0.05, N)
            ys_p = ys + rng.normal(0, 0.05, N)
            rs_p = rs * (1 + rng.normal(0, 0.02, N))
            rs_p = np.maximum(rs_p, 0.01)
            xs_p, ys_p, rs_p = relax(xs_p, ys_p, rs_p, max_iter=1000)
            s_p = score_if_valid(xs_p, ys_p, rs_p)
            if s_p > best_score:
                best_score = s_p
                best_result = np.column_stack((xs_p, ys_p, rs_p))

    # Fallback to seed-like grid if nothing worked
    if best_result is None or best_score <= 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final verification
    if not is_valid(best_result[:, 0], best_result[:, 1], best_result[:, 2]):
        # Try to fix by scaling down
        xs, ys, rs = best_result[:, 0], best_result[:, 1], best_result[:, 2]
        for _ in range(100):
            P = compute_perimeter(xs, ys, rs)
            if P <= 4.0:
                break
            scale = 4.0 / P
            rs *= scale
            xs = (xs - xs.mean()) * scale + xs.mean()
            ys = (ys - ys.mean()) * scale + ys.mean()
        best_result = np.column_stack((xs, ys, rs))
        if not is_valid(xs, ys, rs):
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
            return np.array(cells[:N])

    return best_result


if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(np.sum(res[:, 2]))

