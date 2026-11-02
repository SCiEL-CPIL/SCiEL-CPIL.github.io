"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_score = -1.0
    best_result = None

    # Initial seed: grid packing
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    init = np.array(cells[:N])

    # Simulated annealing over: positions (x,y), radii, aspect ratio
    # Rectangle: width w, height h, w + h = 2 (perimeter = 4)
    # We parameterize by aspect ratio r = w/h, so w = 2*r/(1+r), h = 2/(1+r)

    def evaluate(circles, aspect):
        """Return (sum_radii, is_valid). circles: (N,3) array [x, y, r]."""
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)

        # Check bounds: circles must be inside rectangle
        # x in [r, w-r], y in [r, h-r]
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]

        # If any circle violates bounds, it's invalid
        if np.any(xs < rs - 1e-9) or np.any(xs > w - rs + 1e-9) or \
           np.any(ys < rs - 1e-9) or np.any(ys > h - rs + 1e-9):
            return 0.0, False

        # Check pairwise distances
        n = len(circles)
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < rs[i] + rs[j] - 1e-9:
                    return 0.0, False

        return float(np.sum(rs)), True

    def objective(circles, aspect):
        """Maximize sum of radii subject to constraints."""
        score, valid = evaluate(circles, aspect)
        if not valid:
            return 0.0
        return score

    # State: positions, radii, aspect
    def make_state():
        # Start from grid layout, map to rectangle
        aspect = 1.0  # square
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        side = int(np.ceil(np.sqrt(N)))
        circles = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            circles[i, 0] = (col + 0.5) / side * w
            circles[i, 1] = (row + 0.5) / side * h
            circles[i, 2] = 0.5 / side * min(w, h) * 0.95
        return circles, aspect

    best_circles, best_aspect = make_state()
    best_circles, best_aspect = best_circles.copy(), best_aspect
    cur_score = objective(best_circles, best_aspect)
    if cur_score > best_score:
        best_score = cur_score
        best_result = (best_circles.copy(), best_aspect)

    T = 0.05  # initial temperature
    T_min = 1e-6
    cooling = 0.9995

    random.seed(42)
    np.random.seed(42)

    max_iter = 500000
    for it in range(max_iter):
        if time.time() > deadline:
            break

        if T < T_min:
            break

        # Perturb: move one circle, adjust one radius, or change aspect
        pert_type = random.random()
        cur_circles, cur_aspect = best_circles.copy(), best_aspect

        if pert_type < 0.4:
            # Move a random circle
            idx = random.randint(0, N - 1)
            dx = random.gauss(0, 0.02)
            dy = random.gauss(0, 0.02)
            cur_circles[idx, 0] += dx
            cur_circles[idx, 1] += dy
        elif pert_type < 0.7:
            # Adjust a random radius
            idx = random.randint(0, N - 1)
            dr = random.gauss(0, 0.01)
            cur_circles[idx, 2] = max(0.01, cur_circles[idx, 2] + dr)
        else:
            # Change aspect ratio
            cur_aspect *= (1.0 + random.gauss(0, 0.05))
            cur_aspect = max(0.2, min(5.0, cur_aspect))
            # Rescale positions to fit new rectangle
            old_w = 2.0 * best_aspect / (1.0 + best_aspect)
            old_h = 2.0 / (1.0 + best_aspect)
            new_w = 2.0 * cur_aspect / (1.0 + cur_aspect)
            new_h = 2.0 / (1.0 + cur_aspect)
            cur_circles[:, 0] *= new_w / old_w
            cur_circles[:, 1] *= new_h / old_h

        new_score = objective(cur_circles, cur_aspect)
        delta = new_score - cur_score

        if delta > 0 or random.random() < np.exp(delta / T):
            best_circles = cur_circles
            best_aspect = cur_aspect
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_result = (best_circles.copy(), best_aspect)

        T *= cooling

    # Final polish: try to increase radii greedily
    if best_result is not None:
        final_circles, final_aspect = best_result[0].copy(), best_result[1]
        w = 2.0 * final_aspect / (1.0 + final_aspect)
        h = 2.0 / (1.0 + final_aspect)

        # Try to grow each circle slightly
        for _ in range(100):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                # Try growing circle i
                dr = 0.005
                old_r = final_circles[i, 2]
                final_circles[i, 2] = old_r + dr
                score, valid = evaluate(final_circles, final_aspect)
                if valid and score > best_score:
                    best_score = score
                    best_result = (final_circles.copy(), final_aspect)
                    improved = True
                else:
                    final_circles[i, 2] = old_r
            if not improved:
                break

    if best_result is None:
        # Fallback to initial grid
        return init

    circles, aspect = best_result
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)

    # Clamp radii to ensure validity
    for i in range(N):
        r = min(circles[i, 2], circles[i, 0] - 1e-6, w - circles[i, 0] - 1e-6,
                circles[i, 1] - 1e-6, h - circles[i, 1] - 1e-6)
        r = max(r, 0.01)
        circles[i, 2] = r

    return circles
# EVOLVE-BLOCK-END
