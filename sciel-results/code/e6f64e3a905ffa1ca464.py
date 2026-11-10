"""Optimized 8-point configuration maximizing product of distances at fixed diameter."""
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    from math import cos, sin, pi, sqrt, log

    # Start from regular octagon
    # Regular octagon vertices on unit circle
    def regular_octagon():
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

    def compute_score(points):
        """Compute Delta / n^n where Delta is product over ordered pairs of distances,
        after scaling to diameter 2."""
        n = len(points)
        # Compute all pairwise distances
        d = [[0.0] * n for _ in range(n)]
        max_dist = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dist = sqrt(dx * dx + dy * dy)
                d[i][j] = dist
                d[j][i] = dist
                if dist > max_dist:
                    max_dist = dist

        if max_dist < 1e-15:
            return 0.0

        # Scale so diameter = 2
        scale = 2.0 / max_dist
        # Product over ordered pairs (i != j) of distances
        # log(Delta) = sum_{i != j} log(d[i][j] * scale)
        # = sum_{i != j} log(d[i][j]) + n*(n-1)*log(scale)
        log_delta = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    if d[i][j] < 1e-15:
                        return 0.0
                    log_delta += math.log(d[i][j])
        log_delta += n * (n - 1) * math.log(scale)

        # Score = Delta / n^n = exp(log_delta - n*log(n))
        log_score = log_delta - n * math.log(n)
        # Avoid overflow/underflow
        if log_score > 700:
            return float('inf')
        if log_score < -700:
            return 0.0
        return math.exp(log_score)

    def compute_score_fast(points):
        """Faster version using numpy."""
        import numpy as np
        pts = np.array(points, dtype=np.float64)
        n = len(pts)
        # Compute distance matrix
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        max_dist = np.max(dists)
        if max_dist < 1e-15:
            return 0.0
        scale = 2.0 / max_dist
        # Product over ordered pairs i != j
        mask = ~np.eye(n, dtype=bool)
        log_delta = np.sum(np.log(dists[mask] * scale))
        log_score = log_delta - n * np.log(n)
        if log_score > 700:
            return float('inf')
        if log_score < -700:
            return 0.0
        return float(np.exp(log_score))

    # Try several candidate configurations
    candidates = []

    # 1. Regular octagon
    reg = regular_octagon()
    candidates.append((reg, compute_score_fast(reg)))

    # 2. Slightly perturbed regular octagon - try various patterns
    # Pattern: alternate radii
    for r_ratio in [0.95, 0.98, 1.02, 1.05, 0.90, 1.10]:
        pts = []
        for k in range(8):
            r = 1.0 if k % 2 == 0 else r_ratio
            theta = 2 * pi * k / 8
            pts.append([r * cos(theta), r * sin(theta)])
        candidates.append((pts, compute_score_fast(pts)))

    # 3. Two concentric squares (rotated)
    for r_inner in [0.3, 0.5, 0.7, 0.8, 0.9]:
        for rot in [0, pi/8, pi/4]:
            pts = []
            # Outer square
            for k in range(4):
                theta = 2 * pi * k / 4 + rot
                pts.append([cos(theta), sin(theta)])
            # Inner square
            for k in range(4):
                theta = 2 * pi * k / 4 + rot + pi/4
                pts.append([r_inner * cos(theta), r_inner * sin(theta)])
            candidates.append((pts, compute_score_fast(pts)))

    # 4. Points on two circles with different angular offsets
    for r1 in [0.8, 0.9, 1.0]:
        for r2 in [0.5, 0.7, 0.9]:
            for off in [0, pi/8, pi/4, 3*pi/8]:
                pts = []
                for k in range(4):
                    theta = 2 * pi * k / 4
                    pts.append([r1 * cos(theta), r1 * sin(theta)])
                for k in range(4):
                    theta = 2 * pi * k / 4 + off
                    pts.append([r2 * cos(theta), r2 * sin(theta)])
                candidates.append((pts, compute_score_fast(pts)))

    # 5. Regular heptagon + center
    pts = []
    for k in range(7):
        theta = 2 * pi * k / 7
        pts.append([cos(theta), sin(theta)])
    pts.append([0.0, 0.0])
    candidates.append((pts, compute_score_fast(pts)))

    # 6. Square with 4 more points
    for r_extra in [0.5, 0.7, 0.9, 1.0]:
        for rot_extra in [0, pi/4, pi/8]:
            pts = []
            # Square
            for k in range(4):
                theta = 2 * pi * k / 4
                pts.append([cos(theta), sin(theta)])
            # 4 extra points
            for k in range(4):
                theta = 2 * pi * k / 4 + rot_extra
                pts.append([r_extra * cos(theta), r_extra * sin(theta)])
            candidates.append((pts, compute_score_fast(pts)))

    # 7. Random perturbations of regular octagon
    import random
    random.seed(42)
    for trial in range(50):
        pts = []
        for k in range(8):
            r = 1.0 + random.uniform(-0.1, 0.1)
            theta = 2 * pi * k / 8 + random.uniform(-0.05, 0.05)
            pts.append([r * cos(theta), r * sin(theta)])
        candidates.append((pts, compute_score_fast(pts)))

    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Best candidate so far
    best_pts = candidates[0][0]
    best_score = candidates[0][1]

    # Now do local optimization on the best candidate
    # Use gradient-free optimization (Nelder-Mead style)
    def optimize(points, max_iters=500):
        """Simple local optimization using coordinate descent + perturbations."""
        current = [p[:] for p in points]
        current_score = compute_score_fast(current)
        best = [p[:] for p in current]
        best_sc = current_score

        step = 0.05
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(8):
                for coord in range(2):
                    # Try small perturbations
                    for delta in [-step, step]:
                        trial = [p[:] for p in current]
                        trial[i][coord] += delta
                        sc = compute_score_fast(trial)
                        if sc > current_score + 1e-12:
                            current = trial
                            current_score = sc
                            improved = True
                            if sc > best_sc:
                                best = [p[:] for p in current]
                                best_sc = sc
                            break
            if not improved:
                step *= 0.8
                if step < 1e-6:
                    break

        return best, best_sc

    # Optimize top candidates
    for idx in range(min(5, len(candidates))):
        if time.time() > deadline - 10:
            break
        pts, sc = candidates[idx]
        opt_pts, opt_sc = optimize(pts, max_iters=200)
        if opt_sc > best_score:
            best_pts = opt_pts
            best_score = opt_sc

    # Final check
    if time.time() > deadline - 2:
        return best_pts

    # Try a few more random restarts with optimization if time permits
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        pts = []
        for k in range(8):
            r = random.uniform(0.5, 1.5)
            theta = random.uniform(0, 2 * pi)
            pts.append([r * cos(theta), r * sin(theta)])
        # Quick check
        sc = compute_score_fast(pts)
        if sc > 0.9:  # Only optimize promising starts
            opt_pts, opt_sc = optimize(pts, max_iters=100)
            if opt_sc > best_score:
                best_pts = opt_pts
                best_score = opt_sc

    return best_pts
