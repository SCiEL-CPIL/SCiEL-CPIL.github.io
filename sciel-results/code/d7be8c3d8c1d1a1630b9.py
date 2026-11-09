"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - t0

    def doubled_areas_np(points):
        """Return dict: doubled_area -> count (positive only) using NumPy."""
        pts = np.array(points, dtype=np.int32)
        n = len(pts)
        # Compute all C(n,3) determinants
        # For points i,j,k: det = xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj)
        # = xi*yj + xj*yk + xk*yi - xi*yk - xj*yi - xk*yj
        # Vectorize over all triples
        i_idx, j_idx, k_idx = np.indices((n, n, n))
        # Only keep i < j < k
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_f = i_idx[mask]
        j_f = j_idx[mask]
        k_f = k_idx[mask]
        
        xi = pts[i_f, 0].astype(np.int64)
        yi = pts[i_f, 1].astype(np.int64)
        xj = pts[j_f, 0].astype(np.int64)
        yj = pts[j_f, 1].astype(np.int64)
        xk = pts[k_f, 0].astype(np.int64)
        yk = pts[k_f, 1].astype(np.int64)
        
        d = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        d = d[d > 0]
        if len(d) == 0:
            return {}
        vals, counts = np.unique(d, return_counts=True)
        return dict(zip(vals.tolist(), counts.tolist()))

    def score_of(points):
        if len(points) != 30 or len(set(points)) != 30:
            return -1
        for (x, y) in points:
            if not (-30 <= x <= 30 and -30 <= y <= 30):
                return -1
        cnt = doubled_areas_np(points)
        if not cnt:
            return 0
        return max(cnt.values())

    best_points = None
    best_score = -1

    def try_points(pts):
        nonlocal best_points, best_score
        if len(pts) != 30 or len(set(pts)) != 30:
            return
        for (x, y) in pts:
            if not (-30 <= x <= 30 and -30 <= y <= 30):
                return
        s = score_of(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]

    # Family A: two horizontal lines y=0 and y=h
    for h in [1, 2, 3, 5, 7, 10]:
        for m in range(5, 26):
            n = 30 - m
            if n < 1:
                continue
            if elapsed() > DEADLINE:
                break
            line = [(i, 0) for i in range(m)]
            off = [(i, h) for i in range(n)]
            pts = line + off
            try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family B: three parallel lines
    splits = [(10, 10, 10), (12, 9, 9), (11, 10, 9), (14, 8, 8), (16, 7, 7),
              (15, 8, 7), (9, 10, 11), (8, 11, 11), (13, 9, 8), (15, 10, 5),
              (10, 12, 8), (11, 11, 8), (12, 12, 6), (13, 10, 7), (14, 9, 7)]
    y_configs = [(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 1, 4), (0, 2, 3), (1, 2, 3)]
    for split in splits:
        if sum(split) != 30:
            continue
        if elapsed() > DEADLINE:
            break
        for ys in y_configs:
            pts = []
            for count, y in zip(split, ys):
                pts += [(i, y) for i in range(count)]
            if len(pts) != 30:
                continue
            try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family C: four parallel lines
    splits4 = [(8, 8, 7, 7), (10, 8, 7, 5), (9, 9, 7, 5), (10, 10, 5, 5),
               (12, 8, 6, 4), (10, 8, 8, 4), (9, 8, 7, 6), (10, 9, 6, 5),
               (8, 8, 8, 6), (9, 9, 8, 4), (11, 8, 6, 5), (10, 10, 6, 4)]
    y_configs4 = [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4), (0, 1, 2, 5),
                  (0, 2, 3, 4), (0, 1, 4, 5), (1, 2, 3, 4)]
    for split in splits4:
        if sum(split) != 30:
            continue
        if elapsed() > DEADLINE:
            break
        for ys in y_configs4:
            pts = []
            for count, y in zip(split, ys):
                pts += [(i, y) for i in range(count)]
            if len(pts) != 30:
                continue
            try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family D: two lines with staggered x-offsets
    for h in [1, 2, 3]:
        for m in range(8, 23):
            n = 30 - m
            if n < 1:
                continue
            if elapsed() > DEADLINE:
                break
            for x_off in range(-5, 6):
                line = [(i, 0) for i in range(m)]
                off = [(i + x_off, h) for i in range(n)]
                if any(x < -30 or x > 30 for x, y in off):
                    continue
                pts = line + off
                try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family E: Grid-like configurations
    for rows in range(2, 9):
        if elapsed() > DEADLINE:
            break
        for cols in range(2, 15):
            total = rows * cols
            if total > 30:
                continue
            base_pts = [(x, y) for y in range(rows) for x in range(cols)]
            remaining = 30 - total
            if remaining < 0:
                continue
            for x_start in range(-3, 6):
                for y_extra in [rows, rows + 1]:
                    extra = [(x_start + i, y_extra) for i in range(remaining)]
                    if any(x < -30 or x > 30 for x, y in extra):
                        continue
                    pts = base_pts + extra
                    try_points(pts)
            if elapsed() > DEADLINE:
                break

    # Family F: Optimized 2-line configs
    for m in range(10, 21):
        n = 30 - m
        if elapsed() > DEADLINE:
            break
        for off in range(-4, 5):
            pts2 = [(i, 0) for i in range(m)] + [(i + off, 1) for i in range(n)]
            if all(-30 <= x <= 30 for x, y in pts2):
                try_points(pts2)
        if elapsed() > DEADLINE:
            break

    # Family G: 3 lines with optimized distribution
    for m0 in range(6, 18):
        for m1 in range(6, 18):
            m2 = 30 - m0 - m1
            if m2 < 1:
                continue
            if elapsed() > DEADLINE:
                break
            pts = [(i, 0) for i in range(m0)] + [(i, 1) for i in range(m1)] + [(i, 2) for i in range(m2)]
            try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family H: Concentric circle-like / regular polygon approximations
    # Points on a circle of radius r, spread evenly
    import math
    for r in range(5, 31, 3):
        if elapsed() > DEADLINE:
            break
        for start_angle in range(0, 360, 30):
            pts = []
            for i in range(30):
                angle = math.radians(start_angle + i * 360.0 / 30)
                x = round(r * math.cos(angle))
                y = round(r * math.sin(angle))
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
                else:
                    pts.append((0, 0))  # placeholder, will be filtered
            # Deduplicate
            seen = set()
            unique_pts = []
            for p in pts:
                if p not in seen:
                    seen.add(p)
                    unique_pts.append(p)
            if len(unique_pts) == 30:
                try_points(unique_pts)
        if elapsed() > DEADLINE:
            break

    # Family I: Hexagonal / triangular lattice patches
    for r in range(3, 12):
        if elapsed() > DEADLINE:
            break
        # Generate hexagonal lattice points within radius r
        pts = set()
        for q in range(-r, r + 1):
            for rr in range(-r, r + 1):
                s = -q - rr
                if abs(s) <= r:
                    # Convert axial to cartesian
                    x = q + rr / 2
                    y = rr * math.sqrt(3) / 2
                    pts.add((round(x * 2), round(y * 2)))  # scale to integers
        pts_list = list(pts)
        if len(pts_list) >= 30:
            # Take first 30 or a subset
            import random
            random.seed(42)
            random.shuffle(pts_list)
            for i in range(0, min(10, len(pts_list) - 29)):
                subset = pts_list[i:i+30]
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in subset):
                    try_points(subset)
        if elapsed() > DEADLINE:
            break

    # Local search with enhanced move sets
    def local_search(points, max_iters=500):
        """Try small perturbations to improve score with larger move sets."""
        best_pts = list(points)
        best_s = score_of(best_pts)
        if best_s < 0:
            return best_pts, best_s
        
        # Precompute for speed
        n = 30
        for it in range(max_iters):
            if elapsed() > DEADLINE - 3:
                break
            # Cycle through points with some randomness
            idx = it % n
            ox, oy = best_pts[idx]
            improved = False
            
            # Try moves in increasing radius
            for radius in [1, 2, 3, 4, 5]:
                if improved:
                    break
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        if abs(dx) + abs(dy) > radius:
                            continue
                        nx, ny = ox + dx, oy + dy
                        if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                            continue
                        new_pts = best_pts[:]
                        new_pts[idx] = (nx, ny)
                        if len(set(new_pts)) != 30:
                            continue
                        s = score_of(new_pts)
                        if s > best_s:
                            best_pts = new_pts
                            best_s = s
                            improved = True
                            break
                    if improved:
                        break
            
            # If no improvement in a full cycle, try random larger jumps
            if not improved and it % 30 == 29:
                for _ in range(5):
                    dx = np.random.randint(-8, 9)
                    dy = np.random.randint(-8, 9)
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    new_pts = best_pts[:]
                    new_pts[idx] = (nx, ny)
                    if len(set(new_pts)) != 30:
                        continue
                    s = score_of(new_pts)
                    if s > best_s:
                        best_pts = new_pts
                        best_s = s
                        improved = True
                        break
        return best_pts, best_s

    # Apply local search to the best candidate
    if best_points is not None and elapsed() < DEADLINE - 5:
        refined_pts, refined_s = local_search(best_points, max_iters=500)
        if refined_s > best_score:
            best_score = refined_s
            best_points = refined_pts

    # Also try local search on a few other good candidates
    if elapsed() < DEADLINE - 15:
        # Try several diverse candidates
        candidates = []
        candidates.append([(i, 0) for i in range(15)] + [(i, 1) for i in range(15)])
        candidates.append([(i, 0) for i in range(10)] + [(i, 1) for i in range(10)] + [(i, 2) for i in range(10)])
        candidates.append([(i, 0) for i in range(12)] + [(i, 1) for i in range(9)] + [(i, 2) for i in range(9)])
        candidates.append([(i, 0) for i in range(8)] + [(i, 1) for i in range(8)] + [(i, 2) for i in range(7)] + [(i, 3) for i in range(7)])
        
        for cand in candidates:
            if elapsed() > DEADLINE - 10:
                break
            s_cand = score_of(cand)
            if s_cand >= best_score - 100:
                refined_pts, refined_s = local_search(cand, max_iters=300)
                if refined_s > best_score:
                    best_score = refined_s
                    best_points = refined_pts

    # Final polish: try to improve by swapping pairs of points
    if best_points is not None and elapsed() < DEADLINE - 5:
        def pair_swap_search(points, max_iters=200):
            best_pts = list(points)
            best_s = score_of(best_pts)
            for it in range(max_iters):
                if elapsed() > DEADLINE - 3:
                    break
                i = it % 30
                j = (it * 7 + 1) % 30
                if i == j:
                    continue
                # Try swapping with a random other point's neighborhood
                ox, oy = best_pts[i]
                px, py = best_pts[j]
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = ox + dx, oy + dy
                        if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                            continue
                        new_pts = best_pts[:]
                        new_pts[i] = (nx, ny)
                        if len(set(new_pts)) != 30:
                            continue
                        s = score_of(new_pts)
                        if s > best_s:
                            best_pts = new_pts
                            best_s = s
                            break
                    else:
                        continue
                    break
            return best_pts, best_s
        
        refined_pts, refined_s = pair_swap_search(best_points, max_iters=300)
        if refined_s > best_score:
            best_score = refined_s
            best_points = refined_pts

    if best_points is None:
        return [(x, y) for x in range(6) for y in range(5)]

    return best_points
# EVOLVE-BLOCK-END
