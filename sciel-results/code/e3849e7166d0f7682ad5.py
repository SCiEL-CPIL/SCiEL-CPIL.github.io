"""Optimized lattice-based search for maximum area multiplicity."""
import time
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 80.0  # Safety margin for 90s limit

    # Precompute triple indices for n=30
    triple_indices = np.array([(i, j, k) for i in range(30) for j in range(i+1, 30) for k in range(j+1, 30)], dtype=np.int64)
    I, J, K = triple_indices[:, 0], triple_indices[:, 1], triple_indices[:, 2]
    N_TRIPLES = len(I)

    def evaluate_fast(points):
        """Vectorized evaluation of max multiplicity of positive doubled areas."""
        pts = np.array(points, dtype=np.int64)
        x = pts[:, 0]
        y = pts[:, 1]
        det = (x[J] - x[I]) * (y[K] - y[I]) - (y[J] - y[I]) * (x[K] - x[I])
        det = np.abs(det)
        det = det[det > 0]
        if len(det) == 0:
            return 0
        vals, cnts = np.unique(det, return_counts=True)
        return int(cnts.max())

    def valid(points):
        if len(points) != 30:
            return False
        if len(set(points)) != 30:
            return False
        for px, py in points:
            if not (-30 <= px <= 30 and -30 <= py <= 30):
                return False
        return True

    def make_lattice(a, b, u1, u2, v1, v2, ox=0, oy=0):
        """Create a x b lattice with basis vectors (u1,u2) and (v1,v2)."""
        pts = []
        for i in range(a):
            for j in range(b):
                x = ox + i * u1 + j * v1
                y = oy + i * u2 + j * v2
                pts.append((x, y))
        return pts

    def make_grid(a, b, ox=0, oy=0, scale=1):
        """Create a x b grid of points scaled by 'scale' and offset by (ox, oy)."""
        return make_lattice(a, b, scale, 0, 0, scale, ox=ox, oy=oy)

    candidates = []

    # Systematic lattice search over all small basis vectors
    # Basis vectors from -6 to 6
    vecs = [(x, y) for x in range(-6, 7) for y in range(-6, 7) if not (x == 0 and y == 0)]
    
    # Filter to reasonable basis pairs
    basis_pairs = []
    for i, (u1, u2) in enumerate(vecs):
        for j, (v1, v2) in enumerate(vecs[i+1:], i+1):
            d = u1 * v2 - u2 * v1
            if d != 0 and abs(d) <= 12:  # Limit determinant for feasibility
                basis_pairs.append(((u1, u2), (v1, v2), d))
    
    # Try various grid shapes and basis pairs
    grid_shapes = [(5, 6), (6, 5), (4, 8), (8, 4), (3, 10), (10, 3), (2, 15), (15, 2)]
    
    for a, b in grid_shapes:
        for (u1, u2), (v1, v2), d in basis_pairs:
            # Estimate bounding box
            xs = [i * u1 + j * v1 for i in range(a) for j in range(b)]
            ys = [i * u2 + j * v2 for i in range(a) for j in range(b)]
            if not xs or not ys:
                continue
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            width = max_x - min_x
            height = max_y - min_y
            if width > 60 or height > 60:
                continue
            
            # Try a few offsets to center
            for ox in range(-3, 4):
                for oy in range(-3, 4):
                    pts = make_lattice(a, b, u1, u2, v1, v2, ox=ox, oy=oy)
                    if valid(pts):
                        candidates.append(pts)
    
    # Also try two-line constructions
    for a in range(10, 21):
        b = 30 - a
        for d in range(1, 9):
            for spacing in [1, 2]:
                pts = []
                for i in range(a):
                    pts.append((i * spacing - (a - 1) * spacing // 2, 0))
                for i in range(b):
                    pts.append((i * spacing - (b - 1) * spacing // 2, d))
                max_x = max(p[0] for p in pts)
                min_x = min(p[0] for p in pts)
                max_y = max(p[1] for p in pts)
                min_y = min(p[1] for p in pts)
                cx = (max_x + min_x) // 2
                cy = (max_y + min_y) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                if valid(pts):
                    candidates.append(pts)
    
    # Three-line constructions
    for a, b, c in [(10,10,10), (9,11,10), (11,9,10), (8,12,10), (12,8,10)]:
        for d1 in [1, 2, 3]:
            for d2 in range(d1+1, d1+5):
                pts = []
                for i in range(a):
                    pts.append((i - a//2, 0))
                for i in range(b):
                    pts.append((i - b//2, d1))
                for i in range(c):
                    pts.append((i - c//2, d2))
                max_x = max(p[0] for p in pts)
                min_x = min(p[0] for p in pts)
                max_y = max(p[1] for p in pts)
                min_y = min(p[1] for p in pts)
                cx = (max_x + min_x) // 2
                cy = (max_y + min_y) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                if valid(pts):
                    candidates.append(pts)

    # Remove duplicates
    seen = set()
    unique_candidates = []
    for pts in candidates:
        key = tuple(sorted(pts))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(pts)
    candidates = unique_candidates

    # Evaluate all candidates
    best_points = None
    best_score = 0

    for pts in candidates:
        if time.time() > deadline - 10:
            break
        score = evaluate_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts

    if best_points is None:
        best_points = make_grid(5, 6, ox=0, oy=0)
        best_score = evaluate_fast(best_points)

    # Local search with targeted moves
    if time.time() < deadline - 10:
        current = [list(p) for p in best_points]
        current_score = best_score

        def local_search(points, start_score, time_limit):
            pts = [list(p) for p in points]
            score = start_score
            directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1),
                          (2,0), (-2,0), (0,2), (0,-2), (2,1), (-2,1), (2,-1), (-2,-1),
                          (3,0), (-3,0), (0,3), (0,-3)]

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                for idx in range(30):
                    if time.time() > time_limit:
                        break
                    ox, oy = pts[idx]
                    for dx, dy in directions:
                        nx, ny = ox + dx, oy + dy
                        if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                            continue
                        new_pts = [tuple(p) for p in pts]
                        new_pts[idx] = (nx, ny)
                        if len(set(new_pts)) != 30:
                            continue
                        s = evaluate_fast(new_pts)
                        if s > score:
                            pts[idx] = [nx, ny]
                            score = s
                            improved = True
                            break
                    if improved:
                        break
            return pts, score

        current, current_score = local_search(best_points, best_score, deadline - 5)
        if current_score > best_score:
            best_points = [tuple(p) for p in current]
            best_score = current_score

    return best_points
# EVOLVE-BLOCK-END
