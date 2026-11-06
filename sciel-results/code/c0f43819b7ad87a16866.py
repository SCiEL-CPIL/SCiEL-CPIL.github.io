"""Number-theoretic construction: Grids of indices mapped to lattice points."""
import time
import numpy as np
from collections import Counter
import itertools

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin (evaluator kills at 90s)

    def evaluate_fast(points):
        """
        Vectorized evaluation of max multiplicity of positive doubled areas.
        points: list of 30 (x, y) tuples.
        """
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        
        counts = Counter()
        for i in range(n):
            pi = pts[i]
            rest = pts[i+1:]
            n_rest = len(rest)
            if n_rest < 2:
                continue
            
            for j in range(n_rest):
                pj = rest[j]
                v1 = pj - pi
                if j + 1 >= n_rest:
                    continue
                v2s = rest[j+1:] - pi
                
                dets = np.abs(v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0])
                dets = dets[dets > 0]
                if len(dets) > 0:
                    vals, cnts = np.unique(dets, return_counts=True)
                    for v, c in zip(vals, cnts):
                        counts[int(v)] += c
        
        if not counts:
            return 0
        return max(counts.values())

    def make_grid(a, b, ox=0, oy=0, scale=1):
        """Create a x b grid of points scaled by 'scale' and offset by (ox, oy)."""
        pts = []
        for i in range(a):
            for j in range(b):
                x = ox + i * scale
                y = oy + j * scale
                pts.append((x, y))
        return pts

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

    # Candidate Configurations
    candidates = []

    # Standard axis-aligned grids with various scales
    # Reduced offsets to save time
    for a, b in [(5,6), (6,5), (3,10), (10,3), (2,15), (15,2), (4,8), (8,4)]:
        for scale in [1, 2, 3]:
            for ox in range(-10, 11, 5):
                for oy in range(-10, 11, 5):
                    pts = make_grid(a, b, ox=ox, oy=oy, scale=scale)
                    if valid(pts):
                        candidates.append(pts)

    # Sheared lattices: basis (1,0) and (s,1)
    for a, b in [(5,6), (6,5), (4,8), (8,4), (3,10)]:
        for s in range(-2, 3):
            for scale in [1, 2]:
                for ox in range(-10, 11, 5):
                    for oy in range(-10, 11, 5):
                        pts = make_lattice(a, b, scale, 0, s*scale, scale, ox=ox, oy=oy)
                        if valid(pts):
                            candidates.append(pts)

    # Rotated lattices: try various integer basis vectors
    # Reduced basis vectors
    basis_vectors = [
        (1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (2, -1), (1, 2), (1, -2),
        (3, 1), (3, -1), (1, 3), (1, -3)
    ]
    for a, b in [(5,6), (6,5), (4,8), (8,4)]:
        for i, (u1, u2) in enumerate(basis_vectors):
            for j, (v1, v2) in enumerate(basis_vectors[i+1:], i+1):
                det_val = u1 * v2 - u2 * v1
                if abs(det_val) != 1:
                    continue
                for ox in range(-10, 11, 5):
                    for oy in range(-10, 11, 5):
                        pts = make_lattice(a, b, u1, u2, v1, v2, ox=ox, oy=oy)
                        if valid(pts):
                            candidates.append(pts)

    # Two-line construction
    for a, b in [(15,15), (14,16), (16,14), (13,17), (17,13)]:
        for d in [1, 2, 3, 4]:
            for shift in range(0, 5):
                pts = []
                for i in range(a):
                    pts.append((i, 0))
                for i in range(b):
                    pts.append((i + shift, d))
                max_x = max(p[0] for p in pts)
                min_x = min(p[0] for p in pts)
                max_y = max(p[1] for p in pts)
                min_y = min(p[1] for p in pts)
                cx = (max_x + min_x) // 2
                cy = (max_y + min_y) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                if valid(pts):
                    candidates.append(pts)

    # Three-line construction
    for a, b, c in [(10,10,10), (9,11,10), (11,9,10)]:
        for d1 in [1, 2]:
            for d2 in [d1+1, d1+2]:
                for shift in range(0, 3):
                    pts = []
                    for i in range(a):
                        pts.append((i, 0))
                    for i in range(b):
                        pts.append((i + shift, d1))
                    for i in range(c):
                        pts.append((i + 2*shift, d2))
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

    # Sort by heuristic
    def grid_heuristic(pts):
        if len(pts) != 30:
            return 0
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        w = max(xs) - min(xs) if xs else 0
        h = max(ys) - min(ys) if ys else 0
        if w == 0 or h == 0:
            return 0
        ratio = w / h if h > 0 else float('inf')
        return -abs(1 - ratio)

    candidates.sort(key=grid_heuristic, reverse=True)

    best_points = None
    best_score = 0

    # Evaluate candidates with strict time limit
    eval_deadline = time.time() + 10.0
    for pts in candidates:
        if time.time() > eval_deadline:
            break
        if not valid(pts):
            continue
        score = evaluate_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts

    if best_points is None:
        best_points = make_grid(5, 6, ox=0, oy=0)
        best_score = evaluate_fast(best_points)

    # Local search
    if time.time() < deadline - 20:
        def local_search(points, start_score, time_limit, step_sizes=[1]):
            pts = [list(p) for p in points]
            score = start_score
            directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
            
            for step in step_sizes:
                if time.time() > time_limit:
                    break
                improved = True
                while improved and time.time() < time_limit:
                    improved = False
                    for idx in range(30):
                        if time.time() > time_limit:
                            break
                        ox, oy = pts[idx]
                        for dx, dy in directions:
                            nx, ny = ox + dx * step, oy + dy * step
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
        
        current, current_score = local_search(best_points, best_score, deadline - 15, step_sizes=[1])
        best_points = [tuple(p) for p in current]
        best_score = current_score

    return best_points
# EVOLVE-BLOCK-END
