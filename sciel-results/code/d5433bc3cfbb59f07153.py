import time
import numpy as np
from collections import Counter

def construct():
    start = time.time()
    deadline = start + 85.0

    def evaluate_fast(points):
        """
        Vectorized evaluation of max multiplicity of positive doubled areas.
        points: list of 30 (x, y) tuples.
        """
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        if n < 3:
            return 0
            
        counts = Counter()
        for i in range(n):
            pi = pts[i]
            rest = pts[i+1:]
            n_rest = len(rest)
            if n_rest < 2:
                continue
            
            v = rest - pi
            v_x = v[:, 0]
            v_y = v[:, 1]
            
            # Cross product of all pairs in 'rest' relative to pi
            # cross[j, k] = |v_x[j]*v_y[k] - v_y[j]*v_x[k]|
            cross = np.abs(v_x[:, None] * v_y[None, :] - v_y[:, None] * v_x[None, :])
            
            # We only care about pairs (j, k) with j < k to avoid double counting and self-pairs
            # The matrix is symmetric. Diagonal is 0.
            mask = np.triu(np.ones((n_rest, n_rest), dtype=bool), k=1)
            cross_vals = cross[mask]
            
            # Filter out zero areas
            cross_vals = cross_vals[cross_vals > 0]
            
            if len(cross_vals) > 0:
                vals, cnts = np.unique(cross_vals, return_counts=True)
                for val, cnt in zip(vals, cnts):
                    counts[int(val)] += cnt
        
        if not counts:
            return 0
        return max(counts.values())

    def valid(points):
        if len(points) != 30:
            return False
        if len(set(points)) != 30:
            return False
        for px, py in points:
            if not (-30 <= px <= 30 and -30 <= py <= 30):
                return False
        return True

    def make_grid(a, b, ox=0, oy=0, scale=1):
        pts = []
        for i in range(a):
            for j in range(b):
                x = ox + i * scale
                y = oy + j * scale
                pts.append((x, y))
        return pts

    def make_lattice(a, b, u1, u2, v1, v2, ox=0, oy=0):
        pts = []
        for i in range(a):
            for j in range(b):
                x = ox + i * u1 + j * v1
                y = oy + i * u2 + j * v2
                pts.append((x, y))
        return pts

    # Generate Candidates
    candidates = []

    # 1. Standard Grids
    for a, b in [(5, 6), (6, 5), (3, 10), (10, 3), (2, 15), (15, 2), (4, 8), (8, 4), (1, 30), (30, 1)]:
        for scale in [1, 2]:
            for ox in range(-15, 16, 5):
                for oy in range(-15, 16, 5):
                    pts = make_grid(a, b, ox=ox, oy=oy, scale=scale)
                    if valid(pts):
                        candidates.append(pts)

    # 2. Sheared Lattices
    for a, b in [(5, 6), (6, 5), (4, 8), (8, 4)]:
        for s in range(-3, 4):
            for scale in [1, 2]:
                for ox in range(-15, 16, 5):
                    for oy in range(-15, 16, 5):
                        pts = make_lattice(a, b, scale, 0, s*scale, scale, ox=ox, oy=oy)
                        if valid(pts):
                            candidates.append(pts)

    # 3. Random Points
    import random
    random.seed(42)
    for _ in range(50):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.add((x, y))
        pts = list(pts)
        candidates.append(pts)

    # Deduplicate
    seen = set()
    unique_candidates = []
    for pts in candidates:
        key = tuple(sorted(pts))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(pts)
    candidates = unique_candidates

    # Evaluate Initial Candidates
    best_points = None
    best_score = 0
    scored_candidates = []
    
    for pts in candidates:
        if time.time() > deadline - 5:
            break
        score = evaluate_fast(pts)
        scored_candidates.append((score, pts))
        if score > best_score:
            best_score = score
            best_points = pts

    # Sort by score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Local Search on Top Candidates
    # We take the top 5 distinct configurations
    top_seeds = []
    seen_keys = set()
    for score, pts in scored_candidates:
        key = tuple(sorted(pts))
        if key not in seen_keys:
            seen_keys.add(key)
            top_seeds.append((score, pts))
            if len(top_seeds) >= 5:
                break

    def local_search(points, start_score, time_limit):
        pts = [list(p) for p in points]
        score = start_score
        directions = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if dx != 0 or dy != 0]
        # Sort directions by magnitude to try small steps first
        directions.sort(key=lambda d: d[0]**2 + d[1]**2)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Shuffle points to avoid bias
            random.shuffle(pts)
            for idx in range(30):
                if time.time() > time_limit:
                    break
                ox, oy = pts[idx]
                # Try directions
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

    # Run local search on each seed with a time slice
    if top_seeds:
        time_per_seed = (deadline - 5 - time.time()) / len(top_seeds)
        for score, pts in top_seeds:
            if time.time() > deadline - 2:
                break
            limit = time.time() + time_per_seed
            current, current_score = local_search(pts, score, limit)
            if current_score > best_score:
                best_score = current_score
                best_points = [tuple(p) for p in current]

    if best_points is None:
        best_points = make_grid(5, 6)
        
    return best_points
