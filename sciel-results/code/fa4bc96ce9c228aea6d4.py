"""Number-theoretic construction: Grids of indices mapped to lattice points."""
import time
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 50.0 # Safety margin

    def evaluate_fast(points):
        """
        Vectorized evaluation of max multiplicity of positive doubled areas.
        points: list of 30 (x, y) tuples.
        """
        pts = np.array(points, dtype=np.int64)
        n = len(pts)
        
        # Generate all combinations of indices (i, j, k) with i < j < k
        # Using itertools is faster for setup, but numpy is faster for computation
        # For n=30, 4060 triples is small.
        
        # We can do this without itertools to avoid overhead, or just use it.
        # Let's use a simple loop to build arrays, then vectorize the det calculation.
        # Or even better:
        # i, j, k ranges
        # Since n is small, we can just compute all pairwise differences? No, that's O(N^2) memory, fine.
        
        # Method:
        # 1. Create indices for all triples
        # 2. Compute vectors
        # 3. Compute dets
        # 4. Count
        
        # Efficient way:
        # For each i, consider pairs (j, k) with j, k > i.
        # This avoids generating redundant permutations.
        
        counts = Counter()
        for i in range(n):
            pi = pts[i]
            # All points j > i
            rest = pts[i+1:]
            n_rest = len(rest)
            if n_rest < 2:
                continue
            
            # Compute vectors from pi to all points in rest
            # v_j = rest[j] - pi
            # We need det(v_j, v_k) for all j < k in rest
            # This is equivalent to det(rest[j] - pi, rest[k] - pi)
            
            # Let's compute the cross products for all pairs in rest relative to pi
            # This is O(N^2) per i, so O(N^3) total.
            # For N=30, 30 * (29*28/2) ~ 12000 ops. Very fast.
            
            # Vectorized for the inner loop:
            # We can compute all cross products of rest[j] with rest[k]?
            # No, they must share the same base point pi.
            
            # Let's just loop j, and vectorize k.
            for j in range(n_rest):
                pj = rest[j]
                v1 = pj - pi
                # v2 for all k > j
                # rest[j+1:]
                if j + 1 >= n_rest:
                    continue
                v2s = rest[j+1:] - pi
                
                # Cross product: v1[0]*v2[1] - v1[1]*v2[0]
                dets = np.abs(v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0])
                
                # Filter positive
                dets = dets[dets > 0]
                if len(dets) > 0:
                    # Update counter
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

    # Candidate Configurations
    # We want to maximize the frequency of a specific area.
    # Regular grids tend to have high frequency of small areas (like 1, 2, etc. in index space).
    # We will test standard grids, sheared grids, and rotated lattices.
    
    candidates = []
    
    def make_lattice(a, b, u1, u2, v1, v2, ox=0, oy=0):
        """Create a x b lattice with basis vectors (u1,u2) and (v1,v2)."""
        pts = []
        for i in range(a):
            for j in range(b):
                x = ox + i * u1 + j * v1
                y = oy + i * u2 + j * v2
                pts.append((x, y))
        return pts
    
    # Standard axis-aligned grids
    for a, b in [(5,6), (6,5), (3,10), (10,3), (2,15), (15,2), (4,8), (8,4), (3,10), (10,3)]:
        for ox in [-12, -10, -8, 0]:
            for oy in [-12, -10, -8, 0]:
                candidates.append(make_grid(a, b, ox=ox, oy=oy))
    
    # Sheared lattices: basis (1,0) and (s,1) for various shear values
    for a, b in [(5,6), (6,5), (4,8), (8,4)]:
        for s in [-2, -1, 0, 1, 2]:
            for ox in [-12, -8, 0]:
                for oy in [-12, -8, 0]:
                    pts = make_lattice(a, b, 1, 0, s, 1, ox=ox, oy=oy)
                    if valid(pts):
                        candidates.append(pts)
    
    # Rotated lattices: try various integer basis vectors
    # Basis vectors must be primitive (gcd of components = 1)
    basis_vectors = [
        (1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (2, -1), (1, 2), (1, -2),
        (3, 1), (3, -1), (1, 3), (1, -3), (2, 3), (2, -3), (3, 2), (3, -2)
    ]
    for a, b in [(5,6), (6,5), (4,8), (8,4), (3,10), (10,3)]:
        for i, (u1, u2) in enumerate(basis_vectors):
            for j, (v1, v2) in enumerate(basis_vectors[i+1:], i+1):
                # Check if basis is unimodular (|det| = 1) for primitive lattice
                det_val = u1 * v2 - u2 * v1
                if abs(det_val) != 1:
                    continue
                for ox in [-15, -10, -5, 0]:
                    for oy in [-15, -10, -5, 0]:
                        pts = make_lattice(a, b, u1, u2, v1, v2, ox=ox, oy=oy)
                        if valid(pts):
                            candidates.append(pts)
    
    # Two-line construction: a points on one line, b points on parallel line
    # Points: (i, 0) for i in range(a) and (i, d) for i in range(b)
    # Area of triangle with two points on line 1 and one on line 2: |d * (x2 - x1)| / 2
    # Doubled area = |d * (x2 - x1)|
    for a, b in [(15,15), (14,16), (16,14), (13,17), (17,13)]:
        for d in [1, 2, 3, 4, 5]:
            pts = []
            for i in range(a):
                pts.append((i, 0))
            for i in range(b):
                pts.append((i, d))
            # Center it
            max_x = max(p[0] for p in pts)
            min_x = min(p[0] for p in pts)
            max_y = max(p[1] for p in pts)
            min_y = min(p[1] for p in pts)
            cx = (max_x + min_x) // 2
            cy = (max_y + min_y) // 2
            pts = [(x - cx, y - cy) for x, y in pts]
            if valid(pts):
                candidates.append(pts)
    
    # Three-line construction: points on three parallel lines
    for a, b, c in [(10,10,10), (9,11,10), (11,9,10), (8,12,10), (12,8,10)]:
        for d1 in [1, 2, 3]:
            for d2 in [d1+1, d1+2, d1+3]:
                pts = []
                for i in range(a):
                    pts.append((i, 0))
                for i in range(b):
                    pts.append((i, d1))
                for i in range(c):
                    pts.append((i, d2))
                max_x = max(p[0] for p in pts)
                min_x = min(p[0] for p in pts)
                max_y = max(p[1] for p in pts)
                min_y = min(p[1] for p in pts)
                cx = (max_x + min_x) // 2
                cy = (max_y + min_y) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                if valid(pts):
                    candidates.append(pts)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for pts in candidates:
        key = tuple(sorted(pts))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(pts)
    candidates = unique_candidates
    
    best_points = None
    best_score = 0
    
    # Sort candidates by heuristic: prefer more square-like grids first
    # (more square grids tend to have better area multiplicity)
    def grid_heuristic(pts):
        if len(pts) != 30:
            return 0
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        w = max(xs) - min(xs) if xs else 0
        h = max(ys) - min(ys) if ys else 0
        # Prefer aspect ratio close to 1
        if w == 0 or h == 0:
            return 0
        ratio = w / h if h > 0 else float('inf')
        return -abs(1 - ratio)
    
    candidates.sort(key=grid_heuristic, reverse=True)
    
    for pts in candidates:
        if time.time() > deadline - 5:
            break
        if not valid(pts):
            continue
        score = evaluate_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts
            
    if best_points is None:
        # Fallback
        best_points = make_grid(5, 6, ox=0, oy=0)
        best_score = evaluate_fast(best_points)
        
    # Optional: Local Search
    # Given the timeout issue, we will be very conservative.
    # Only if we have plenty of time.
    
    if time.time() < deadline - 10:
        current = [list(p) for p in best_points]
        current_score = best_score
        
        # Enhanced local search: try multiple step sizes and directions
        # First pass: fine-tune with step 1
        # Second pass: try larger moves if time permits
        
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
        
        # Fine-grained search
        current, current_score = local_search(best_points, best_score, deadline - 5, step_sizes=[1])
        best_points = [tuple(p) for p in current]
        best_score = current_score
        
        # Coarse search if time remains
        if time.time() < deadline - 10:
            current, current_score = local_search(best_points, best_score, deadline - 5, step_sizes=[2])
            if current_score > best_score:
                best_points = [tuple(p) for p in current]
                best_score = current_score
        
        # Very coarse search if time remains
        if time.time() < deadline - 15:
            current, current_score = local_search(best_points, best_score, deadline - 5, step_sizes=[3, 4])
            if current_score > best_score:
                best_points = [tuple(p) for p in current]
                best_score = current_score

    return best_points
# EVOLVE-BLOCK-END
