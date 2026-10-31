"""Rewritten program: simulated annealing with smart init for Heilbronn triangles."""
import time
import numpy as np

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
TRI_AREA = 0.5 * np.sqrt(3) / 2.0  # area of unit equilateral triangle


def _barycentric_inside(p):
    """Check if point p is inside the triangle (with small margin)."""
    # Using barycentric coordinates
    v0 = V2 - V0
    v1 = V1 - V0
    v2 = p - V0
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    den = d00 * d11 - d01 * d01
    if den < 1e-12:
        return False
    u = (d11 * d02 - d01 * d12) / den
    v = (d00 * d12 - d01 * d02) / den
    eps = 1e-9
    return u >= -eps and v >= -eps and (u + v) <= 1.0 + eps


def _clamp_inside(p):
    """Clamp point to be inside the triangle (project onto triangle if outside)."""
    if _barycentric_inside(p):
        return p
    # Project onto triangle: find closest point
    # Simple approach: use barycentric and clamp
    v0 = V2 - V0
    v1 = V1 - V0
    v2 = p - V0
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    den = d00 * d11 - d01 * d01
    if den < 1e-12:
        return V0.copy()
    u = (d11 * d02 - d01 * d12) / den
    v = (d00 * d12 - d01 * d02) / den
    # Clamp barycentric coords to [0, 1] with u+v <= 1
    if u < 0:
        u = 0.0
    if v < 0:
        v = 0.0
    if u + v > 1.0:
        s = u + v
        u /= s
        v /= s
    return V0 + u * v0 + v * v1


def _min_tri_area(pts):
    """Compute the minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if area < min_area:
                    min_area = area
    return min_area


def _score(pts):
    """Score = min triangle area / triangle area."""
    return _min_tri_area(pts) / TRI_AREA


def _smart_init(rng):
    """
    Smart initialization: place points on a hexagonal-like grid pattern
    inside the triangle, then add small random jitter.
    """
    pts = []
    # Start with vertices and midpoints for a base structure
    # Use a hexagonal packing pattern
    # For 11 points, try a pattern based on triangular lattice
    # Layer 0: 1 point at centroid-ish
    # Better: use known good configurations as starting points
    
    # Approach: place points on a grid clipped to triangle
    # Use barycentric grid
    n_rows = 4
    candidates = []
    for i in range(n_rows + 1):
        for j in range(n_rows + 1 - i):
            # Barycentric: (i/n_rows, j/n_rows, (n_rows-i-j)/n_rows)
            u = i / n_rows
            v = j / n_rows
            w = 1.0 - u - v
            if w < -1e-9:
                continue
            w = max(0.0, w)
            p = u * V0 + v * V1 + w * V2
            candidates.append(p)
    
    # We have (n_rows+1)(n_rows+2)/2 = 15 candidates for n_rows=4
    # Pick 11 of them, preferring spread
    # Add small jitter
    rng.shuffle(candidates)
    # Actually, let's use a more deliberate pattern
    
    # Hexagonal pattern: for 11 points in equilateral triangle
    # Use rows: 1, 2, 3, 3, 2 = 11 or 1, 2, 3, 4, 1 = 11
    # Let's try a grid-based approach with jitter
    
    pts = []
    # Place points on a triangular lattice
    # Row 0 (bottom): 4 points
    # Row 1: 3 points
    # Row 2: 2 points
    # Row 3: 1 point
    # Total: 10, need 1 more
    
    # Better: use a pattern that's known to work
    # Place at vertices, edge midpoints, and interior
    pts = [
        V0 * 0.1 + V1 * 0.0 + V2 * 0.0,  # near V0
        V0 * 0.0 + V1 * 0.9 + V2 * 0.0,  # near V1
        V0 * 0.0 + V1 * 0.0 + V2 * 0.9,  # near V2
    ]
    
    # Add edge midpoints
    pts.append((V0 + V1) / 2.0)
    pts.append((V1 + V2) / 2.0)
    pts.append((V2 + V0) / 2.0)
    
    # Add centroid
    pts.append((V0 + V1 + V2) / 3.0)
    
    # Add 4 more interior points
    # Use a smaller triangle pattern
    for frac in [0.3, 0.7]:
        p1 = V0 * frac + V1 * (1 - frac) * 0.5 + V2 * (1 - frac) * 0.5
        pts.append(p1)
    
    # Add 2 more
    pts.append(V0 * 0.5 + V1 * 0.25 + V2 * 0.25)
    pts.append(V0 * 0.25 + V1 * 0.5 + V2 * 0.25)
    
    # Ensure we have exactly 11
    while len(pts) < N:
        # Add random interior point
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            p = u * V0 + v * V1 + (1 - u - v) * V2
            pts.append(p)
    pts = pts[:N]
    
    # Add small jitter
    for i in range(N):
        pts[i] = pts[i] + np.array([rng.gauss(0, 0.02), rng.gauss(0, 0.02)])
        pts[i] = _clamp_inside(pts[i])
    
    return np.array(pts)


def _random_init(rng):
    """Random initialization inside triangle."""
    pts = []
    for _ in range(N):
        while True:
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                p = u * V0 + v * V1 + (1 - u - v) * V2
                pts.append(p)
                break
    return np.array(pts)


def _move_point(pts, idx, rng, step_size):
    """Propose a move for point idx."""
    new_pts = pts.copy()
    new_pts[idx] = new_pts[idx] + np.array([rng.gauss(0, step_size), rng.gauss(0, step_size)])
    new_pts[idx] = _clamp_inside(new_pts[idx])
    return new_pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_pts = None
    best_score = -1.0
    
    # Multiple restarts with different strategies
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        
        if restart == 0:
            pts = _smart_init(np.random.default_rng(42))
        else:
            rng = np.random.default_rng(42 + restart)
            if restart % 3 == 0:
                pts = _smart_init(rng)
            else:
                pts = _random_init(rng)
        
        current_score = _score(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
        
        # Simulated annealing
        T = 0.05
        T_min = 0.0001
        cooling = 0.995
        step_size = 0.05
        
        # Local improvement loop
        while T > T_min and time.time() < deadline - 5.0:
            # Pick a random point to move
            idx = np.random.randint(0, N)
            new_pts = _move_point(pts, idx, np.random, step_size)
            new_score = _score(new_pts)
            
            delta = new_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / T):
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
            
            T *= cooling
            step_size *= 0.999
        
        # If we have time, do a quick local refinement on best
        if time.time() < deadline - 10.0:
            # Fine-tune best_pts with small steps
            fine_pts = best_pts.copy()
            fine_score = best_score
            T_fine = 0.001
            for step in range(200):
                if time.time() > deadline - 5.0:
                    break
                idx = np.random.randint(0, N)
                new_pts = _move_point(fine_pts, idx, np.random, 0.005)
                new_score = _score(new_pts)
                delta = new_score - fine_score
                if delta > 0 or np.random.random() < np.exp(delta / T_fine):
                    fine_pts = new_pts
                    fine_score = new_score
                T_fine *= 0.98
            if fine_score > best_score:
                best_score = fine_score
                best_pts = fine_pts.copy()
    
    # Final check: ensure all points are inside
    for i in range(N):
        best_pts[i] = _clamp_inside(best_pts[i])
    
    return best_pts
