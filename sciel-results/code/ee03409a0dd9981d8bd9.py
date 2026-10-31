import numpy as np
import time
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute all C(11,3) = 165 triangle index triples
_tri_indices = np.array(list(combinations(range(N), 3)))

def _all_areas(pts):
    """Vectorized computation of all 165 triangle areas."""
    pi = pts[_tri_indices[:, 0]]
    pj = pts[_tri_indices[:, 1]]
    pk = pts[_tri_indices[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area(pts):
    """Return the minimum triangle area."""
    return float(np.min(_all_areas(pts)))

def _clamp_to_triangle(pts):
    """Clamp points to be inside the unit equilateral triangle."""
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    # y >= 0
    y = np.maximum(y, 0.0)
    # y <= H * (1 - x) for x in [0, 0.5], or y <= H * x for x in [0.5, 1]
    # Actually the triangle is: x >= 0, y >= 0, y <= H*(1-x) for left side
    # The triangle vertices are (0,0), (1,0), (0.5, H)
    # Edges: bottom y=0, right edge from (1,0) to (0.5,H): y = -H*(x-1) = H*(1-x) for x in [0.5,1]
    #         left edge from (0,0) to (0.5,H): y = H*x for x in [0, 0.5]
    # So: y <= H*x for x <= 0.5, y <= H*(1-x) for x >= 0.5
    # Combined: y <= H * min(x, 1-x) ... no that's not right.
    # Let me think again. The triangle has vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    # The three edges:
    # 1. (0,0) to (1,0): y = 0, so y >= 0
    # 2. (1,0) to (0.5, H): parametric: x = 1 - t*0.5, y = t*H => t = y/H, x = 1 - y/(2H) => y = 2H*(1-x) for x in [0.5, 1]
    #    Wait: 2H = sqrt(3), so y = sqrt(3)*(1-x) for x in [0.5,1]
    # 3. (0,0) to (0.5, H): x = t*0.5, y = t*H => t = 2x, y = 2H*x = sqrt(3)*x for x in [0, 0.5]
    # So constraints: y >= 0, y <= sqrt(3)*x for x in [0,0.5], y <= sqrt(3)*(1-x) for x in [0.5,1]
    # Equivalently: y >= 0, y <= H*2*x, y <= H*2*(1-x)
    # Since H = sqrt(3)/2, 2H = sqrt(3)
    
    # Better approach: use barycentric coordinates
    # For point (x,y), barycentric coords relative to (0,0),(1,0),(0.5,H):
    # c = y/H (weight for vertex 2)
    # b = x - 0.5*c (weight for vertex 1)  
    # a = 1 - b - c (weight for vertex 0)
    # Need a, b, c >= 0
    
    x_clamped = np.zeros_like(x)
    y_clamped = np.zeros_like(y)
    for i in range(len(x)):
        xi, yi = x[i], y[i]
        c = yi / H
        b = xi - 0.5 * c
        a = 1.0 - b - c
        # If any coord is negative, project back
        if a < 0 or b < 0 or c < 0:
            # Find closest point on boundary
            # Try clamping each coordinate to 0 and check feasibility
            best_x, best_y = xi, yi
            best_dist = np.inf
            for (ca, cb, cc) in [(max(a,0), max(b,0), max(c,0)),
                                  (0, b, c), (a, 0, c), (a, b, 0)]:
                if ca + cb + cc > 0:
                    ca_n, cb_n, cc_n = ca/(ca+cb+cc), cb/(ca+cb+cc), cc/(ca+cb+cc)
                    px = cb_n * 1.0 + cc_n * 0.5
                    py = cc_n * H
                    d = (px - xi)**2 + (py - yi)**2
                    if d < best_dist:
                        best_dist = d
                        best_x, best_y = px, py
            x_clamped[i] = best_x
            y_clamped[i] = best_y
        else:
            x_clamped[i] = xi
            y_clamped[i] = yi
    return np.column_stack([x_clamped, y_clamped])

def _maximize_triangle_area_direct(p_i, p_j, p_k, pts):
    """
    Given three points forming the critical triangle, try to increase their area
    by moving them along directions that increase the area, while keeping them in the triangle.
    Uses a direct geometric approach: for each vertex, move it in the direction 
    perpendicular to the opposite edge (outward from the triangle's interior relative to that edge).
    """
    # Area = 0.5 * |cross(PJ-PI, PK-PI)|
    # To maximize area, we want to move points apart.
    # Strategy: for each vertex, move it in the direction away from the centroid of the other two.
    
    pi = pts[_tri_indices[0, 0]]  # placeholder, we'll use actual indices
    # Actually let's work with the actual point indices
    return pts  # placeholder

def _direct_maximize_critical(pts, time_left_fn):
    """
    Core strategy: iteratively find the minimum-area triangle and directly 
    maximize its area by moving its vertices, then repeat.
    """
    best_pts = pts.copy()
    best_score = _min_area(best_pts)
    
    max_rounds = 500
    for round_num in range(max_rounds):
        if time_left_fn() < 5.0:
            break
        
        areas = _all_areas(best_pts)
        min_idx = np.argmin(areas)
        i, j, k = _tri_indices[min_idx]
        
        pi = best_pts[i].copy()
        pj = best_pts[j].copy()
        pk = best_pts[k].copy()
        
        # Centroid of the critical triangle
        centroid = (pi + pj + pk) / 3.0
        
        # For each vertex, compute direction to move it to increase area
        # Moving vertex i away from the line jk increases area
        # The direction that increases area for vertex i is perpendicular to (pk-pj), 
        # pointing away from the triangle interior
        
        # Direction for vertex i: perpendicular to edge jk, pointing outward
        # Edge jk vector
        ejk = pk - pj
        # Perpendicular (rotated 90 degrees)
        perp_ijk = np.array([-ejk[1], ejk[0]])
        # Determine which direction increases area: the one where centroid is on the opposite side
        # Area contribution of vertex i: cross(pj-pi, pk-pi)
        # If we move pi by d, new area ~ cross(pj-pi-d, pk-pi-d)
        # Derivative w.r.t. d: -(pj-pi) x d_hat... let's just try both directions
        cross_val = np.cross(pj - pi, pk - pi)
        if cross_val >= 0:
            # pi should move in direction that increases cross product
            # d(cross(pj-pi, pk-pi))/d(pi) = -(pj-pi) x ... 
            # Simpler: move pi away from centroid of jk
            mid_jk = (pj + pk) / 2.0
            dir_i = pi - mid_jk
            if np.linalg.norm(dir_i) < 1e-10:
                dir_i = perp_ijk
        else:
            mid_jk = (pj + pk) / 2.0
            dir_i = pi - mid_jk
            if np.linalg.norm(dir_i) < 1e-10:
                dir_i = -perp_ijk
        
        # Similarly for j and k
        eik = pk - pi
        perp_ike = np.array([-eik[1], eik[0]])
        mid_ik = (pi + pk) / 2.0
        dir_j = pj - mid_ik
        if np.linalg.norm(dir_j) < 1e-10:
            dir_j = perp_ike
        
        ejk2 = pk - pj
        perp_jk = np.array([-ejk2[1], ejk2[0]])
        mid_pj = (pi + pj) / 2.0
        dir_k = pk - mid_pj
        if np.linalg.norm(dir_k) < 1e-10:
            dir_k = perp_jk
        
        # Normalize directions
        for d in [dir_i, dir_j, dir_k]:
            n = np.linalg.norm(d)
            if n > 1e-10:
                d /= n
        
        # Try multiple step sizes and pick the best
        step_sizes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        best_new_pts = best_pts.copy()
        best_new_score = best_score
        
        for step in step_sizes:
            candidate = best_pts.copy()
            candidate[i] = pi + dir_i * step
            candidate[j] = pj + dir_j * step
            candidate[k] = pk + dir_k * step
            
            # Clamp to triangle
            candidate = _clamp_to_triangle(candidate)
            
            new_score = _min_area(candidate)
            if new_score > best_new_score + 1e-15:
                best_new_score = new_score
                best_new_pts = candidate.copy()
        
        if best_new_score > best_score + 1e-15:
            best_score = best_new_score
            best_pts = best_new_pts
        else:
            # No improvement from this critical triangle, try a small random perturbation
            # of the critical vertices to escape local optima
            rng = np.random.default_rng(round_num)
            improved = False
            for attempt in range(20):
                if time_left_fn() < 5.0:
                    break
                candidate = best_pts.copy()
                noise_i = rng.normal(0, 0.01, 2)
                noise_j = rng.normal(0, 0.01, 2)
                noise_k = rng.normal(0, 0.01, 2)
                candidate[i] = pi + noise_i
                candidate[j] = pj + noise_j
                candidate[k] = pk + noise_k
                candidate = _clamp_to_triangle(candidate)
                new_score = _min_area(candidate)
                if new_score > best_score + 1e-15:
                    best_score = new_score
                    best_pts = candidate.copy()
                    improved = True
                    break
            
            if not improved:
                # Stuck, break out
                break
    
    return best_pts, best_score

def _init_structured():
    """Structured initialization: vertices, midpoints, centroid, symmetric inner points."""
    pts = []
    # Vertices
    pts.append([0.0, 0.0])
    pts.append([1.0, 0.0])
    pts.append([0.5, H])
    # Edge midpoints
    pts.append([0.5, 0.0])
    pts.append([0.75, H/2])
    pts.append([0.25, H/2])
    # Centroid
    pts.append([0.5, H/3])
    # 4 symmetric points around centroid
    cx, cy = 0.5, H/3
    r = 0.12
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return np.array(pts[:N])

def _init_random(seed):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        x = r1 + 0.5 * r2
        y = H * r2
        pts.append([x, y])
    return np.array(pts)

def _coordinate_descent_polish(pts, time_left_fn):
    """
    Coordinate descent: for each point, optimize its position by trying
    small moves in x and y independently.
    """
    best_pts = pts.copy()
    best_score = _min_area(best_pts)
    
    n_iter = 200
    for it in range(n_iter):
        if time_left_fn() < 5.0:
            break
        
        improved = False
        for idx in range(N):
            px, py = best_pts[idx]
            # Try small perturbations in x
            for dx in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                candidate = best_pts.copy()
                candidate[idx, 0] = px + dx
                candidate = _clamp_to_triangle(candidate)
                score = _min_area(candidate)
                if score > best_score + 1e-15:
                    best_score = score
                    best_pts = candidate.copy()
                    px = candidate[idx, 0]
                    improved = True
            # Try small perturbations in y
            for dy in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                candidate = best_pts.copy()
                candidate[idx, 1] = py + dy
                candidate = _clamp_to_triangle(candidate)
                score = _min_area(candidate)
                if score > best_score + 1e-15:
                    best_score = score
                    best_pts = candidate.copy()
                    py = candidate[idx, 1]
                    improved = True
        
        if not improved:
            break
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
    
    best_pts = None
    best_score = -1.0
    
    # Multiple initializations
    inits = [_init_structured()]
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        if time_left() < 15.0:
            break
        inits.append(_init_random(seed))
    
    for init_pts in inits:
        if time_left() < 10.0:
            break
        
        # Phase 1: Direct critical-triangle maximization
        try:
            pts1, score1 = _direct_maximize_critical(init_pts, time_left)
            if score1 > best_score:
                best_score = score1
                best_pts = pts1.copy()
        except Exception:
            pass
        
        # Phase 2: Coordinate descent polish
        if time_left() > 8.0 and best_pts is not None:
            try:
                pts2, score2 = _coordinate_descent_polish(best_pts, time_left)
                if score2 > best_score:
                    best_score = score2
                    best_pts = pts2.copy()
            except Exception:
                pass
    
    # Final fallback
    if best_pts is None:
        best_pts = _init_structured()
    
    # Final polish pass
    if time_left() > 5.0:
        try:
            pts_final, score_final = _coordinate_descent_polish(best_pts, time_left)
            if score_final > best_score:
                best_score = score_final
                best_pts = pts_final.copy()
        except Exception:
            pass
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Best score: {_min_area(pts)}")
