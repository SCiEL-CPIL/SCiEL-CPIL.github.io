"""Heilbronn triangles: 11 points in unit-area equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)

# Vertices of the equilateral triangle with area 1
# Side length s = sqrt(4/sqrt(3)) = 2 * 3^(-1/4)
S = 2.0 * (3.0 ** (-0.25))
A = np.array([0.0, 0.0])
B = np.array([S, 0.0])
C = np.array([S / 2.0, S * S3 / 2.0])

def _bary_to_xy(u, v, w):
    """Convert barycentric (u,v,w) with u+v+w=1 to cartesian."""
    return u * A + v * B + w * C

def _xy_to_bary(pt):
    """Convert cartesian to barycentric (u,v,w)."""
    x, y = pt
    # Solve: pt = u*A + v*B + w*C, u+v+w=1
    # A=(0,0), B=(S,0), C=(S/2, S*sqrt(3)/2)
    # x = v*S + w*S/2
    # y = w*S*sqrt(3)/2
    w = y / (S * S3 / 2.0)
    v = (x - w * S / 2.0) / S
    u = 1.0 - v - w
    return u, v, w

def _area2(p1, p2, p3):
    """Twice the signed area of triangle (p1, p2, p3)."""
    return abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])) / 2.0

def _min_area(pts):
    """Minimum area among all triangles formed by 3 of the N points."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _area2(pts[i], pts[j], pts[k])
                if a < min_a:
                    min_a = a
    return min_a

def _min_area_vec(pts):
    """Vectorized min area for speed."""
    n = len(pts)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _area2(pts[i], pts[j], pts[k])
                areas[i,j,k] = a
                areas[j,i,k] = a
                areas[k,j,i] = a
    # Only consider i<j<k to avoid duplicates, but we stored all
    # Find min over i<j<k
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if areas[i,j,k] < min_a:
                    min_a = areas[i,j,k]
    return min_a

def _objective(flat_pts):
    """Objective to minimize: negative of min area (with penalty for out-of-bounds)."""
    pts = flat_pts.reshape(-1, 2)
    # Convert to barycentric to check bounds
    penalty = 0.0
    for pt in pts:
        u, v, w = _xy_to_bary(pt)
        if u < -1e-6 or v < -1e-6 or w < -1e-6:
            penalty += max(0, -u) + max(0, -v) + max(0, -w)
    if penalty > 1e-3:
        return 1e6 + penalty  # Large penalty
    min_a = _min_area(pts)
    return -min_a

def _project_to_triangle(pts):
    """Project points onto the triangle (clamp barycentric coords)."""
    result = []
    for pt in pts:
        u, v, w = _xy_to_bary(pt)
        # Clamp to simplex
        eps = 1e-9
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        w = max(0.0, min(1.0, w))
        # Renormalize
        s = u + v + w
        if s < 1e-12:
            u, v, w = 1/3, 1/3, 1/3
        else:
            u, v, w = u/s, v/s, w/s
        result.append(_bary_to_xy(u, v, w))
    return np.array(result)

def construct():
    deadline = time.time() + 130.0  # Safety margin
    best_score = 0.0
    best_pts = None
    
    # Initial configurations to try
    inits = []
    
    # 1. Grid-like: 3x3 grid in barycentric + corners adjusted
    # Use a structured set of barycentric coordinates
    grid_pts = []
    # Try a 3x3 grid in barycentric space (u,v with w=1-u-v)
    for i in range(3):
        for j in range(3):
            u = 0.1 + 0.4 * i / 2.0
            v = 0.1 + 0.4 * j / 2.0
            w = 1.0 - u - v
            if u >= 0 and v >= 0 and w >= 0:
                grid_pts.append(_bary_to_xy(u, v, w))
    # Add 2 more points if needed
    while len(grid_pts) < N:
        grid_pts.append(_bary_to_xy(1/3, 1/3, 1/3))
    inits.append(np.array(grid_pts[:N]))
    
    # 2. Random but well-distributed
    rng = np.random.RandomState(42)
    for trial in range(5):
        pts = []
        for _ in range(N):
            # Random point in triangle
            r1, r2 = rng.random(), rng.random()
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            u, v, w = r1, r2, 1 - r1 - r2
            pts.append(_bary_to_xy(u, v, w))
        inits.append(np.array(pts))
    
    # 3. Symmetric-ish: centroid + ring
    centroid = _bary_to_xy(1/3, 1/3, 1/3)
    ring_pts = [centroid]
    for angle in range(10):
        theta = 2 * np.pi * angle / 10
        r = 0.35
        # Place in barycentric: move from centroid
        # Use cartesian offset projected
        offset = np.array([r * np.cos(theta), r * np.sin(theta)])
        pt = centroid + offset
        u, v, w = _xy_to_bary(pt)
        if u >= -0.01 and v >= -0.01 and w >= -0.01:
            pt = _project_to_triangle(pt.reshape(1,2))[0]
        ring_pts.append(pt)
    inits.append(np.array(ring_pts[:N]))
    
    # 4. Fekete-like: maximize min distance first, then optimize area
    # Start with well-spread points
    spread_pts = []
    # Place points at vertices and midpoints and centroid
    spread_pts.append(A + 0.05*(B-A))  # near A
    spread_pts.append(B - 0.05*(B-A))  # near B
    spread_pts.append(C - 0.05*(C-A))  # near C
    spread_pts.append((A+B)/2)
    spread_pts.append((B+C)/2)
    spread_pts.append((C+A)/2)
    spread_pts.append(centroid)
    # Add 4 more
    for angle in range(4):
        theta = 2 * np.pi * angle / 4 + np.pi/4
        r = 0.25
        offset = np.array([r * np.cos(theta), r * np.sin(theta)])
        pt = centroid + offset
        pt = _project_to_triangle(pt.reshape(1,2))[0]
        spread_pts.append(pt)
    inits.append(np.array(spread_pts[:N]))
    
    # Optimize each initial configuration
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Project to ensure inside
        init_pts = _project_to_triangle(init_pts)
        
        flat_init = init_pts.flatten()
        
        # Use L-BFGS-B to maximize min area (minimize negative)
        try:
            res = minimize(
                _objective,
                flat_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if time.time() > deadline:
                break
            
            opt_pts = _project_to_triangle(res.x.reshape(-1, 2))
            score = _min_area(opt_pts)
            
            if score > best_score:
                best_score = score
                best_pts = opt_pts
        except Exception:
            pass
    
    # If no optimization succeeded, use best initial
    if best_pts is None:
        for init_pts in inits:
            proj = _project_to_triangle(init_pts)
            s = _min_area(proj)
            if s > best_score:
                best_score = s
                best_pts = proj
    
    # Final safety: ensure all points are strictly inside
    if best_pts is not None:
        best_pts = _project_to_triangle(best_pts)
        # Slight inset to avoid boundary issues
        for i in range(len(best_pts)):
            u, v, w = _xy_to_bary(best_pts[i])
            eps = 0.005
            u = max(eps, min(1 - 2*eps, u))
            v = max(eps, min(1 - 2*eps, v))
            w = max(eps, 1 - u - v)
            s = u + v + w
            if s > 0:
                best_pts[i] = _bary_to_xy(u/s, v/s, w/s)
    
    # Return in the expected coordinate system: (0,0), (1,0), (1/2, sqrt(3)/2)
    # Our triangle has side S, so scale down by S
    if best_pts is not None:
        return best_pts / S
    else:
        # Fallback: uniform grid
        pts = []
        for i in range(3):
            for j in range(3):
                x = 0.1 + 0.3 * i
                y = 0.1 + 0.3 * j * (S3 / 2) / S
                # Check inside
                if x <= 1 - y / (S3/2) and x >= 0 and y >= 0:
                    pts.append([x, y])
        while len(pts) < N:
            pts.append([0.5, 0.3])
        return np.array(pts[:N])
