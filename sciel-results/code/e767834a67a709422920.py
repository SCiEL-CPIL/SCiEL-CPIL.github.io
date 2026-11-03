import numpy as np
import time
from scipy.optimize import minimize

N = 11
T_LIMIT = 120.0

def _tri_area(a, b, c):
    return 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))

def _min_tri_area(pts):
    n = len(pts)
    min_a = float('inf')
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                a = _tri_area(pts[i], pts[j], pts[k])
                if a < min_a:
                    min_a = a
    return min_a

def _in_triangle(p):
    """Check if point is inside or on boundary of triangle (0,0),(1,0),(0.5,sqrt(3)/2)."""
    x, y = p
    if y < -1e-10:
        return False
    if y > np.sqrt(3)/2 + 1e-10:
        return False
    if x < -1e-10 or x > 1 + 1e-10:
        return False
    # Check y <= sqrt(3)*x and y <= sqrt(3)*(1-x)
    if y > np.sqrt(3)*x + 1e-10:
        return False
    if y > np.sqrt(3)*(1-x) + 1e-10:
        return False
    return True

def _project_pt(p):
    """Project point to inside the triangle."""
    x, y = float(p[0]), float(p[1])
    # Simple iterative clipping
    for _ in range(20):
        if y < 0:
            y = 0
        if y > np.sqrt(3)/2:
            y = np.sqrt(3)/2
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        if y > np.sqrt(3)*x:
            # Project onto line y = sqrt(3)*x
            # Normal form: sqrt(3)*x - y = 0, normal = (sqrt(3), -1), norm = 2
            d = (np.sqrt(3)*x - y) / 2
            x = x + d * np.sqrt(3)
            y = y + d * (-1)
        if y > np.sqrt(3)*(1-x):
            # Line: sqrt(3)*x + y - sqrt(3) = 0, normal = (sqrt(3), 1), norm = 2
            d = (np.sqrt(3)*(1-x) - y) / 2
            x = x + d * np.sqrt(3)
            y = y + d * 1
        if _in_triangle(np.array([x, y])):
            break
    return np.array([x, y])

def _project_pts(pts):
    return np.array([_project_pt(p) for p in pts])

def _smooth_min_obj(x_flat, beta):
    """Negative smooth minimum triangle area."""
    pts = x_flat.reshape(N, 2)
    n = N
    areas = []
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                areas.append(_tri_area(pts[i], pts[j], pts[k]))
    areas = np.array(areas)
    areas = np.maximum(areas, 1e-12)
    m = np.min(areas)
    # log-sum-exp trick
    shifted = areas - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    smooth_min = m - log_sum / beta
    return -smooth_min

def _make_starts():
    """Generate diverse starting configurations."""
    starts = []
    V0 = np.array([0.0, 0.0])
    V1 = np.array([1.0, 0.0])
    V2 = np.array([0.5, np.sqrt(3)/2])
    C = np.array([0.5, np.sqrt(3)/6])  # centroid
    
    # Symmetric template 1: 3 vertices + 6 edge pts (1/3,2/3) + 2 interior
    pts = [V0, V1, V2]
    for frac in [1/3, 2/3]:
        pts.append(V0 + frac*(V1-V0))
        pts.append(V1 + frac*(V2-V1))
        pts.append(V2 + frac*(V0-V2))
    pts.append(C)
    pts.append(C + np.array([0, 0.08]))
    if len(pts) == N:
        starts.append(np.array(pts))
    
    # Symmetric template 2: different fractions
    pts = [V0, V1, V2]
    for frac in [0.35, 0.65]:
        pts.append(V0 + frac*(V1-V0))
        pts.append(V1 + frac*(V2-V1))
        pts.append(V2 + frac*(V0-V2))
    pts.append(C)
    pts.append(C + np.array([0, 0.12]))
    if len(pts) == N:
        starts.append(np.array(pts))
    
    # Random starts
    for seed in range(20):
        rng = np.random.RandomState(seed)
        pts = [V0, V1, V2]
        # 6 edge points
        for _ in range(6):
            edge = rng.randint(3)
            t = rng.uniform(0.15, 0.85)
            if edge == 0:
                pts.append(V0 + t*(V1-V0))
            elif edge == 1:
                pts.append(V1 + t*(V2-V1))
            else:
                pts.append(V2 + t*(V0-V2))
        # 2 interior
        for _ in range(2):
            for _attempt in range(50):
                u, v = rng.random(), rng.random()
                if u + v < 0.9:
                    p = V0 + u*(V1-V0) + v*(V2-V0)
                    pts.append(p)
                    break
            else:
                pts.append(C)
        if len(pts) == N:
            starts.append(np.array(pts))
    
    # Grid-like
    pts = [V0, V1, V2]
    h1 = np.sqrt(3)/6
    pts.append(np.array([1/3, h1]))
    pts.append(np.array([2/3, h1]))
    h2 = np.sqrt(3)/3
    pts.append(np.array([1/6, h2]))
    pts.append(np.array([1/2, h2]))
    pts.append(np.array([5/6, h2]))
    pts.append(C)
    pts.append(np.array([0.5, np.sqrt(3)/6 + 0.1*np.sqrt(3)/6]))
    if len(pts) == N:
        starts.append(np.array(pts))
    
    return starts

def construct():
    t0 = time.time()
    deadline = t0 + T_LIMIT - 5.0
    
    best_pts = None
    best_val = 0.0
    
    starts = _make_starts()
    
    for start in starts:
        if time.time() > deadline:
            break
        
        for beta in [15, 40]:
            if time.time() > deadline:
                break
            
            x0 = start.flatten()
            
            def obj(x):
                return _smooth_min_obj(x, beta)
            
            try:
                res = minimize(obj, x0, method='Nelder-Mead',
                               options={'maxiter': 300, 'fatol': 1e-10, 'xatol': 1e-10})
                if hasattr(res, 'x'):
                    new_pts = res.x.reshape(N, 2)
                    new_pts = _project_pts(new_pts)
                    val = _min_tri_area(new_pts)
                    if val > best_val:
                        best_val = val
                        best_pts = new_pts
            except Exception:
                pass
        
        # Local perturbation refinement
        if best_pts is not None and time.time() < deadline:
            current = best_pts.copy()
            current_val = best_val
            rng = np.random.RandomState(int(time.time()*1000) % 10000)
            for _ in range(100):
                if time.time() > deadline:
                    break
                pert = current + 0.005 * rng.randn(N, 2)
                pert = _project_pts(pert)
                val = _min_tri_area(pert)
                if val > current_val:
                    current = pert
                    current_val = val
                    if current_val > best_val:
                        best_val = current_val
                        best_pts = current
    
    if best_pts is None:
        # Fallback
        rng = np.random.RandomState(7)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                pts.append([u + 0.5*v, np.sqrt(3)/2*v])
        best_pts = np.array(pts)
    
    return best_pts
