import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

N = 13

def _min_triangle_area_and_indices(points):
    """Return (min_area, (i,j,k)) for the minimum-area triangle."""
    n = len(points)
    min_area = np.inf
    min_idx = (0, 1, 2)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
    return min_area, min_idx

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def get_score(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2):
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area, _ = _min_triangle_area_and_indices(points)
    return min_area / hull_area

def _compute_all_triangle_areas(points):
    """Compute all C(13,3)=286 triangle areas as a flat array."""
    n = len(points)
    areas = np.empty((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                areas[idx] = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                idx += 1
    return areas

def _soft_min_objective(flat_pts, beta):
    """
    Soft-min of triangle areas / hull area.
    flat_pts is shape (26,). Returns (value, gradient).
    """
    pts = flat_pts.reshape(N, 2)
    
    # Compute all triangle areas
    n = N
    areas = np.empty((n*(n-1)*(n-2))//6)
    # Also track which point indices for gradient
    # For gradient, we need d(area_t)/d(pt_p)
    
    idx = 0
    tri_indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                cross = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
                areas[idx] = 0.5 * abs(cross)
                tri_indices.append((i, j, k, cross))
                idx += 1
    
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # = -(1/beta) * log(sum(exp(-beta * areas)))
    max_a = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_a))
    soft_min_val = max_a - np.log(np.sum(exp_terms)) / beta
    
    # Hull area
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10, np.zeros(2*N)
    
    obj = soft_min_val / hull_area
    
    # Gradient via finite differences (expensive but reliable for 26D)
    # Actually let's do analytical gradient for the soft-min part
    # d(soft_min)/d(areas_t) = exp(-beta*areas_t) / sum(exp(-beta*areas))
    # But we need d(areas_t)/d(pts)
    
    # For speed, use numerical gradient with central differences
    grad = np.zeros(2*N)
    eps = 1e-7
    for d in range(2*N):
        fp = flat_pts.copy()
        fm = flat_pts.copy()
        fp[d] += eps
        fm[d] -= eps
        
        pp = fp.reshape(N, 2)
        pm = fm.reshape(N, 2)
        
        # Compute soft min for perturbed
        ap = np.empty((n*(n-1)*(n-2))//6)
        am = np.empty((n*(n-1)*(n-2))//6)
        idx2 = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    cross_p = (pp[j][0]-pp[i][0])*(pp[k][1]-pp[i][1]) - (pp[j][1]-pp[i][1])*(pp[k][0]-pp[i][0])
                    cross_m = (pm[j][0]-pm[i][0])*(pm[k][1]-pm[i][1]) - (pm[j][1]-pm[i][1])*(pm[k][0]-pm[i][0])
                    ap[idx2] = 0.5 * abs(cross_p)
                    am[idx2] = 0.5 * abs(cross_m)
                    idx2 += 1
        
        hp = _hull_area(pp)
        hm = _hull_area(pm)
        
        if hp < 1e-10 or hm < 1e-10:
            grad[d] = 0.0
            continue
        
        mp = np.max(ap)
        sp = np.exp(-beta * (ap - mp))
        sm = np.max(am)
        sm_arr = np.exp(-beta * (am - sm))
        
        soft_p = (mp - np.log(np.sum(sp)) / beta) / hp
        soft_m = (sm - np.log(np.sum(sm_arr)) / beta) / hm
        
        grad[d] = (soft_p - soft_m) / (2 * eps)
    
    return obj, -grad  # We want to minimize negative of score, so maximize score

def _soft_min_objective_fast(flat_pts, beta):
    """
    Faster soft-min using vectorized area computation.
    """
    pts = flat_pts.reshape(N, 2)
    n = N
    
    # Vectorized triangle area computation
    # For each triple (i,j,k), area = 0.5*|cross(pts[j]-pts[i], pts[k]-pts[i])|
    i_idx = []
    j_idx = []
    k_idx = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    a = pts[i_idx]
    b = pts[j_idx]
    c = pts[k_idx]
    
    cross = (b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(c[:,0]-a[:,0])
    areas = 0.5 * np.abs(cross)
    
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    
    max_a = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_a))
    soft_min_val = max_a - np.log(np.sum(exp_terms)) / beta
    
    return soft_min_val / hull_area

def _soft_min_grad(flat_pts, beta):
    """Gradient of the soft-min objective (to minimize)."""
    pts = flat_pts.reshape(N, 2)
    n = N
    
    i_idx = []
    j_idx = []
    k_idx = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    a = pts[i_idx]
    b = pts[j_idx]
    c = pts[k_idx]
    
    cross = (b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(c[:,0]-a[:,0])
    areas = 0.5 * np.abs(cross)
    
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return np.zeros(2*N)
    
    max_a = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_a))
    weights = exp_terms / np.sum(exp_terms)  # d(soft_min)/d(area_t)
    
    # d(area_t)/d(pts) for each triangle
    # area_t = 0.5 * |cross_t|
    # sign_t = sign(cross_t)
    # d(area_t)/d(a) = 0.5 * sign_t * (-d(cross)/da) where d(cross)/da = (-v1, v0) with v = c-a
    # Actually: cross = (b-a) x (c-a) = (b0-a0)(c1-a1) - (b1-a1)(c0-a0)
    # d(cross)/da0 = -(c1-a1) = -v1
    # d(cross)/da1 = (c0-a0) = v0
    # d(cross)/db0 = (c1-a1) = v1
    # d(cross)/db1 = -(c0-a0) = -v0
    # d(cross)/dc0 = -(b1-a1) = -u1
    # d(cross)/dc1 = (b0-a0) = u0
    # where u = b-a, v = c-a
    
    signs = np.sign(cross)
    # Handle zeros
    signs[signs == 0] = 1.0
    
    u = b - a  # (T, 2)
    v = c - a  # (T, 2)
    
    # For each triangle t, contribution to gradient of point p:
    # d(soft_min)/d(pts[p]) = sum_t weights[t] * 0.5 * sign[t] * d(cross_t)/d(pts[p])
    
    grad = np.zeros((n, 2))
    
    # For point i (as 'a' in triangle):
    for t in range(len(i_idx)):
        i = i_idx[t]
        j = j_idx[t]
        k = k_idx[t]
        w = weights[t] * 0.5 * signs[t]
        
        # d(area)/da = 0.5*sign*(-v1, v0)
        grad[i] += w * np.array([-v[t,1], v[t,0]])
        # d(area)/db = 0.5*sign*(v1, -v0)
        grad[j] += w * np.array([v[t,1], -v[t,0]])
        # d(area)/dc = 0.5*sign*(-u1, u0)
        grad[k] += w * np.array([-u[t,1], u[t,0]])
    
    # Now divide by hull_area and negate (we're minimizing -score)
    grad_flat = grad.reshape(2*N)
    
    # Also need d(hull_area)/d(pts) for the quotient rule
    # This is complex; approximate by treating hull_area as constant
    # (it changes slowly compared to min triangle area)
    
    return -grad_flat / hull_area  # Negative because we want to minimize

def _optimize_soft_min(init_pts, deadline, betas=[30, 80, 200, 500, 1000]):
    """
    Optimize using L-BFGS-B on the soft-min surrogate with progressive beta.
    """
    flat = init_pts.reshape(-1).copy()
    best_flat = flat.copy()
    best_score = get_score(init_pts)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        def obj(x):
            return _soft_min_objective_fast(x, beta)
        
        def grad(x):
            return _soft_min_grad(x, beta)
        
        try:
            result = minimize(obj, flat, jac=grad, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                                   'maxfun': 500})
            if result.fun < obj(best_flat.reshape(-1)):
                new_score = get_score(result.x.reshape(N, 2))
                if new_score > best_score:
                    best_score = new_score
                    best_flat = result.x.copy()
            flat = result.x.copy()
        except Exception:
            pass
    
    return best_flat.reshape(N, 2), best_score

def _nelder_mead_polish(init_pts, deadline):
    """Final polish with Nelder-Mead on the exact objective."""
    flat = init_pts.reshape(-1).copy()
    best_flat = flat.copy()
    best_score = get_score(init_pts)
    
    def neg_score(x):
        pts = x.reshape(N, 2)
        return -get_score(pts)
    
    try:
        result = minimize(neg_score, flat, method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10,
                               'adaptive': True})
        new_score = get_score(result.x.reshape(N, 2))
        if new_score > best_score:
            best_score = new_score
            best_flat = result.x.copy()
    except Exception:
        pass
    
    return best_flat.reshape(N, 2), best_score

def _generate_structured_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Regular 12-gon + center
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    configs.append(np.vstack([pts12, np.array([[0.0, 0.0]])]))
    
    # Two concentric rings: 7 outer, 6 inner
    angles7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(angles7), np.sin(angles7)])
    angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    inner = np.column_stack([0.55*np.cos(angles6), 0.55*np.sin(angles6)])
    configs.append(np.vstack([outer, inner]))
    
    # 8 outer, 5 inner
    angles8 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    outer = np.column_stack([np.cos(angles8), np.sin(angles8)])
    angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    inner = np.column_stack([0.5*np.cos(angles5), 0.5*np.sin(angles5)])
    configs.append(np.vstack([outer, inner]))
    
    # 9 outer, 4 inner
    angles9 = np.linspace(0, 2*np.pi, 9, endpoint=False)
    outer = np.column_stack([np.cos(angles9), np.sin(angles9)])
    angles4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    inner = np.column_stack([0.45*np.cos(angles4), 0.45*np.sin(angles4)])
    configs.append(np.vstack([outer, inner]))
    
    # 10 outer, 3 inner
    angles10 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    outer = np.column_stack([np.cos(angles10), np.sin(angles10)])
    angles3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    inner = np.column_stack([0.4*np.cos(angles3), 0.4*np.sin(angles3)])
    configs.append(np.vstack([outer, inner]))
    
    # 11 outer, 2 inner
    angles11 = np.linspace(0, 2*np.pi, 11, endpoint=False)
    outer = np.column_stack([np.cos(angles11), np.sin(angles11)])
    inner = np.array([[0.3, 0.0], [-0.3, 0.0]])
    configs.append(np.vstack([outer, inner]))
    
    # 6 outer, 7 inner (more interior points)
    angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer = np.column_stack([np.cos(angles6), np.sin(angles6)])
    # 7 inner: 1 center + 6 at radius 0.4
    inner6 = np.column_stack([0.4*np.cos(angles6 + np.pi/6), 0.4*np.sin(angles6 + np.pi/6)])
    configs.append(np.vstack([outer, np.array([[0.0, 0.0]]), inner6]))
    
    # Random configurations with various seeds
    for seed in [42, 123, 777, 2024, 999, 31415]:
        rng = np.random.default_rng(seed)
        pts = rng.random((N, 2)) * 2 - 1
        # Ensure reasonable spread
        configs.append(pts)
    
    # Delaunay-inspired: start from a grid-like perturbation
    rng = np.random.default_rng(555)
    # 4x4 grid minus one, with jitter
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if len(grid_pts) < N:
                grid_pts.append([i/3.0, j/3.0])
    grid_pts = np.array(grid_pts[:N])
    grid_pts += rng.normal(0, 0.1, grid_pts.shape)
    configs.append(grid_pts)
    
    # Another grid variant
    rng = np.random.default_rng(888)
    grid_pts2 = []
    for i in range(5):
        for j in range(3):
            if len(grid_pts2) < N:
                grid_pts2.append([i/4.0, j/2.0])
    grid_pts2 = np.array(grid_pts2[:N])
    grid_pts2 += rng.normal(0, 0.08, grid_pts2.shape)
    configs.append(grid_pts2)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (evaluator kills at 150s)
    
    configs = _generate_structured_configs()
    
    best_score = -np.inf
    best_points = None
    
    # Phase 1: Soft-min optimization from multiple starts
    for idx, init_points in enumerate(configs):
        if time.time() > deadline - 30.0:
            break
        
        try:
            optimized, score = _optimize_soft_min(init_points, deadline - 30.0)
            if score > best_score:
                best_score = score
                best_points = optimized
        except Exception:
            pass
    
    if best_points is None:
        best_points = configs[0]
        best_score = get_score(best_points)
    
    # Phase 2: Nelder-Mead polish on the best result
    if time.time() < deadline - 5.0:
        try:
            polished, score = _nelder_mead_polish(best_points, deadline - 5.0)
            if score > best_score:
                best_score = score
                best_points = polished
        except Exception:
            pass
    
    # Phase 3: Try a few more NM polishes from other good starts
    if time.time() < deadline - 5.0 and best_points is not None:
        # Try polishing the second-best config too
        for idx, init_points in enumerate(configs[:4]):
            if time.time() > deadline - 5.0:
                break
            if idx == 0:
                continue
            try:
                polished, score = _nelder_mead_polish(init_points, deadline - 5.0)
                if score > best_score:
                    best_score = score
                    best_points = polished
            except Exception:
                pass
    
    return best_points
