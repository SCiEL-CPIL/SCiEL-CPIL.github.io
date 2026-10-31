import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import itertools

N = 11
H = np.sqrt(3) / 2
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute all C(11,3)=165 triangle index triplets
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Vectorized computation of all 165 triangle areas."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area(pts):
    """Minimum area among all triangles."""
    return float(np.min(_all_areas(pts)))

def _soft_min(pts, beta=100.0):
    """Smooth approximation of min area for gradient-based methods."""
    areas = _all_areas(pts)
    m = np.min(areas)
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -400, 400)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / beta

def _delaunay_smooth_obj(pts, beta=50.0):
    """
    Smooth objective that focuses on Delaunay triangles,
    which are more likely to contain the minimum area.
    """
    try:
        tri = Delaunay(pts)
        # Get unique triangles from Delaunay
        simplices = tri.simplices
        # Compute areas of Delaunay triangles
        pi = pts[simplices[:, 0]]
        pj = pts[simplices[:, 1]]
        pk = pts[simplices[:, 2]]
        dx1 = pj[:, 0] - pi[:, 0]
        dy1 = pj[:, 1] - pi[:, 1]
        dx2 = pk[:, 0] - pi[:, 0]
        dy2 = pk[:, 1] - pi[:, 1]
        delaunay_areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
        
        # Also include all areas for safety
        all_areas = _all_areas(pts)
        # Combine: use soft min of Delaunay areas with higher weight
        # and soft min of all areas with lower weight
        dm = np.min(delaunay_areas)
        shifted_d = -beta * (delaunay_areas - dm)
        shifted_d = np.clip(shifted_d, -400, 400)
        log_sum_d = np.log(np.sum(np.exp(shifted_d)))
        soft_delaunay = dm - log_sum_d / beta
        
        am = np.min(all_areas)
        shifted_a = -beta * 0.5 * (all_areas - am)
        shifted_a = np.clip(shifted_a, -400, 400)
        log_sum_a = np.log(np.sum(np.exp(shifted_a)))
        soft_all = am - log_sum_a / (beta * 0.5)
        
        # Weighted combination
        return 0.6 * soft_delaunay + 0.4 * soft_all
    except Exception:
        return _soft_min(pts, beta)

def _clamp_point(p):
    """Clamp point to triangle using barycentric coordinates."""
    x, y = p
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    
    eps = 1e-9
    u = max(eps, min(1.0 - 2*eps, u))
    v = max(eps, min(1.0 - 2*eps, v))
    w = max(eps, min(1.0 - 2*eps, w))
    s = u + v + w
    u /= s; v /= s; w /= s
    return u * V0 + v * V1 + w * V2

def _clamp_all(pts):
    """Clamp all points inside triangle."""
    return np.array([_clamp_point(p) for p in pts])

def _obj_smooth(flat, beta=100.0):
    """Objective for L-BFGS-B: negative soft-min area."""
    pts = _clamp_all(flat.reshape(-1, 2))
    return -_soft_min(pts, beta)

def _obj_delaunay(flat, beta=50.0):
    """Objective using Delaunay-focused smooth min."""
    pts = _clamp_all(flat.reshape(-1, 2))
    return -_delaunay_smooth_obj(pts, beta)

def _obj_hard(flat):
    """Objective for Nelder-Mead: negative hard min area."""
    pts = _clamp_all(flat.reshape(-1, 2))
    return -_min_area(pts)

def _init_hexagonal():
    """Hexagonal grid initialization."""
    cx, cy = 1/3.0, H/3.0
    pts = []
    pts.append([cx, cy])
    r1 = 0.15
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([cx + r1 * np.cos(angle), cy + r1 * np.sin(angle)])
    r2 = 0.35
    angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for angle in angles:
        pts.append([cx + r2 * np.cos(angle), cy + r2 * np.sin(angle)])
    return np.array(pts[:N])

def _init_grid():
    """Grid-based initialization: 3x3 plus 2 extra points."""
    # 3x3 grid in barycentric coordinates, mapped to Cartesian
    # Barycentric grid positions
    grid_pts = []
    for i in range(3):
        for j in range(3):
            # Map (i,j) to barycentric (u,v,w) with u+v+w=1
            # Use i/2, j/2 scaled
            u = (1 - i/2.0) * (1 - j/2.0)
            v = (i/2.0) * (1 - j/2.0)
            w = (j/2.0)
            # Normalize
            s = u + v + w
            u /= s; v /= s; w /= s
            pt = u * V0 + v * V1 + w * V2
            grid_pts.append(pt)
    # Add 2 extra points near center
    cx, cy = 1/3.0, H/3.0
    grid_pts.append([cx - 0.05, cy])
    grid_pts.append([cx + 0.05, cy])
    return np.array(grid_pts[:N])

def _init_random(seed):
    """Random initialization inside triangle."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1, r2 = rng.random(), rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        pts.append(r1 * V1 + r2 * V2)
    return np.array(pts)

def _lbfgs_optimize(pts, beta_start=50.0, beta_end=500.0, n_steps=10, maxiter_per_step=300, deadline=None, use_delaunay=False):
    """L-BFGS-B optimization with annealing beta."""
    current = pts
    for i in range(n_steps):
        if deadline and time.time() > deadline:
            break
        beta = beta_start + (beta_end - beta_start) * (i / max(n_steps - 1, 1))
        try:
            if use_delaunay:
                res = minimize(
                    _obj_delaunay, current.flatten(), args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': maxiter_per_step, 'ftol': 1e-14, 'gtol': 1e-12}
                )
            else:
                res = minimize(
                    _obj_smooth, current.flatten(), args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': maxiter_per_step, 'ftol': 1e-14, 'gtol': 1e-12}
                )
            current = _clamp_all(res.x.reshape(-1, 2))
        except Exception:
            break
    return current

def _nm_optimize(pts, maxiter=2000, deadline=None):
    """Nelder-Mead optimization on hard objective."""
    try:
        res = minimize(
            _obj_hard, pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        return _clamp_all(res.x.reshape(-1, 2))
    except Exception:
        return pts

def _local_polish(pts, deadline=None):
    """
    Custom local refinement: iteratively perturb the worst triangle's vertices
    and re-optimize.
    """
    current = pts.copy()
    best_score = _min_area(current)
    
    for iteration in range(50):
        if deadline and time.time() > deadline:
            break
        
        areas = _all_areas(current)
        min_idx = np.argmin(areas)
        min_area_val = areas[min_idx]
        
        # Get the three points forming the worst triangle
        i, j, k = _TRI_IDX[min_idx]
        
        # Try moving each of the three points slightly away from the others
        for move_idx in [i, j, k]:
            if deadline and time.time() > deadline:
                break
            
            # Compute direction to move away from the centroid of the triangle
            centroid = (current[i] + current[j] + current[k]) / 3.0
            direction = current[move_idx] - centroid
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([0.1, 0.1])
            else:
                direction /= norm
            
            # Try several small moves
            for scale in [0.01, 0.02, 0.05, 0.1]:
                if deadline and time.time() > deadline:
                    break
                
                trial = current.copy()
                trial[move_idx] = trial[move_idx] + scale * direction
                trial = _clamp_all(trial)
                
                # Quick polish with NM
                trial = _nm_optimize(trial, maxiter=500, deadline=deadline)
                score = _min_area(trial)
                
                if score > best_score:
                    best_score = score
                    current = trial.copy()
                    break
    
    return current

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    def tleft():
        return deadline - time.time()
    
    best_pts = None
    best_score = -1.0
    
    # Initializations
    inits = [_init_hexagonal(), _init_grid()]
    for s in [1, 7, 13, 42, 100, 200, 300, 500, 700, 900, 1200, 1500, 2000, 2500, 3000]:
        inits.append(_init_random(s))
    
    # Phase 1: Multi-start L-BFGS with annealing (both standard and Delaunay)
    for init in inits:
        if tleft() < 15:
            break
        
        # Standard soft-min
        pts = _lbfgs_optimize(init, beta_start=50, beta_end=500, n_steps=8, maxiter_per_step=200, deadline=deadline, use_delaunay=False)
        pts = _nm_optimize(pts, maxiter=1000, deadline=deadline)
        score = _min_area(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Delaunay-focused
        if tleft() < 10:
            break
        pts_d = _lbfgs_optimize(init, beta_start=50, beta_end=500, n_steps=8, maxiter_per_step=200, deadline=deadline, use_delaunay=True)
        pts_d = _nm_optimize(pts_d, maxiter=1000, deadline=deadline)
        score_d = _min_area(pts_d)
        if score_d > best_score:
            best_score = score_d
            best_pts = pts_d.copy()
    
    if best_pts is None:
        best_pts = _init_hexagonal()
        best_score = _min_area(best_pts)
        
    # Phase 2: Local search around best solution
    rng = np.random.default_rng(99999)
    for trial in range(40):
        if tleft() < 10:
            break
        scale = 0.05 * (0.9 ** trial)
        if scale < 0.001:
            scale = 0.001
        pert = best_pts + rng.normal(0, scale, size=(N, 2))
        pert = _clamp_all(pert)
        
        pts = _lbfgs_optimize(pert, beta_start=100, beta_end=1000, n_steps=5, maxiter_per_step=150, deadline=deadline)
        pts = _nm_optimize(pts, maxiter=1500, deadline=deadline)
        score = _min_area(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    # Phase 3: Custom local polish focusing on worst triangles
    if tleft() > 15:
        best_pts = _local_polish(best_pts, deadline=deadline)
        best_score = _min_area(best_pts)
        
    # Phase 4: Final high-precision polish
    if tleft() > 5:
        best_pts = _nm_optimize(best_pts, maxiter=3000, deadline=deadline)
        best_score = _min_area(best_pts)
        
    return best_pts
