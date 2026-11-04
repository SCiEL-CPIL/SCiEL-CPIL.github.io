"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# Pre-compute indices for N=14 for fast triangle area calculation
_IDX_CACHE = {}
def _get_indices(n):
    if n not in _IDX_CACHE:
        i_idx, j_idx, k_idx = [], [], []
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    i_idx.append(i)
                    j_idx.append(j)
                    k_idx.append(k)
        _IDX_CACHE[n] = (np.array(i_idx), np.array(j_idx), np.array(k_idx))
    return _IDX_CACHE[n]

def _all_triangle_areas(pts):
    n = len(pts)
    if n < 3:
        return np.array([0.0])
    i_idx, j_idx, k_idx = _get_indices(n)
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(pts):
    return float(np.min(_all_triangle_areas(pts)))

def _convex_hull_area(pts):
    try:
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 1e-10
        hull = ConvexHull(pts)
        return float(hull.volume)
    except Exception:
        return 1e-10

def _score(pts):
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _soft_min_area(pts, beta):
    """Differentiable soft minimum using log-sum-exp."""
    areas = _all_triangle_areas(pts)
    min_a = np.min(areas)
    shifted = areas - min_a
    max_shift = np.max(shifted)
    if max_shift > 50:
        return min_a
    # log-sum-exp for numerical stability
    m = -beta * shifted
    m_max = np.max(m)
    log_sum = m_max + np.log(np.sum(np.exp(m - m_max)))
    return min_a - (1.0 / beta) * log_sum

def _soft_score(pts, beta):
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    soft_min = _soft_min_area(pts, beta)
    return soft_min / hull_area

def _two_ring_points(r, alpha):
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = angles_outer + alpha
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
    return np.vstack([outer_pts, inner_pts])

def _three_ring_points(r1, r2, a1, a2):
    """Three rings: 5 outer, 4 middle, 5 inner."""
    ang5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    ang4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    outer = np.column_stack([np.cos(ang5), np.sin(ang5)])
    middle = np.column_stack([r1 * np.cos(ang4 + a1), r1 * np.sin(ang4 + a1)])
    inner = np.column_stack([r2 * np.cos(ang5 + a2), r2 * np.sin(ang5 + a2)])
    return np.vstack([outer, middle, inner])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = -np.inf
    best_pts = None
    
    def update_best(pts):
        nonlocal best_score, best_pts
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            return True
        return False

    # Generate diverse initial configurations
    inits = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two-ring with various parameters
    for r in [0.3, 0.5, 0.7]:
        for alpha in [np.pi/7, np.pi/14, 2*np.pi/7]:
            inits.append(_two_ring_points(r, alpha))
    
    # 3. Three-ring configurations
    for r1, r2 in [(0.6, 0.3), (0.5, 0.2), (0.7, 0.4)]:
        for a1, a2 in [(np.pi/4, np.pi/5), (0, np.pi/10), (np.pi/8, np.pi/12)]:
            inits.append(_three_ring_points(r1, r2, a1, a2))
    
    # 4. Random convex sets
    rng = np.random.RandomState(42)
    for _ in range(3):
        angles_rand = np.sort(rng.uniform(0, 2*np.pi, N))
        radii = rng.uniform(0.5, 1.0, N)
        pts_rand = np.column_stack([radii * np.cos(angles_rand), radii * np.sin(angles_rand)])
        inits.append(pts_rand)
    
    # Evaluate all initializations
    for init in inits:
        update_best(init)
    
    # Multi-start local optimization with differentiable soft-min surrogate
    if best_pts is not None:
        current_pts = best_pts.copy()
        current_score = best_score
        
        # Continuation on beta for each diverse start
        starts_to_optimize = [best_pts] + inits[:6]  # Best + top diverse starts
        
        for idx, start_pts in enumerate(starts_to_optimize):
            if time.time() > deadline - 15.0:
                break
            
            pts_curr = start_pts.copy()
            score_curr = _score(pts_curr)
            
            # Continuation: ramp up beta for sharper approximation of hard min
            betas = [10.0, 50.0, 200.0, 1000.0]
            for beta in betas:
                if time.time() > deadline - 10.0:
                    break
                try:
                    def neg_soft(x, b=beta):
                        p = x.reshape(-1, 2)
                        return -_soft_score(p, b)
                    
                    res = minimize(
                        neg_soft,
                        pts_curr.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                    )
                    new_pts = res.x.reshape(-1, 2)
                    new_score = _score(new_pts)
                    if new_score > score_curr:
                        pts_curr = new_pts
                        score_curr = new_score
                except Exception:
                    pass
            
            if score_curr > current_score:
                current_pts = pts_curr
                current_score = score_curr
                update_best(pts_curr)
        
        # Final hard polish on the best solution using coordinate descent
        if time.time() < deadline - 5.0:
            pts_polish = current_pts.copy()
            improved = True
            max_rounds = 20
            for rnd in range(max_rounds):
                if time.time() > deadline - 3.0:
                    break
                if not improved:
                    break
                improved = False
                for i in range(N):
                    for dim in range(2):
                        # Try small perturbations in each direction
                        for delta in [0.01, -0.01, 0.005, -0.005]:
                            trial = pts_polish.copy()
                            trial[i, dim] += delta
                            s_trial = _score(trial)
                            if s_trial > _score(pts_polish) + 1e-12:
                                pts_polish = trial
                                improved = True
                                break
                if improved:
                    update_best(pts_polish)
    
    if best_pts is None:
        return inits[0]
        
    return best_pts
# EVOLVE-BLOCK-END
