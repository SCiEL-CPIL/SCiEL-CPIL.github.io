import numpy as np
import time
from scipy.optimize import minimize, basinhopping
from scipy.spatial import ConvexHull

N = 14

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Vectorized for speed.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
        
    n = len(points)
    # Indices for all combinations of 3 points
    # i, j, k with i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1).T
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    # Vectorized cross product magnitude / 2
    # area = 0.5 * | (p_j - p_i) x (p_k - p_i) |
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
        
    return min_area / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    score = compute_score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = 115.0 # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.default_rng(42)
    
    inits = []
    
    # 1. Regular N-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(p_reg)
    
    # 2. Two concentric rings (7 points each)
    # Outer radius 1.0, Inner radius 0.5
    # Rotate inner by pi/7
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = angles_outer + np.pi/7
    p_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    p_inner = np.column_stack((0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)))
    p_two_rings = np.vstack((p_outer, p_inner))
    inits.append(p_two_rings)
    
    # 3. Three rings: 5 outer, 5 middle, 4 inner
    # Try different radii and rotations
    n1, n2, n3 = 5, 5, 4
    r1, r2, r3 = 1.0, 0.6, 0.3
    a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + np.pi/n2
    a3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + np.pi/n3
    p1 = np.column_stack((r1*np.cos(a1), r1*np.sin(a1)))
    p2 = np.column_stack((r2*np.cos(a2), r2*np.sin(a2)))
    p3 = np.column_stack((r3*np.cos(a3), r3*np.sin(a3)))
    p_three_rings = np.vstack((p1, p2, p3))
    inits.append(p_three_rings)
    
    # 4. Random points in unit circle
    for _ in range(5):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        p_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        inits.append(p_rand)
        
    # 5. Jittered regular polygon
    for _ in range(5):
        p_jitter = p_reg + rng.normal(0, 0.1, (N, 2))
        inits.append(p_jitter)

    # Evaluate initial scores
    for p in inits:
        s = compute_score(p)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            
    # Sort inits by score
    scored_inits = [(compute_score(p), p) for p in inits]
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # Try BasinHopping from the best few initial points
    # BasinHopping is good for escaping local minima
    for init_score, init_points in scored_inits[:3]: # Top 3 inits
        if time.time() - start_time > deadline:
            break
            
        x0 = init_points.flatten()
        
        try:
            # BasinHopping with Nelder-Mead local optimizer
            # min_steps: minimum number of steps between local minimizations
            # T: temperature
            # stepsize: step size for the perturbation
            res = basinhopping(
                neg_score, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10},
                minsteps=5,
                T=1.0,
                stepsize=0.5,
                interval=10,
                disp=False
            )
            
            new_points = res.x.reshape((N, 2))
            new_score = compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        except Exception:
            pass
            
        # Also try standard Powell on the best found so far if time permits
        if time.time() - start_time > deadline:
            break
        
        # Use the best point found so far (either initial or basin hopping result) as start for Powell
        x_powell_start = best_points.flatten()
        try:
            res2 = minimize(neg_score, x_powell_start, method='Powell', 
                            options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-10})
            new_points = res2.x.reshape((N, 2))
            new_score = compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        except Exception:
            pass

    # Final polish: Tight Nelder-Mead
    if best_points is not None and time.time() - start_time < deadline - 5:
        x_best = best_points.flatten()
        try:
            res_final = minimize(neg_score, x_best, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            new_points = res_final.x.reshape((N, 2))
            new_score = compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        except Exception:
            pass

    return best_points
