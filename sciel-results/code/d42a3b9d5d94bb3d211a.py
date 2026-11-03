import numpy as np
import time
import sys

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2

# Precompute triples (i, j, k)
_triples = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_triples = np.array(_triples)
_N_TRIPLES = len(_triples)

# For fast updates, we need to know which triples involve a specific point k.
# triples_containing[k] is a list of indices in _triples that include k.
_triples_containing = [[] for _ in range(N)]
for idx, (i, j, k) in enumerate(_triples):
    _triples_containing[i].append(idx)
    _triples_containing[j].append(idx)
    _triples_containing[k].append(idx)

def _bary_to_cart(bary_pts):
    """
    bary_pts: (N, 2) array of (u, v). w = 1 - u - v.
    Returns (N, 2) Cartesian points.
    Vertices: A(0,0), B(1,0), C(0.5, H)
    P = u*A + v*B + w*C = u*(0,0) + v*(1,0) + w*(0.5, H)
    x = v + 0.5*w = v + 0.5*(1-u-v) = 0.5 + 0.5v - 0.5u
    y = H*w = H*(1-u-v)
    """
    u = bary_pts[:, 0]
    v = bary_pts[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = H * w
    return np.column_stack([x, y])

def _calc_areas(bary_pts):
    """
    Calculate areas for all 165 triples.
    Area = 0.5 * |det| * Area_Total (Area_Total is constant, ignore for maximization)
    Det of [[u_i, v_i, w_i], [u_j, v_j, w_j], [u_k, v_k, w_k]]
    """
    u = bary_pts[:, 0]
    v = bary_pts[:, 1]
    w = 1.0 - u - v
    
    ui = u[_triples[:, 0]]
    vi = v[_triples[:, 0]]
    wi = 1.0 - ui - vi
    
    uj = u[_triples[:, 1]]
    vj = v[_triples[:, 1]]
    wj = 1.0 - uj - vj
    
    uk = u[_triples[:, 2]]
    vk = v[_triples[:, 2]]
    wk = 1.0 - uk - vk
    
    # det = u_i(v_j w_k - v_j w_k) - v_i(u_j w_k - w_j u_k) + w_i(u_j v_k - v_j u_k)
    # This is standard expansion.
    det = (ui * (vj * wk - vj * wj) - 
           vi * (uj * wk - wj * uk) + 
           wi * (uj * vk - vj * uk))
           
    # Wait, vj*wj is wrong. v_j * w_k.
    # Let's re-verify determinant expansion for rows r1, r2, r3:
    # | a b c |
    # | d e f |
    # | g h i |
    # = a(ei - fh) - b(di - fg) + c(dh - eg)
    # Here:
    # r1: ui, vi, wi
    # r2: uj, vj, wj
    # r3: uk, vk, wk
    
    # term1 = ui * (vj * wk - vj * wk? No, vj * wk - wj * vk)
    # term2 = vi * (uj * wk - wj * uk)
    # term3 = wi * (uj * vk - vj * uk)
    
    det = (ui * (vj * wk - wj * vk) - 
           vi * (uj * wk - wj * uk) + 
           wi * (uj * vk - vj * uk))
           
    areas = 0.5 * np.abs(det)
    return areas

def _update_areas_for_point(bary_pts, k, current_areas):
    """
    Update areas for triples involving point k.
    Returns new areas array (copy) and the index of min area.
    """
    new_areas = current_areas.copy()
    relevant_triple_indices = _triples_containing[k]
    
    if not relevant_triple_indices:
        return new_areas, np.argmin(new_areas)
        
    # Extract bary coords for all points involved in these triples
    # We need u, v, w for points i, j, k for each triple.
    # Since k is fixed, we can pre-fetch its coords.
    uk = bary_pts[k, 0]
    vk = bary_pts[k, 1]
    wk = 1.0 - uk - vk
    
    # Get the other two points for each relevant triple
    # triple is (i, j, k). We need to know which position k is in.
    # To avoid branching, we can just recompute the determinant for these few triples.
    
    # indices in _triples array
    idxs = relevant_triple_indices
    ti = _triples[idxs, 0]
    tj = _triples[idxs, 1]
    tk = _triples[idxs, 2]
    
    # We know one of ti, tj, tk is k.
    # Let's just compute determinants using full vectorization on these small subsets.
    u_all = bary_pts[:, 0]
    v_all = bary_pts[:, 1]
    w_all = 1.0 - u_all - v_all
    
    u1 = u_all[ti]
    v1 = v_all[ti]
    w1 = w_all[ti]
    
    u2 = u_all[tj]
    v2 = v_all[tj]
    w2 = w_all[tj]
    
    u3 = u_all[tk]
    v3 = v_all[tk]
    w3 = w_all[tk]
    
    det = (u1 * (v2 * w3 - w2 * v3) - 
           v1 * (u2 * w3 - w2 * u3) + 
           w1 * (u2 * v3 - v2 * u3))
           
    new_areas[idxs] = 0.5 * np.abs(det)
    
    min_idx = np.argmin(new_areas)
    return new_areas, min_idx

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_min_area = 0.0
    best_bary_pts = None
    
    rng = np.random.default_rng(123)
    
    # Helper to check if bary points are valid
    def is_valid_bary(pts):
        u = pts[:, 0]
        v = pts[:, 1]
        w = 1.0 - u - v
        return np.all(u >= -1e-9) and np.all(v >= -1e-9) and np.all(w >= -1e-9)
    
    # Simulated Annealing / Local Search parameters
    # We will run multiple restarts.
    # Each restart runs SA for a fixed number of iterations or time budget.
    
    max_restarts = 50 # Upper bound, time is the real limit
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize random points
        # Use Dirichlet to generate valid barycentric coords
        # Shape (N, 3)
        bary3 = rng.dirichlet([1, 1, 1], size=N)
        bary2 = bary3[:, :2] # u, v
        
        current_areas = _calc_areas(bary2)
        current_min_area = np.min(current_areas)
        
        if current_min_area > best_min_area:
            best_min_area = current_min_area
            best_bary_pts = bary2.copy()
            
        # Run Local Search (SA)
        # Temperature
        T = 0.01
        T_min = 1e-6
        cooling = 0.995
        
        # Adaptive step size
        step_size = 0.05
        
        # Pre-allocate for speed?
        # The loop is tight.
        
        iters = 0
        max_iters = 100000 # Cap iterations per restart to prevent infinite loop if time check fails
        
        while time.time() < deadline and iters < max_iters:
            iters += 1
            
            # Pick random point
            k = rng.integers(0, N)
            
            # Generate perturbation
            # Simple Gaussian noise
            du = rng.normal(0, step_size)
            dv = rng.normal(0, step_size)
            
            new_u = bary2[k, 0] + du
            new_v = bary2[k, 1] + dv
            
            # Check validity
            new_w = 1.0 - new_u - new_v
            if new_u < 0 or new_v < 0 or new_w < 0:
                # Option 1: Reject
                # Option 2: Project (complex)
                # Let's reject for simplicity, or clamp
                # Clamping can lead to stuck points. Rejecting is safer for SA.
                continue
            
            # Update point
            old_u = bary2[k, 0]
            old_v = bary2[k, 1]
            
            bary2[k, 0] = new_u
            bary2[k, 1] = new_v
            
            # Update areas
            new_areas, min_idx = _update_areas_for_point(bary2, k, current_areas)
            new_min_area = new_areas[min_idx]
            
            # Check improvement
            delta = new_min_area - current_min_area
            
            if delta >= 0:
                # Accept
                current_areas = new_areas
                current_min_area = new_min_area
                
                if current_min_area > best_min_area:
                    best_min_area = current_min_area
                    best_bary_pts = bary2.copy()
                    
                # Slightly increase step size if improving? Or keep same?
                # Keep same for now.
            else:
                # SA Acceptance
                if T > T_min:
                    if np.random.random() < np.exp(delta / T):
                        # Accept
                        current_areas = new_areas
                        current_min_area = new_min_area
                    else:
                        # Reject, revert
                        bary2[k, 0] = old_u
                        bary2[k, 1] = old_v
                else:
                    # Reject, revert
                    bary2[k, 0] = old_u
                    bary2[k, 1] = old_v
            
            T *= cooling
            
            # Adaptive step size: if many rejections, decrease step?
            # Simple heuristic: every 1000 iters, halve step size if no recent improvement?
            # For simplicity, we'll just use fixed cooling.
            
            if iters % 1000 == 0:
                step_size *= 0.95 # Slowly shrink step size to allow convergence
                
    if best_bary_pts is None:
        # Fallback
        bary3 = np.ones((N, 3)) / 3.0
        best_bary_pts = bary3[:, :2]
        
    # Convert to Cartesian
    cart_pts = _bary_to_cart(best_bary_pts)
    
    # Ensure strict validity (clip tiny negatives)
    # The evaluator might check strict inequalities.
    # Our barycentric generation and rejection should handle this,
    # but let's do a final safety check and projection if needed.
    # Actually, if we rejected any negative bary, we are safe.
    
    return cart_pts
