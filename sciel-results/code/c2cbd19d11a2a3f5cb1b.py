import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
TIME_LIMIT = 115.0

# Triangle vertices in Barycentric coordinates
# V0: (1, 0, 0)
# V1: (0, 1, 0)
# V2: (0, 0, 1)
# Note: The standard barycentric coordinates for vertices are unit vectors.
# Area of triangle with barycentric coords P, Q, R is |det([P, Q, R])| * TotalArea.
# We maximize min |det|.

def get_bary_from_params(params, sym_type):
    """
    Reconstruct 11 barycentric points from symmetric parameters.
    sym_type: 0 for (1 axis, 5 pairs), 1 for (3 axis, 4 pairs), 2 for (5 axis, 3 pairs)
    """
    points = np.zeros((N, 3))
    
    if sym_type == 0:
        # 1 axis point, 5 pairs
        w0 = params[0]
        u0 = (1 - w0) / 2
        v0 = u0
        points[0] = [u0, v0, w0]
        
        idx = 1
        for j in range(5):
            u = params[1 + 2*j]
            v = params[2 + 2*j]
            w = 1 - u - v
            # Clamp to ensure validity if optimization goes slightly out of bounds
            u = max(0, min(1, u))
            v = max(0, min(1, v))
            w = max(0, min(1, w))
            s = u + v + w
            if s > 0:
                u /= s; v /= s; w /= s
            
            points[idx] = [u, v, w]
            points[idx+1] = [v, u, w]
            idx += 2
            
    elif sym_type == 1:
        # 3 axis points, 4 pairs
        for i in range(3):
            w = params[i]
            u = (1 - w) / 2
            v = u
            points[i] = [u, v, w]
            
        idx = 3
        for j in range(4):
            u = params[3 + 2*j]
            v = params[4 + 2*j]
            w = 1 - u - v
            u = max(0, min(1, u))
            v = max(0, min(1, v))
            w = max(0, min(1, w))
            s = u + v + w
            if s > 0:
                u /= s; v /= s; w /= s
                
            points[idx] = [u, v, w]
            points[idx+1] = [v, u, w]
            idx += 2
            
    elif sym_type == 2:
        # 5 axis points, 3 pairs
        for i in range(5):
            w = params[i]
            u = (1 - w) / 2
            v = u
            points[i] = [u, v, w]
            
        idx = 5
        for j in range(3):
            u = params[5 + 2*j]
            v = params[6 + 2*j]
            w = 1 - u - v
            u = max(0, min(1, u))
            v = max(0, min(1, v))
            w = max(0, min(1, w))
            s = u + v + w
            if s > 0:
                u /= s; v /= s; w /= s
                
            points[idx] = [u, v, w]
            points[idx+1] = [v, u, w]
            idx += 2
            
    return points

def calc_min_area_bary(bary_points):
    """
    Calculate the minimum area (relative to total triangle area) of any triangle 
    formed by 3 of the given barycentric points.
    """
    n = bary_points.shape[0]
    min_area = np.inf
    # Iterate over all combinations of 3 points
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = bary_points[i]
                p2 = bary_points[j]
                p3 = bary_points[k]
                det = np.linalg.det(np.array([p1, p2, p3]))
                area = abs(det)
                if area < min_area:
                    min_area = area
    return min_area

def objective_sym(params, sym_type):
    """
    Objective function for symmetric parameter optimization.
    Returns negative min area to be minimized.
    """
    bary = get_bary_from_params(params, sym_type)
    min_area = calc_min_area_bary(bary)
    return -min_area

def bary_to_cart(bary):
    """Convert barycentric (u, v, w) to Cartesian (x, y) for triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = bary[:, 2]
    # x = v * 1 + w * 0.5 + u * 0
    # y = v * 0 + w * sqrt(3)/2 + u * 0
    x = v + 0.5 * w
    y = w * (np.sqrt(3) / 2)
    return np.column_stack([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_score = -np.inf
    best_bary = None
    
    # Define bounds for the 11 parameters
    # Axis points: w in [0, 1]
    # Pair points: u in [0, 1], v in [0, 1-u] approx, but we clamp inside.
    # For DE, simple bounds [0,1] for all params work, with clamping in reconstruction.
    bounds = [(0, 1)] * 11
    
    sym_types = [0, 1, 2]
    
    for st in sym_types:
        if time.time() > deadline - 10:
            break
            
        try:
            # Use Differential Evolution on the reduced 11D space
            # This is fast and robust for small dimensions
            result = differential_evolution(
                objective_sym,
                bounds,
                args=(st,),
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                polish=False # We will do a manual polish
            )
            
            best_params = result.x
            score = -result.fun
            
            # Local refinement in the reduced space using Nelder-Mead
            res_nm = minimize(
                objective_sym,
                best_params,
                args=(st,),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_nm.fun < result.fun:
                best_params = res_nm.x
                score = -res_nm.fun
            
            if score > best_score:
                best_score = score
                best_bary = get_bary_from_params(best_params, st)
                
        except Exception as e:
            pass
            
    # Final polish in full 22D space to break symmetry if beneficial
    if best_bary is not None and time.time() < deadline - 5:
        # Convert best symmetric bary to a flat array of 22 coords (u, v) for 11 points
        # Note: get_bary_from_params returns (N,3). We need to pass (N,2) to a full optimizer?
        # The seed program used (N,2) barycentric (u,v) implicitly w=1-u-v.
        # Let's stick to the (N,2) representation for the final polish to be consistent with potential non-symmetric improvements.
        
        # Convert (N,3) bary to (N,2) (u,v)
        u = best_bary[:, 0]
        v = best_bary[:, 1]
        init_flat = np.column_stack([u, v]).flatten()
        
        def full_obj(flat):
            u = flat[0::2]
            v = flat[1::2]
            w = 1 - u - v
            # Clamp
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
            w = np.clip(w, 0, 1)
            s = u + v + w
            mask = s > 0
            u[mask] /= s[mask]
            v[mask] /= s[mask]
            w[mask] /= s[mask]
            
            bary = np.column_stack([u, v, w])
            min_area = calc_min_area_bary(bary)
            return -min_area

        try:
            res_full = minimize(
                full_obj,
                init_flat,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            u = res_full.x[0::2]
            v = res_full.x[1::2]
            w = 1 - u - v
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
            w = np.clip(w, 0, 1)
            s = u + v + w
            mask = s > 0
            u[mask] /= s[mask]
            v[mask] /= s[mask]
            w[mask] /= s[mask]
            
            final_bary = np.column_stack([u, v, w])
            final_score = calc_min_area_bary(final_bary)
            
            if final_score > best_score:
                best_score = final_score
                best_bary = final_bary
                
        except Exception:
            pass

    # Convert best barycentric to Cartesian
    cart = bary_to_cart(best_bary)
    return cart

# EVOLVE-BLOCK-END
