import math
import time
import numpy as np
from scipy.optimize import linprog, differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side(verts, outer_angle_deg):
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    vdots = verts @ normals.T
    
    A_rows = []
    b_vals = []
    
    for j in range(6):
        nx, ny = normals[j]
        A_rows.append(np.tile([-nx, -ny, -1.0], (M, 1)))
        b_vals.append(-vdots[:, j])
        A_rows.append(np.tile([nx, ny, -1.0], (M, 1)))
        b_vals.append(vdots[:, j])
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    A = np.vstack([A, np.array([[0.0, 0.0, -1.0]])])
    b = np.concatenate([b, [0.0]])
    
    c_obj = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs', options={'presolve': True, 'dual_feasibility_tolerance': 1e-12, 'primal_feasibility_tolerance': 1e-12})
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return cx, cy, side
    except Exception:
        pass
    return None

def _check_overlap(verts_list, eps=1e-9):
    """
    Check if any two hexagons in verts_list overlap.
    verts_list is a list of numpy arrays of shape (6,2).
    Returns True if overlap is detected.
    """
    n = len(verts_list)
    for i in range(n):
        # Bounding box quick check
        min_x_i, min_y_i = np.min(verts_list[i], axis=0)
        max_x_i, max_y_i = np.max(verts_list[i], axis=0)
        
        for j in range(i + 1, n):
            min_x_j, min_y_j = np.min(verts_list[j], axis=0)
            max_x_j, max_y_j = np.max(verts_list[j], axis=0)
            
            # Check bounding box overlap
            if max_x_i < min_x_j - eps or max_x_j < min_x_i - eps or \
               max_y_i < min_y_j - eps or max_y_j < min_y_i - eps:
                continue
            
            # Precise check: Point in Polygon or Edge Intersection
            # We check if any vertex of i is inside j, or any vertex of j is inside i
            # Or if any edges intersect.
            
            # Simple point in polygon (winding number or ray casting) is robust enough for convex
            def point_in_hex(px, py, hex_verts):
                # Ray casting
                inside = False
                n_v = len(hex_verts)
                j = n_v - 1
                for i_v in range(n_v):
                    xi, yi = hex_verts[i_v]
                    xj, yj = hex_verts[j]
                    if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                        inside = not inside
                    j = i_v
                return inside

            overlap = False
            # Check vertices of i in j
            for v in verts_list[i]:
                if point_in_hex(v[0], v[1], verts_list[j]):
                    overlap = True
                    break
            if not overlap:
                # Check vertices of j in i
                for v in verts_list[j]:
                    if point_in_hex(v[0], v[1], verts_list[i]):
                        overlap = True
                        break
            
            if overlap:
                return True
    return False

def _objective(params):
    # params: x1, y1, th1, ..., x11, y11, th11, xo, yo, tho
    # 11 * 3 + 3 = 36 parameters
    if time.time() > _DEADLINE - 5.0:
        return 1e10
        
    inner_params = params[:33]
    outer_params = params[33:]
    xo, yo, tho = outer_params
    
    verts_list = []
    for i in range(N):
        x, y, th = inner_params[i*3], inner_params[i*3+1], inner_params[i*3+2]
        verts = _hex_vertices(x, y, th)
        verts_list.append(verts)
    
    # Check overlap
    if _check_overlap(verts_list):
        return 1000.0
        
    # Calculate minimal outer side
    all_vertices = np.vstack(verts_list)
    res = _min_enclosing_hex_side(all_vertices, tho)
    if res is None:
        return 1e10
        
    return res[2]

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback: simple line
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Initial guess: a compact cluster
    # Hex grid coordinates
    sqrt3 = _SQRT3
    scale = 1.0000001
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    # 11 hexagons in a flower-like shape
    # Center + 6 neighbors + 4 more
    coords = [(0,0)]
    ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    coords.extend(ring1)
    ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
    # Pick 4 from ring2 to make 11
    # (2,0), (-2,0), (0,2), (0,-2) are far, maybe closer ones?
    # Let's just use the first 4 from ring2
    coords.extend(ring2[:4])
    
    init_params = []
    for cx, cy in coords:
        x = cx*u[0] + cy*v[0]
        y = cx*u[1] + cy*v[1]
        init_params.extend([x, y, 0.0])
    
    # Outer center and angle
    init_params.extend([0.0, 0.0, 0.0])
    
    # Bounds for differential evolution
    # Positions roughly in [-5, 5]
    # Angles in [-180, 180]
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((-180.0, 180.0))
    bounds.append((-5.0, 5.0)) # xo
    bounds.append((-5.0, 5.0)) # yo
    bounds.append((-180.0, 180.0)) # tho
    
    # Run differential evolution
    # We want to minimize the side length
    # DE is stochastic and slow, so we limit maxiter and popsize
    # We use a time limit via the objective function check
    
    best_result = None
    best_side = float('inf')
    
    # Try a few runs with different seeds or just one long run
    # DE with small population might be faster
    try:
        res_de = differential_evolution(_objective, bounds, 
                                        seed=42, 
                                        maxiter=50, 
                                        popsize=10,
                                        tol=1e-6,
                                        init='latinhypercube')
        if res_de.success and res_de.fun < best_side:
            best_side = res_de.fun
            best_result = res_de.x
    except Exception:
        pass
    
    # If DE didn't find anything good or didn't run, try local optimization from init
    if best_result is None or best_side > 10.0:
        # Use the initial guess
        candidate = np.array(init_params)
        res_init = _objective(candidate)
        if res_init < best_side:
            best_side = res_init
            best_result = candidate
    
    # Refine with local optimization
    if best_result is not None:
        # COBYLA is derivative-free and handles constraints (though we don't use them here, penalty handles it)
        # SLSQP requires gradients, which we don't have easily for the penalty function
        # Let's use COBYLA or Nelder-Mead
        try:
            res_local = minimize(_objective, best_result, 
                                 method='COBYLA', 
                                 options={'maxiter': 200, 'rhobeg': 0.1})
            if res_local.success and res_local.fun < best_side:
                best_side = res_local.fun
                best_result = res_local.x
        except Exception:
            pass
        
        # One more polish with Nelder-Mead
        try:
            res_nm = minimize(_objective, best_result, 
                              method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            if res_nm.success and res_nm.fun < best_side:
                best_side = res_nm.fun
                best_result = res_nm.x
        except Exception:
            pass

    # Extract final result
    if best_result is None:
        # Fallback
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        x = best_result[i*3]
        y = best_result[i*3+1]
        th = best_result[i*3+2]
        inner.append([x, y, th])
        
    xo = best_result[33]
    yo = best_result[34]
    tho = best_result[35]
    
    # Verify the side length one last time to be sure
    verts_list = []
    for x, y, th in inner:
        verts_list.append(_hex_vertices(x, y, th))
    
    if _check_overlap(verts_list):
        # Invalid result, return fallback
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    all_vertices = np.vstack(verts_list)
    res_final = _min_enclosing_hex_side(all_vertices, tho)
    if res_final is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    cx, cy, side = res_final
    return inner, (cx, cy), side, tho
