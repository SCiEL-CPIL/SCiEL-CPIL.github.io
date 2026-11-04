import numpy as np
import time
from scipy.optimize import minimize

N = 12

def get_hex_verts(cx, cy, side, angle_deg):
    """
    Returns vertices of a regular hexagon.
    side: length of side
    angle_deg: rotation in degrees
    """
    angle_rad = np.deg2rad(angle_deg)
    # Vertices of unit hexagon (side 1) centered at origin, rotated by angle
    # Standard hexagon vertices: (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Wait, side length 1 means distance from center to vertex is 1?
    # For regular hexagon, side length = radius. Yes.
    angles = angle_rad + np.arange(6) * np.pi / 3
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def get_hex_normals(cx, cy, side, angle_deg):
    """
    Returns outward normals of the edges.
    """
    angle_rad = np.deg2rad(angle_deg)
    # Normals are at angles offset by 30 degrees from vertices?
    # Vertex 0 is at angle 0. Edge 0 is between V0 and V1.
    # Midpoint of V0 and V1 is at angle 30 deg. Normal is at 30 deg.
    # So normals are at angle_rad + np.pi/6 + np.arange(6) * np.pi / 3
    angles = angle_rad + np.pi / 6 + np.arange(6) * np.pi / 3
    ns = np.column_stack((np.cos(angles), np.sin(angles)))
    return ns

def min_gap(p1_verts, p2_verts, p1_normals, p2_normals):
    """
    Compute minimum separation distance between two convex polygons.
    Returns the maximum gap over all separating axes.
    If gap > 0, they are separated by gap.
    If gap < 0, they overlap by -gap.
    """
    all_normals = np.vstack((p1_normals, p2_normals))
    max_gap = -np.inf
    for n in all_normals:
        # Project p1 and p2 onto n
        proj1 = p1_verts @ n
        proj2 = p2_verts @ n
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Gap in this direction
        if max1 <= min2:
            gap = min2 - max1
        elif max2 <= min1:
            gap = min1 - max2
        else:
            gap = 0.0 # Overlapping in this projection
            
        if gap > max_gap:
            max_gap = gap
            
    return max_gap

def penalty_and_grad(params, S):
    """
    params: array of size 36: [x1, y1, th1, x2, y2, th2, ...]
    S: side length of outer hexagon
    Returns: penalty value, gradient (optional, let's compute numerically or analytically)
    """
    # Unpack
    xs = params[0::3]
    ys = params[1::3]
    ths = params[2::3]
    
    # Outer hexagon properties
    # Centered at (0,0), angle 0
    outer_apothem = S * np.sqrt(3) / 2.0
    outer_normals = get_hex_normals(0, 0, 1, 0) # Side length doesn't matter for normals direction
    
    verts_list = []
    normals_list = []
    
    for i in range(N):
        v = get_hex_verts(xs[i], ys[i], 1.0, ths[i])
        n = get_hex_normals(xs[i], ys[i], 1.0, ths[i])
        verts_list.append(v)
        normals_list.append(n)
        
    penalty = 0.0
    grad = np.zeros_like(params)
    
    # 1. Containment constraints
    for i in range(N):
        v = verts_list[i]
        # For each outer normal, check max projection
        for k in range(6):
            n_out = outer_normals[k]
            proj = v @ n_out
            max_proj = np.max(proj)
            violation = max_proj - outer_apothem
            if violation > 0:
                penalty += violation ** 2
                # Gradient w.r.t. xs[i], ys[i]
                # d(max_proj)/d(xs[i]) is the x-component of the normal of the vertex that gave max_proj
                # We need to know which vertex.
                idx_max = np.argmax(proj)
                vx, vy = v[idx_max]
                # The normal to the edge of the inner hex that contains this vertex?
                # No, the gradient of the projection of a vertex on a fixed axis n_out is just n_out.
                # Wait, the vertex position depends on xs[i], ys[i], ths[i].
                # v_j = (xs[i], ys[i]) + R(theta_i) * v0_j
                # d v_j / d xs[i] = (1, 0)
                # d v_j / d ys[i] = (0, 1)
                # d v_j / d ths[i] = (-vy_rel, vx_rel) where v_rel is relative to center
                # proj = v_j . n_out
                # d proj / d xs[i] = n_out[0]
                # d proj / d ys[i] = n_out[1]
                # d proj / d ths[i] = (d v_j / d ths[i]) . n_out
                
                # We only need gradient for the active vertex (idx_max)
                # Relative vertex
                rel_v = v[idx_max] - np.array([xs[i], ys[i]])
                # Derivative of rel_v wrt theta
                d_rel_v_d_th = np.array([-rel_v[1], rel_v[0]])
                
                grad[3*i + 0] += 2 * violation * n_out[0]
                grad[3*i + 1] += 2 * violation * n_out[1]
                grad[3*i + 2] += 2 * violation * (d_rel_v_d_th @ n_out)
                
    # 2. Overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            gap = min_gap(verts_list[i], verts_list[j], normals_list[i], normals_list[j])
            violation = -gap # If gap < 0, violation > 0
            if violation > 0:
                penalty += violation ** 2
                # Gradient is tricky because we need the active axis
                # Re-compute to find active axis
                all_normals = np.vstack((normals_list[i], normals_list[j]))
                active_idx = 0
                max_gap_found = -np.inf
                for idx, n in enumerate(all_normals):
                    proj_i = verts_list[i] @ n
                    proj_j = verts_list[j] @ n
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    
                    if max_i <= min_j:
                        g = min_j - max_i
                    elif max_j <= min_i:
                        g = min_i - max_j
                    else:
                        g = 0.0
                    if g > max_gap_found:
                        max_gap_found = g
                        active_idx = idx
                        
                n_act = all_normals[active_idx]
                # Determine which vertices are active for max_i and min_j etc.
                # This is getting complicated.
                # For simplicity and robustness, let's use numerical gradient or just L-BFGS-B without gradient?
                # L-BFGS-B without gradient uses finite differences. With 36 vars, it's 36 function calls per step.
                # Function call is ~1000 ops. 36000 ops per step. 100 steps = 3.6M ops. Very fast.
                # Let's return None for gradient to let SciPy do finite differences.
                
    return penalty, None

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    best_S = 40.0 # Fallback
    best_params = None
    
    # Initial guess: Grid
    # Hex grid spacing: dx = sqrt(3), dy = 1.5 * sqrt(3)
    dx = np.sqrt(3)
    dy = 1.5 * np.sqrt(3)
    
    # Generate a few initial configurations
    configs = []
    
    # Config 1: Dense grid
    pts = []
    for r in range(-2, 3):
        for c in range(-2, 3):
            x = c * dx + (r % 2) * dx / 2
            y = r * dy
            pts.append((x, y, 0.0))
            if len(pts) >= N:
                break
        if len(pts) >= N:
            break
    if len(pts) < N:
        # Pad if necessary
        while len(pts) < N:
            pts.append((len(pts)*0.1, len(pts)*0.1, 0.0))
    params1 = np.array([item for p in pts[:N] for item in p])
    configs.append(params1)
    
    # Config 2: Random
    np.random.seed(42)
    params2 = np.random.uniform(-2, 2, size=36)
    configs.append(params2)
    
    # Config 3: Circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    params3 = np.zeros(36)
    for i in range(N):
        params3[3*i] = 1.5 * np.cos(angles[i])
        params3[3*i+1] = 1.5 * np.sin(angles[i])
        params3[3*i+2] = 0.0
    configs.append(params3)
    
    # Try to optimize for decreasing S
    current_S = 4.5
    # We know 7 hexes fit in S=2. 12 hexes should fit in S < 3.5
    # Let's start from a safe S and go down
    
    for init_params in configs:
        if time.time() > deadline:
            break
            
        S = 4.0
        params = init_params.copy()
        
        # Optimize to minimize penalty at fixed S
        # We want to find a valid packing for this S.
        # If we find one, we try smaller S.
        
        # First, ensure we have a valid packing at S=4.0
        res = minimize(penalty_and_grad, params, args=(S,), method='L-BFGS-B', jac=True, 
                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun < 1e-6:
            # Valid packing found at S
            best_S = S
            best_params = res.x
            
            # Try to reduce S
            while time.time() < deadline - 5:
                new_S = S - 0.05
                if new_S < 2.0: break # Physical limit approx
                
                res2 = minimize(penalty_and_grad, res.x, args=(new_S,), method='L-BFGS-B', jac=True,
                                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res2.fun < 1e-6:
                    S = new_S
                    best_S = S
                    best_params = res2.x
                    res = res2
                else:
                    # Failed to reduce, try smaller step?
                    # Just break to save time, or try a slightly larger reduction
                    break
        else:
            # Initial S too small or optimization failed
            # Try increasing S slightly? No, 4.0 is large enough for 12 hexes.
            # Maybe restart with different params?
            pass

    # Final check: if we didn't find a good one, return seed
    if best_S >= 40.0:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        th = best_params[3*i+2]
        inner.append([x, y, th])
        
    return inner, (0.0, 0.0), best_S, 0.0
