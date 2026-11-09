"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12
DEADLINE = 150.0

def _hex_vertices(cx, cy, s, ang_deg):
    ang = math.radians(ang_deg)
    verts = []
    for k in range(6):
        theta = ang + k * math.pi / 3.0
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _sat_overlap(v1, v2):
    """
    Check if two convex polygons (lists of vertices) overlap using Separating Axis Theorem.
    v1, v2 are Nx2 arrays.
    Returns True if they overlap, False otherwise.
    """
    # Get normals for v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        # Edge vector
        ex = p2[0] - p1[0]
        ey = p2[1] - p1[1]
        # Normal (perpendicular)
        nx = -ey
        ny = ex
        
        # Project v1 onto normal
        min1 = np.dot(p1, [nx, ny])
        max1 = min1
        for p in v1:
            proj = np.dot(p, [nx, ny])
            if proj < min1: min1 = proj
            if proj > max1: max1 = proj
            
        # Project v2 onto normal
        min2 = np.dot(v2[0], [nx, ny])
        max2 = min2
        for p in v2:
            proj = np.dot(p, [nx, ny])
            if proj < min2: min2 = proj
            if proj > max2: max2 = proj
            
        # Check for separation
        if max1 < min2 or max2 < min1:
            return False
            
    # If we get here, no separating axis found from v1's normals.
    # We must also check v2's normals.
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        ex = p2[0] - p1[0]
        ey = p2[1] - p1[1]
        nx = -ey
        ny = ex
        
        min1 = np.dot(v1[0], [nx, ny])
        max1 = min1
        for p in v1:
            proj = np.dot(p, [nx, ny])
            if proj < min1: min1 = proj
            if proj > max1: max1 = proj
            
        min2 = np.dot(p1, [nx, ny])
        max2 = min2
        for p in v2:
            proj = np.dot(p, [nx, ny])
            if proj < min2: min2 = proj
            if proj > max2: max2 = proj
            
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def _boundary_violation(verts_list, S):
    """
    Calculate sum of squared distances of vertices outside the outer hexagon.
    Outer hexagon is centered at (0,0), side S, rotation 0.
    """
    apothem = S * math.sqrt(3) / 2.0
    normals = np.array([
        [0.0, 1.0],
        [math.sqrt(3)/2, 0.5],
        [math.sqrt(3)/2, -0.5],
        [0.0, -1.0],
        [-math.sqrt(3)/2, -0.5],
        [-math.sqrt(3)/2, 0.5]
    ])
    
    total_violation = 0.0
    for verts in verts_list:
        for v in verts:
            # Check projection against each normal
            for n in normals:
                proj = np.dot(v, n)
                if proj > apothem:
                    total_violation += (proj - apothem)**2
    return total_violation

def _objective(params, S):
    """
    Objective function for optimizer.
    params: array of shape (N, 3) -> [x, y, angle_deg]
    Returns penalty. 0 if valid.
    """
    p = params.reshape(N, 3)
    xs = p[:, 0]
    ys = p[:, 1]
    angles = p[:, 2]
    
    verts_list = []
    for i in range(N):
        v = _hex_vertices(xs[i], ys[i], 1.0, angles[i])
        verts_list.append(v)
        
    # Boundary penalty
    pen = _boundary_violation(verts_list, S)
    
    # Overlap penalty
    overlap_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(verts_list[i], verts_list[j]):
                overlap_count += 1
                
    pen += 10.0 * overlap_count
    
    return pen

def _honeycomb_initial(S):
    """
    Generate a honeycomb-like initial configuration.
    """
    # Hex lattice vectors
    v1 = np.array([1.5, math.sqrt(3)/2])
    v2 = np.array([0.0, math.sqrt(3)])
    
    positions = []
    # 3 rows of 4
    for i in range(3):
        for j in range(4):
            pos = i * v1 + j * v2
            positions.append(pos)
            
    positions = np.array(positions)
    # Center it
    cx = positions[:, 0].mean()
    cy = positions[:, 1].mean()
    positions[:, 0] -= cx
    positions[:, 1] -= cy
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    positions += rng.normal(0, 0.1, positions.shape)
    
    params = np.zeros((N, 3))
    params[:, 0] = positions[:, 0]
    params[:, 1] = positions[:, 1]
    params[:, 2] = 0.0
    
    return params

def _try_pack(S, time_limit=2.0, restarts=5):
    """
    Try to find a valid packing for side length S.
    Returns True if found, False otherwise.
    """
    start = time.time()
    bounds = [(-S, S), (-S, S), (-180, 180)] * N
    
    # Multiple restarts
    for restart in range(restarts):
        if time.time() - start > time_limit:
            break
            
        x0 = _honeycomb_initial(S)
        if restart > 0:
            # Perturb initial guess
            rng = np.random.default_rng(restart)
            x0 += rng.normal(0, 0.5, x0.shape)
            # Keep angles small initially
            x0[:, 2] = rng.uniform(-10, 10, N)
            
        try:
            res = minimize(
                _objective, 
                x0, 
                args=(S,), 
                method='Powell', 
                bounds=bounds,
                options={'maxiter': 100, 'maxfev': 500}
            )
            
            if res.fun < 1e-6:
                # Verify validity strictly
                p = res.x.reshape(N, 3)
                verts_list = []
                for i in range(N):
                    v = _hex_vertices(p[i,0], p[i,1], 1.0, p[i,2])
                    verts_list.append(v)
                
                valid = True
                # Check boundary
                if _boundary_violation(verts_list, S) > 1e-4:
                    valid = False
                # Check overlap
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            if _sat_overlap(verts_list[i], verts_list[j]):
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    return True
                    
        except Exception:
            pass
            
    return False

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0 # 5s safety margin
    
    # Binary search for minimum S
    low = 3.46
    high = 6.0
    
    # First, ensure high is valid
    if not _try_pack(high, time_limit=5.0, restarts=10):
        # If even high is not valid, increase high
        high = 8.0
        if not _try_pack(high, time_limit=5.0, restarts=10):
            # Fallback to a very sparse configuration
            best_params = np.zeros((N, 3))
            for i in range(N):
                best_params[i, 0] = i * 2.0
                best_params[i, 1] = 0.0
            inner = [[float(x), float(y), float(a)] for x, y, a in best_params]
            return inner, (0.0, 0.0), 10.0, 0.0

    # Binary search
    for i in range(30):
        if time.time() > deadline - 10.0:
            break
            
        mid = (low + high) / 2.0
        
        # Time limit for this check
        remaining = deadline - time.time()
        time_limit = max(0.5, remaining / 10.0) # Allow 10 more iterations
        
        if _try_pack(mid, time_limit=time_limit, restarts=5):
            high = mid
        else:
            low = mid
            
    # Final optimization for the best S found
    final_S = high
    
    # Run a more thorough optimization for final_S to get the actual coordinates
    start = time.time()
    bounds = [(-final_S, final_S), (-final_S, final_S), (-180, 180)] * N
    best_res = None
    best_pen = float('inf')
    
    for restart in range(10):
        if time.time() > deadline - 2.0:
            break
            
        x0 = _honeycomb_initial(final_S)
        if restart > 0:
            rng = np.random.default_rng(restart)
            x0 += rng.normal(0, 0.2, x0.shape)
            x0[:, 2] = rng.uniform(-5, 5, N)
            
        try:
            res = minimize(
                _objective, 
                x0, 
                args=(final_S,), 
                method='Powell', 
                bounds=bounds,
                options={'maxiter': 200, 'maxfev': 1000}
            )
            
            if res.fun < best_pen:
                best_pen = res.fun
                best_res = res
                
                # Check if valid
                if res.fun < 1e-6:
                    p = res.x.reshape(N, 3)
                    verts_list = []
                    for i in range(N):
                        v = _hex_vertices(p[i,0], p[i,1], 1.0, p[i,2])
                        verts_list.append(v)
                    
                    valid = True
                    if _boundary_violation(verts_list, final_S) > 1e-4:
                        valid = False
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                if _sat_overlap(verts_list[i], verts_list[j]):
                                    valid = False
                                    break
                            if not valid: break
                    if valid:
                        break # Found a valid one, stop searching
                        
        except Exception:
            pass
            
    if best_res is not None:
        p = best_res.x.reshape(N, 3)
        inner = [[float(x), float(y), float(a)] for x, y, a in p]
        return inner, (0.0, 0.0), float(final_S), 0.0
    else:
        # Fallback
        p = _honeycomb_initial(final_S)
        inner = [[float(x), float(y), float(a)] for x, y, a in p]
        return inner, (0.0, 0.0), float(final_S), 0.0

# EVOLVE-BLOCK-END
