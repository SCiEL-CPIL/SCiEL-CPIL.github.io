"""Fast smooth-objective L-BFGS-B with many diverse restarts for 12 unit hexagons."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 148.0

# Precompute constants
_SQRT3 = np.sqrt(3.0)
_HALF_SQRT3 = _SQRT3 / 2.0
_60DEG = np.radians(60.0)

def _hex_verts_vec(cx, cy, angle_deg):
    """Vectorized hexagon vertices for a unit hexagon."""
    a = np.radians(angle_deg)
    thetas = a + _60DEG * np.arange(6)
    return np.column_stack([cx + np.cos(thetas), cy + np.sin(thetas)])

def _hex_normals_vec(angle_deg):
    """6 outward edge normals of a unit hexagon at given rotation."""
    a = np.radians(angle_deg)
    # Edge normals are at angles a + 30 + 60*k degrees
    thetas = a + np.radians(30) + _60DEG * np.arange(6)
    return np.column_stack([np.cos(thetas), np.sin(thetas)])

def _smooth_min_sep(c1, c2, beta=40.0):
    """
    Smooth min separation between two unit hexagons.
    c1, c2: [x, y, angle_deg]
    Returns smooth min gap (negative = overlap).
    """
    # Get normals from both hexagons
    n1 = _hex_normals_vec(c1[2])
    n2 = _hex_normals_vec(c2[2])
    axes = np.vstack([n1, n2])  # (12, 2)
    
    # Vertices
    v1 = _hex_verts_vec(c1[0], c1[1], c1[2])  # (6,2)
    v2 = _hex_verts_vec(c2[0], c2[1], c2[2])  # (6,2)
    
    # Projections: (12, 6)
    p1 = axes @ v1.T  # (12, 6)
    p2 = axes @ v2.T  # (12, 6)
    
    min1 = p1.min(axis=1)
    max1 = p1.max(axis=1)
    min2 = p2.min(axis=1)
    max2 = p2.max(axis=1)
    
    seps = np.maximum(min1 - max2, min2 - max1)
    
    # Soft min
    mn = seps.min()
    exps = np.exp(beta * (seps - mn))
    return mn - np.log(exps.sum()) / beta

def _smooth_containment(centers, outer_center, outer_angle, beta=30.0):
    """
    Smooth max extent of all inner hexagons relative to outer hexagon center.
    Returns the smooth required 'apothem' (distance from center to edge).
    """
    oa = np.radians(outer_angle)
    # Outer hexagon normals in world frame
    outer_n = _hex_normals_vec(outer_angle)  # (6,2)
    
    all_d = []
    for c in centers:
        verts = _hex_verts_vec(c[0], c[1], c[2])  # (6,2)
        rel = verts - np.array(outer_center)  # (6,2)
        # Project onto each outer normal
        d = np.abs(rel @ outer_n.T)  # (6,6)
        all_d.append(d.flatten())
    
    d_arr = np.concatenate(all_d)
    mn = d_arr.min()
    exps = np.exp(beta * (d_arr - mn))
    soft_max = mn + np.log(exps.sum()) / beta
    return soft_max

def _objective(params, beta_sep=40.0, beta_cont=30.0, lam=10.0):
    """
    Objective: smooth_required_side + lam * sum(max(0, -sep_ij)^2)
    params: [x1,y1,a1, ..., xN,yN,aN, ox,oy,oa]
    """
    n = N
    # Extract centers and angles
    cx = params[0:3*n:3]
    cy = params[1:3*n:3]
    ca = params[2:3*n:3]
    outer_center = [params[3*n], params[3*n+1]]
    outer_angle = params[3*n+2]
    
    # Pre-compute vertices and normals for all inner hexagons
    # Vertices: (N, 6, 2)
    thetas = np.radians(ca)[:, np.newaxis] + _60DEG * np.arange(6)[np.newaxis, :]
    verts = np.column_stack([
        cx[:, np.newaxis] + np.cos(thetas).flatten(),
        cy[:, np.newaxis] + np.sin(thetas).flatten()
    ]).reshape(n, 6, 2)
    
    # Normals: (N, 6, 2)
    n_thetas = np.radians(ca)[:, np.newaxis] + np.radians(30) + _60DEG * np.arange(6)[np.newaxis, :]
    normals = np.column_stack([
        np.cos(n_thetas).flatten(),
        np.sin(n_thetas).flatten()
    ]).reshape(n, 6, 2)
    
    # Smooth containment -> required side
    oa = np.radians(outer_angle)
    outer_n_thetas = oa + np.radians(30) + _60DEG * np.arange(6)
    outer_n = np.column_stack([np.cos(outer_n_thetas), np.sin(outer_n_thetas)]) # (6, 2)
    
    rel = verts - np.array(outer_center)
    d = np.abs(np.einsum('ijk,lk->ijl', rel, outer_n)) # (N, 6, 6)
    d_flat = d.flatten()
    mn = d_flat.min()
    exps = np.exp(beta_cont * (d_flat - mn))
    soft_max = mn + np.log(exps.sum()) / beta_cont
    req_side = soft_max / _HALF_SQRT3
    
    # Smooth overlap penalty
    pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            n1 = normals[i] # (6, 2)
            n2 = normals[j] # (6, 2)
            v1 = verts[i]   # (6, 2)
            v2 = verts[j]   # (6, 2)
            
            axes = np.vstack([n1, n2]) # (12, 2)
            
            p1 = axes @ v1.T # (12, 6)
            p2 = axes @ v2.T # (12, 6)
            
            min1 = p1.min(axis=1)
            max1 = p1.max(axis=1)
            min2 = p2.min(axis=1)
            max2 = p2.max(axis=1)
            
            seps = np.maximum(min1 - max2, min2 - max1)
            
            mn = seps.min()
            exps = np.exp(beta_sep * (seps - mn))
            s = mn - np.log(exps.sum()) / beta_sep
            if s < 0:
                pen += s * s
                
    return req_side + lam * pen

def _objective_grad(params, beta_sep=40.0, beta_cont=30.0, lam=10.0):
    """Objective with numerical gradient (central differences)."""
    f0 = _objective(params, beta_sep, beta_cont, lam)
    h = 1e-4
    grad = np.zeros_like(params)
    for i in range(len(params)):
        p = params.copy()
        p[i] += h
        f1 = _objective(p, beta_sep, beta_cont, lam)
        p[i] -= 2*h
        f2 = _objective(p, beta_sep, beta_cont, lam)
        p[i] += h
        grad[i] = (f1 - f2) / (2*h)
    return f0, grad

def _true_side(centers, outer_center, outer_angle):
    """Exact required outer side length."""
    oa = np.radians(outer_angle)
    outer_n = _hex_normals_vec(outer_angle)
    max_d = 0.0
    for c in centers:
        verts = _hex_verts_vec(c[0], c[1], c[2])
        rel = verts - np.array(outer_center)
        d = np.abs(rel @ outer_n.T)
        max_d = max(max_d, d.max())
    return max_d / _HALF_SQRT3

def _true_overlap(centers):
    """Exact SAT min separation. Returns min over all pairs (negative = overlap)."""
    n = len(centers)
    min_sep = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            c1 = centers[i]
            c2 = centers[j]
            n1 = _hex_normals_vec(c1[2])
            n2 = _hex_normals_vec(c2[2])
            axes = np.vstack([n1, n2])
            v1 = _hex_verts_vec(c1[0], c1[1], c1[2])
            v2 = _hex_verts_vec(c2[0], c2[1], c2[2])
            p1 = axes @ v1.T
            p2 = axes @ v2.T
            seps = np.maximum(p1.min(axis=1) - p2.max(axis=1), p2.min(axis=1) - p1.max(axis=1))
            min_sep = min(min_sep, seps.min())
    return min_sep

def _is_valid(centers, outer_center, outer_angle, tol=1e-7):
    """Check: no overlaps (SAT) and containment."""
    if _true_overlap(centers) < -tol:
        return False
    return True

def _make_initial(seed, mode):
    """Generate diverse initial configs."""
    rng = np.random.RandomState(seed)
    centers = []
    if mode == 0:  # hexagonal lattice
        dx = 2.0 * _SQRT3
        dy = 3.0
        positions = [
            (0, 0), (dx, 0), (-dx, 0),
            (dx/2, dy*0.866), (-dx/2, dy*0.866),
            (dx/2, -dy*0.866), (-dx/2, -dy*0.866),
            (3*dx/2, dy*0.866), (-3*dx/2, dy*0.866),
            (3*dx/2, -dy*0.866), (-3*dx/2, -dy*0.866),
            (0, dy*1.732), (0, -dy*1.732)
        ]
        for (x, y) in positions:
            centers.append([x + rng.normal(0, 0.1), y + rng.normal(0, 0.1), rng.uniform(0, 60)])
    elif mode == 1:  # circular
        for i in range(N):
            theta = 2 * np.pi * i / N + rng.uniform(-0.2, 0.2)
            r = 2.2 + rng.uniform(-0.3, 0.3)
            centers.append([r * np.cos(theta), r * np.sin(theta), np.degrees(theta) + rng.uniform(-10, 10)])
    elif mode == 2:  # random
        for i in range(N):
            centers.append([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0, 60)])
    elif mode == 3:  # two rings
        for i in range(6):
            theta = 2 * np.pi * i / 6
            r = 1.8
            centers.append([r * np.cos(theta), r * np.sin(theta), np.degrees(theta)])
        for i in range(6):
            theta = 2 * np.pi * (i + 0.5) / 6
            r = 3.5
            centers.append([r * np.cos(theta), r * np.sin(theta), np.degrees(theta)])
    else:  # spiral
        for i in range(N):
            theta = i * 2.4  # golden angle-ish
            r = 0.8 * i
            centers.append([r * np.cos(theta), r * np.sin(theta), np.degrees(theta) % 60])
    return centers

def construct():
    t0 = time.time()
    best_side = float('inf')
    best_result = None
    
    n_params = 3 * N + 3  # 39
    
    # Try fewer restarts with L-BFGS-B
    n_restarts = 10
    for r in range(n_restarts):
        elapsed = time.time() - t0
        if elapsed > DEADLINE - 10:
            break
        
        # Generate initial config
        mode = r % 5
        seed = r * 7 + 13
        centers = _make_initial(seed, mode)
        
        # Build params
        params = np.zeros(n_params)
        for i in range(N):
            params[3*i] = centers[i][0]
            params[3*i+1] = centers[i][1]
            params[3*i+2] = centers[i][2]
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        
        # Time check callback
        def time_check_callback(xk):
            if time.time() - t0 > DEADLINE - 5:
                raise TimeoutError("Time limit reached")
                
        # Run L-BFGS-B
        try:
            res = minimize(
                _objective_grad,
                params,
                method='L-BFGS-B',
                jac=True,
                callback=time_check_callback,
                options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except TimeoutError:
            # Use the current best x from the last callback? 
            # minimize doesn't easily expose current x on exception.
            # We can just skip this restart.
            continue
        except Exception:
            continue
        
        # Extract result
        p = res.x
        new_centers = [[p[3*i], p[3*i+1], p[3*i+2]] for i in range(N)]
        oc = [p[3*N], p[3*N+1]]
        oa = p[3*N+2]
        
        # Validate
        if not _is_valid(new_centers, oc, oa):
            continue
        
        side = _true_side(new_centers, oc, oa)
        if side < best_side:
            best_side = side
            best_result = (new_centers, tuple(oc), side, oa)
            
        # Check time again
        if time.time() - t0 > DEADLINE - 5:
            break
    
    # If no valid result, use a safe default
    if best_result is None:
        # Fallback: large circle arrangement
        inner_default = []
        for i in range(N):
            theta = 2 * np.pi * i / N
            x = 4.0 * np.cos(theta)
            y = 4.0 * np.sin(theta)
            inner_default.append([x, y, 0.0])
        side = _true_side(inner_default, (0.0, 0.0), 0.0)
        return inner_default, (0.0, 0.0), side, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
