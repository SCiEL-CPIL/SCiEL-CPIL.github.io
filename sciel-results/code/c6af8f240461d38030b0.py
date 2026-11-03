import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
AREA_REF = 0.5 * np.sqrt(3) / 2.0

def _bary_to_cart(bary):
    """bary: (N, 2) array of (u, v) where w=1-u-v. Returns (N, 2) Cartesian."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(cart):
    """cart: (N, 2) Cartesian. Returns (N, 2) barycentric (u, v) with w=1-u-v."""
    # Solve: P = u*V0 + v*V1 + w*V2, u+v+w=1
    # Since V0=(0,0): P = v*V1 + w*V2
    # x = v * 1 + w * 0.5
    # y = v * 0 + w * sqrt(3)/2
    # So w = y / (sqrt(3)/2) = 2y/sqrt(3)
    # v = x - 0.5*w
    # u = 1 - v - w
    x = cart[:, 0]
    y = cart[:, 1]
    w = 2.0 * y / np.sqrt(3)
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def _clamp_bary(bary):
    """Project barycentric coords into the triangle (u>=0, v>=0, w>=0)."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    w = 1.0 - u - v
    
    # Simple projection: if any coord is negative, set to 0 and renormalize
    # This is not exact projection but a valid heuristic
    for _ in range(10):
        mask_u = u < 0
        mask_v = v < 0
        mask_w = w < 0
        if not (np.any(mask_u) or np.any(mask_v) or np.any(mask_w)):
            break
        u[mask_u] = 0
        v[mask_v] = 0
        w[mask_w] = 0
        # Renormalize
        s = u + v + w
        s[s == 0] = 1.0
        u /= s
        v /= s
        w /= s
    
    return np.column_stack([u, v])

def _min_triangle_area_vectorized(cart):
    """Compute min area of triangles formed by N points. cart is (N, 2)."""
    n = cart.shape[0]
    # Compute all triangle areas
    # For points i, j, k: area = 0.5 * |(pj-pi) x (pk-pi)|
    # Vectorize using broadcasting
    # pi: (N, 1, 2), pj: (1, N, 2), pk: (1, 1, N, 2)
    pi = cart[:, np.newaxis, np.newaxis, :]  # (N, 1, 1, 2)
    pj = cart[np.newaxis, :, np.newaxis, :]  # (1, N, 1, 2)
    pk = cart[np.newaxis, np.newaxis, :, :]  # (1, 1, N, 2)
    
    # Cross product z-component: (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
    dx1 = pj[..., 0] - pi[..., 0]
    dy1 = pj[..., 1] - pi[..., 1]
    dx2 = pk[..., 0] - pi[..., 0]
    dy2 = pk[..., 1] - pi[..., 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    # We need min over i < j < k
    # Create mask for i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
    return np.min(valid_areas)

def _objective_bary(bary_flat):
    """Objective function for optimization. Returns negative min area (to minimize)."""
    bary = bary_flat.reshape(N, 2)
    # Clamp to ensure valid barycentric coords
    bary = _clamp_bary(bary)
    cart = _bary_to_cart(bary)
    
    # Check if points are inside (should be after clamping, but verify)
    # Barycentric coords: u>=0, v>=0, w>=0
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return 1e6
    
    area = _min_triangle_area_vectorized(cart)
    return -area

def _structured_init_bary():
    """Create a structured initial guess in barycentric coordinates."""
    # Start with points on a grid-like structure
    # Divide triangle into smaller regions
    bary = np.zeros((N, 2))
    
    # Place points at vertices and midpoints first
    # Vertex 0: (1, 0)
    # Vertex 1: (0, 1)
    # Vertex 2: (0, 0)
    bary[0] = [1.0, 0.0]  # V0
    bary[1] = [0.0, 1.0]  # V1
    bary[2] = [0.0, 0.0]  # V2
    
    # Midpoints of edges
    bary[3] = [0.5, 0.5]  # Mid V0-V1
    bary[4] = [0.5, 0.0]  # Mid V0-V2
    bary[5] = [0.0, 0.5]  # Mid V1-V2
    
    # Centroid
    bary[6] = [1.0/3, 1.0/3]
    
    # Additional points: divide into smaller triangles
    # Points at 1/4 and 3/4 positions
    bary[7] = [0.25, 0.25]
    bary[8] = [0.75, 0.25]
    bary[9] = [0.25, 0.75]
    bary[10] = [0.5, 0.5]  # Same as midpoint, will be adjusted
    
    return bary

def _nelder_mead_optimize_bary(bary_init, deadline, maxiter=300):
    """Run Nelder-Mead optimization on barycentric coordinates."""
    start = time.time()
    try:
        res = minimize(
            _objective_bary,
            bary_init.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        elapsed = time.time() - start
        if res.fun < 0:  # Valid result
            return res.x.reshape(N, 2), -res.fun
        return bary_init, 0.0
    except Exception:
        return bary_init, 0.0

def _l_bfgs_optimize_bary(bary_init, deadline, maxiter=200):
    """Run L-BFGS-B optimization on barycentric coordinates."""
    start = time.time()
    try:
        # Bounds: u in [0, 1], v in [0, 1], and u+v <= 1
        # For simplicity, use bounds [0, 1] for both u and v
        bounds = [(0, 1), (0, 1)] * N
        res = minimize(
            _objective_bary,
            bary_init.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        elapsed = time.time() - start
        if res.fun < 0:
            bary_opt = _clamp_bary(res.x.reshape(N, 2))
            cart = _bary_to_cart(bary_opt)
            area = _min_triangle_area_vectorized(cart)
            return bary_opt, area
        return bary_init, 0.0
    except Exception:
        return bary_init, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_bary = None
    best_area = -1e6
    
    # Multiple initializations
    inits = []
    
    # 1. Structured initialization
    bary_struct = _structured_init_bary()
    inits.append(bary_struct)
    
    # 2. Random initializations
    for i in range(5):
        # Random points in triangle
        r1 = np.random.rand(N)
        r2 = np.random.rand(N)
        # Transform to triangle
        sqrt_r1 = np.sqrt(r1)
        u = (1 - sqrt_r1) * r2
        v = (1 - sqrt_r1) * (1 - r2)
        bary_rand = np.column_stack([u, v])
        inits.append(bary_rand)
    
    # 3. Perturbed structured
    bary_pert = _structured_init_bary() + np.random.randn(N, 2) * 0.05
    bary_pert = _clamp_bary(bary_pert)
    inits.append(bary_pert)
    
    # Try each initialization with different optimizers
    for bary_init in inits:
        if time.time() > deadline - 10:
            break
        
        # First try Nelder-Mead (derivative-free, good for non-smooth)
        bary_opt, area = _nelder_mead_optimize_bary(bary_init, deadline, maxiter=200)
        if area > best_area:
            best_area = area
            best_bary = bary_opt.copy()
        
        # Then try L-BFGS-B if time permits
        if time.time() > deadline - 20:
            break
        
        bary_opt2, area2 = _l_bfgs_optimize_bary(bary_opt, deadline, maxiter=100)
        if area2 > best_area:
            best_area = area2
            best_bary = bary_opt2.copy()
    
    # If we have time, try differential evolution for global search
    if time.time() < deadline - 30 and best_area < 0.01:
        try:
            bounds = [(0, 1), (0, 1)] * N
            res_de = differential_evolution(
                _objective_bary,
                bounds,
                maxiter=30,
                popsize=10,
                tol=0.01,
                seed=42,
                polish=False
            )
            if res_de.fun < 0:
                bary_de = _clamp_bary(res_de.x.reshape(N, 2))
                cart_de = _bary_to_cart(bary_de)
                area_de = _min_triangle_area_vectorized(cart_de)
                if area_de > best_area:
                    best_area = area_de
                    best_bary = bary_de.copy()
        except Exception:
            pass
    
    # Final polish with Nelder-Mead if we have a good candidate
    if best_bary is not None and time.time() < deadline - 5:
        bary_final, area_final = _nelder_mead_optimize_bary(best_bary, deadline, maxiter=100)
        if area_final > best_area:
            best_area = area_final
            best_bary = bary_final
    
    # Convert to Cartesian
    if best_bary is None:
        # Fallback: return structured init
        best_bary = _structured_init_bary()
    
    best_bary = _clamp_bary(best_bary)
    cart = _bary_to_cart(best_bary)
    
    return cart
