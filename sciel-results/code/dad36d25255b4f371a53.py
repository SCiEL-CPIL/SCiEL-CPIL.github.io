import numpy as np
from itertools import combinations
import time
from scipy.optimize import minimize

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def min_area(bary):
    """Compute the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def soft_min_value(areas, beta):
    """
    Calculate the soft minimum of areas using log-sum-exp.
    Returns a value close to min(areas) but differentiable.
    """
    # Use the stable log-sum-exp trick
    min_val = np.min(areas)
    # If beta is very large, this approaches the true min
    # We want to minimize -soft_min, so we maximize soft_min.
    # soft_min = -1/beta * log(sum(exp(-beta * area)))
    # To avoid overflow/underflow, subtract the min exponent (which is -beta*min_area)
    # exp(-beta * (area - min_area)) is <= 1
    shifted = -beta * (areas - min_val)
    # Sum of exps
    sum_exp = np.sum(np.exp(shifted))
    log_sum = np.log(sum_exp)
    soft_min_val = min_val - (1.0 / beta) * log_sum
    return soft_min_val

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    # Try different scales and offsets
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            rng = np.random.default_rng(42)
            # Try multiple random subsets
            for _ in range(20):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = min_area(bary)
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def generate_symmetric_init():
    """
    Generate an initialization exploiting the 3-fold symmetry of the triangle.
    Points are placed in orbits of size 3 (or 1 if on axis/center).
    This reduces the effective search space and enforces good distribution.
    """
    # Center point
    center = np.array([1/3, 1/3])
    pts = [center]
    
    # Add 3 orbits of 3 points each (total 9 points) + center = 10?
    # We need 11.
    # Let's do:
    # 1 center
    # 3 points on one axis (including center? no, distinct)
    # Actually, let's just generate random points and symmetrize them to be closer to symmetric.
    # Or, generate 3 base points and rotate them.
    
    rng = np.random.default_rng(123)
    
    # 1 center
    pts = [center]
    
    # 3 points in one sector, then rotate
    # Sector is 1/3 of the triangle.
    # Let's pick 3 random points in the triangle and rotate them by 120 and 240.
    # That gives 9 points.
    # Plus center = 10.
    # We need 1 more. Add a point on the boundary or another center?
    # Let's add a point near a vertex? No, that creates small triangles.
    # Let's add a point at the midpoint of a side?
    # Midpoint of side 0-1 is (0.5, 0). Bary: (0.5, 0.5, 0) -> (0.5, 0.5)
    # Midpoint of side 1-2 is (0.75, H/4)? No.
    # Bary for midpoint of (1,0) and (0.5, H/2) is (0.25, 0.75, 0) -> (0.25, 0.75)
    # Midpoint of side 2-0 is (0.25, H/4)? Bary (0.75, 0.25, 0) -> (0.75, 0.25)
    
    # Let's try:
    # 1 center
    # 3 midpoints of sides
    # 6 points: 2 in each "corner" region?
    
    # Simpler:
    # Generate 4 random points.
    # Rotate each by 0, 120, 240.
    # That's 12 points.
    # Remove one? Or use 3 points -> 9, plus center -> 10, plus midpoint -> 11.
    
    # Let's use 3 random points + center + 3 midpoints.
    # Total: 3*3 + 1 + 3 = 13. Too many.
    
    # Let's use 2 random points + center + 3 midpoints.
    # Total: 2*3 + 1 + 3 = 10. Need 1 more.
    # Add a point at the centroid of the 3 midpoints? That's the center.
    
    # Let's just do:
    # 1 center
    # 3 midpoints
    # 6 points: 3 in one sector, rotated.
    # Total: 1 + 3 + 6 = 10.
    # Add 1 more point on the axis of symmetry?
    
    # Let's try a specific known good pattern if I can recall one, or just randomize the "orbit" points.
    
    # Orbits:
    # 1 center (1,1,1)
    # 3 points on the 3 axes of symmetry?
    # Axis 1: (1/3, 1/3) to (1,0). Points: (1/3 + t, 1/3 - t, 1/3)
    # Let's place points at t = 0.1, 0.2, 0.3 along each axis?
    # That's 3 points per axis * 3 axes = 9 points.
    # Plus center = 10.
    # Plus 1 point on a side?
    
    # Let's generate:
    # Center
    # 3 points along axis 1
    # 3 points along axis 2 (rotated)
    # 3 points along axis 3 (rotated)
    # That's 10 points.
    # Add 1 point at a vertex? No.
    # Add 1 point at the midpoint of a side?
    
    # Let's just generate 3 random points in the triangle, rotate them to get 9, add center (10), add midpoint of side (11).
    
    base_pts = []
    for _ in range(3):
        u = rng.random()
        v = rng.random() * (1 - u)
        base_pts.append([u, v])
        
    # Rotate base_pts
    # Rotation in barycentric coords:
    # (u, v, w) -> (v, w, u) -> (w, u, v)
    # So (u, v) -> (v, 1-u-v) -> (1-u-v, u)
    
    all_pts = []
    for pt in base_pts:
        u, v = pt
        w = 1 - u - v
        all_pts.append([u, v])
        all_pts.append([v, w])
        all_pts.append([w, u])
        
    # Add center
    all_pts.append([1/3, 1/3])
    
    # Add midpoint of side 0-1: (0.5, 0.5)
    all_pts.append([0.5, 0.5])
    
    # We have 3*3 + 1 + 1 = 11 points.
    bary = np.array(all_pts)
    return clamp_bary(bary)

def refine_soft_min(bary_init, time_limit=10.0):
    """
    Refine using a soft-minimum objective with annealing beta.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    current_val = min_area(bary_current)
    
    # Annealing schedule for beta
    # Start with small beta (smooth), increase to large beta (sharp)
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > start_time + time_limit * 0.8:
            break
            
        def obj(x):
            b = x.reshape(N, 2)
            areas = calc_areas_bary(b)
            # Check validity
            u = b[:, 0]
            v = b[:, 1]
            w = 1 - u - v
            if np.any(u < -1e-6) or np.any(v < -1e-6) or np.any(w < -1e-6):
                return 1e6
            sm_val = soft_min_value(areas, beta)
            return -sm_val
            
        try:
            result = minimize(
                obj,
                bary_current.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            candidate = clamp_bary(result.x.reshape(N, 2))
            val = min_area(candidate)
            if val > current_val:
                bary_current = candidate
                current_val = val
        except Exception:
            pass
            
    # Final polish with Nelder-Mead on the hard min
    if time.time() < start_time + time_limit:
        try:
            result_nm = minimize(
                lambda x: -min_area(x.reshape(N, 2)),
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
            )
            candidate = clamp_bary(result_nm.x.reshape(N, 2))
            val = min_area(candidate)
            if val > current_val:
                bary_current = candidate
                current_val = val
        except Exception:
            pass

    return bary_current, current_val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Generate diverse initializations
    inits = []
    
    # Hexagonal
    hex_init = generate_hexagonal_init()
    inits.append(hex_init)
    
    # Symmetric
    sym_init = generate_symmetric_init()
    inits.append(sym_init)
    
    # Random
    for i in range(5):
        rng = np.random.default_rng(1000 + i)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    # Perturbed hex
    rng = np.random.default_rng(42)
    for i in range(5):
        noise = rng.normal(0, 0.05, hex_init.shape)
        inits.append(clamp_bary(hex_init + noise))
        
    # 2. Refine each initialization
    for i, init in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        time_budget = min(10.0, remaining / 2) # Simple heuristic
        
        refined, val = refine_soft_min(init, time_limit=time_budget)
        update_best(refined)
        
    # 3. If time remains, do a final high-precision polish on the best found
    if best_bary is not None and time.time() < deadline - 5:
        try:
            # One more SLSQP pass with tighter tolerance
            bounds = [(0, 1)] * (N * 2)
            result_final = minimize(
                lambda x: -min_area(x.reshape(N, 2)),
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass
        
        # One more Nelder-Mead pass
        if time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    lambda x: -min_area(x.reshape(N, 2)),
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
