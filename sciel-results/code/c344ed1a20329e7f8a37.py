"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations
from scipy.spatial import Delaunay

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Extract coordinates for all triangles
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    # Determinant calculation vectorized
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def calc_signed_dets(bary):
    """
    Calculate signed determinants (areas) of all triangles.
    Returns array of shape (165,) with signed values.
    """
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
    
    return det

def clamp_bary(bary, eps=1e-6):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to strict interior
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    # Ensure u + v <= 1 - eps
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    For 11 points, we can use a triangular arrangement.
    """
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * np.sqrt(3) / 2
                if u + v <= 1 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, noise_scale=0.05, seed=None):
    """Generate perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def _penalty_gradient_ascent(bary_init, n_iter=5000, deadline=None, gamma=50.0):
    """
    Penalty-based gradient ascent: maximize min area by penalizing triangles below current min.
    Uses 3x3 grid + 2 free points structure for better conditioning.
    """
    if deadline is not None and time.time() > deadline - 2:
        return bary_init
    
    # Initialize: 3x3 grid + 2 free points
    # Grid positions in barycentric (u, v) for 3x3:
    # (0,0), (1/3,0), (2/3,0)
    # (0,1/3), (1/3,1/3), (2/3,1/3)
    # (0,2/3), (1/3,2/3), (2/3,2/3)
    grid_ideal = np.array([
        [0.0, 0.0], [1/3, 0.0], [2/3, 0.0],
        [0.0, 1/3], [1/3, 1/3], [2/3, 1/3],
        [0.0, 2/3], [1/3, 2/3], [2/3, 2/3]
    ])
    
    # Start with ideal grid, then add 2 free points
    bary = np.zeros((N, 2))
    bary[:9] = grid_ideal
    
    # Place 2 free points at center-ish positions
    bary[9] = [0.5, 0.25]
    bary[10] = [0.25, 0.5]
    
    # If we have a better initial, use it
    if bary_init is not None:
        bary = bary_init.copy()
    
    bary = clamp_bary(bary)
    
    current_min_area = np.min(calc_areas_bary(bary))
    best_bary = bary.copy()
    best_min_area = current_min_area
    
    rng = np.random.default_rng(42)
    step_size = 0.01
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        areas = calc_areas_bary(bary)
        current_min_area = np.min(areas)
        
        # Find triangles that are below threshold (current min + small margin)
        threshold = current_min_area * 0.95
        below_mask = areas < threshold
        
        if not np.any(below_mask):
            # No triangles below threshold, try to increase overall min
            # Use gradient of soft min
            beta = 100.0
            scaled = -beta * areas
            max_s = np.max(scaled)
            weights = np.exp(scaled - max_s)
            weights /= np.sum(weights)
            
            # Compute gradient numerically
            grad = np.zeros_like(bary)
            for idx in range(N):
                for dim in range(2):
                    bary_plus = bary.copy()
                    bary_plus[idx, dim] += 1e-5
                    bary_plus = clamp_bary(bary_plus)
                    areas_plus = calc_areas_bary(bary_plus)
                    
                    bary_minus = bary.copy()
                    bary_minus[idx, dim] -= 1e-5
                    bary_minus = clamp_bary(bary_minus)
                    areas_minus = calc_areas_bary(bary_minus)
                    
                    # Gradient of weighted sum
                    grad_plus = np.sum(weights * areas_plus)
                    grad_minus = np.sum(weights * areas_minus)
                    grad[idx, dim] = (grad_plus - grad_minus) / (2e-5)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad /= grad_norm
            
            # Take step
            new_bary = bary + step_size * grad
            new_bary = clamp_bary(new_bary)
            new_areas = calc_areas_bary(new_bary)
            new_min = np.min(new_areas)
            
            if new_min > current_min_area:
                bary = new_bary
                current_min_area = new_min
                if current_min_area > best_min_area:
                    best_min_area = current_min_area
                    best_bary = bary.copy()
            else:
                step_size *= 0.95
        else:
            # Penalize triangles below threshold
            # For each violating triangle, compute which points to move
            violating_tris = tri_idx[below_mask]
            
            # Aggregate gradient: push points apart for violating triangles
            grad = np.zeros_like(bary)
            for tri in violating_tris:
                i, j, k = tri
                # Compute area
                area = areas[below_mask]
                
                # Simple repulsion: move each point away from centroid of the other two
                for p, q, r in [(i, j, k), (j, k, i), (k, i, j)]:
                    # Move p away from midpoint of q and r
                    mid_u = (bary[q, 0] + bary[r, 0]) / 2
                    mid_v = (bary[q, 1] + bary[r, 1]) / 2
                    du = bary[p, 0] - mid_u
                    dv = bary[p, 1] - mid_v
                    dist = np.sqrt(du**2 + dv**2) + 1e-10
                    # Weight by how much below threshold
                    weight = (threshold - area) / max(threshold, 1e-10)
                    grad[p, 0] += weight * du / dist
                    grad[p, 1] += weight * dv / dist
            
            # Normalize
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad /= grad_norm
            
            # Take step
            new_bary = bary + step_size * grad
            new_bary = clamp_bary(new_bary)
            new_areas = calc_areas_bary(new_bary)
            new_min = np.min(new_areas)
            
            if new_min > current_min_area * 0.98:  # Allow slight decrease
                bary = new_bary
                current_min_area = new_min
                if current_min_area > best_min_area:
                    best_min_area = current_min_area
                    best_bary = bary.copy()
            else:
                step_size *= 0.95
        
        # Anneal step size
        step_size = max(0.001, step_size * 0.999)
        
        # Occasionally add small random perturbation
        if i % 100 == 0:
            bary += rng.normal(0, 0.005, bary.shape)
            bary = clamp_bary(bary)
    
    return best_bary

def _local_polish_bary(bary, max_iter=200, deadline=None):
    """
    Local polish using coordinate descent with small steps.
    For each point, try small perturbations in u and v directions.
    """
    if deadline is not None and time.time() > deadline - 2:
        return bary
    
    current_bary = clamp_bary(bary.copy())
    current_val = -hard_min_obj(current_bary.flatten())
    
    rng = np.random.default_rng(123)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        improved = False
        for i in range(N):
            # Try small steps in u direction
            for du in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01]:
                trial = current_bary.copy()
                trial[i, 0] += du
                trial = clamp_bary(trial)
                trial_val = -hard_min_obj(trial.flatten())
                if trial_val > current_val:
                    current_bary = trial
                    current_val = trial_val
                    improved = True
                    break
            
            # Try small steps in v direction
            for dv in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01]:
                trial = current_bary.copy()
                trial[i, 1] += dv
                trial = clamp_bary(trial)
                trial_val = -hard_min_obj(trial.flatten())
                if trial_val > current_val:
                    current_bary = trial
                    current_val = trial_val
                    improved = True
                    break
        
        if not improved:
            break
    
    return current_bary

def construct():
    start_time = time.time()
    # Evaluator kills after 150s. We use 140s deadline to be safe.
    deadline = start_time + 140.0
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # --- Phase 1: Penalty-based gradient ascent with 3x3 grid structure ---
    # Start from ideal grid + 2 free points
    grid_ideal = np.array([
        [0.0, 0.0], [1/3, 0.0], [2/3, 0.0],
        [0.0, 1/3], [1/3, 1/3], [2/3, 1/3],
        [0.0, 2/3], [1/3, 2/3], [2/3, 2/3]
    ])
    init_bary = np.zeros((N, 2))
    init_bary[:9] = grid_ideal
    init_bary[9] = [0.5, 0.25]
    init_bary[10] = [0.25, 0.5]
    init_bary = clamp_bary(init_bary)
    
    pa_result = _penalty_gradient_ascent(init_bary, n_iter=8000, deadline=deadline)
    update_best(pa_result)
    
    # --- Phase 2: Multi-start with different initial configurations ---
    if time.time() < deadline - 40:
        # Hexagonal init
        hex_init = generate_hexagonal_init()
        pa_result = _penalty_gradient_ascent(hex_init, n_iter=5000, deadline=deadline)
        update_best(pa_result)
    
    # Random inits
    for i in range(5):
        if time.time() > deadline - 30:
            break
        rand_init = generate_random_init(seed=100 + i * 7)
        pa_result = _penalty_gradient_ascent(rand_init, n_iter=4000, deadline=deadline)
        update_best(pa_result)
    
    # --- Phase 3: L-BFGS refinement from best ---
    if best_bary is not None and time.time() < deadline - 20:
        try:
            def soft_obj(x):
                return soft_min_obj(x, beta=100.0)
            
            bounds = [(0, 1)] * (N * 2)
            result = minimize(
                soft_obj,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass
    
    # --- Phase 4: Nelder-Mead polish ---
    if best_bary is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass
    
    # --- Phase 5: Local polish with coordinate descent ---
    if best_bary is not None and time.time() < deadline - 5:
        polished = _local_polish_bary(best_bary, max_iter=100, deadline=deadline)
        update_best(polished)
        
        # Additional polish rounds
        for i in range(3):
            if time.time() > deadline - 3:
                break
            polished = _local_polish_bary(best_bary, max_iter=50, deadline=deadline)
            update_best(polished)
    
    # --- Phase 6: Final NM polish ---
    if best_bary is not None and time.time() < deadline - 3:
        try:
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass
    
    # Ensure we have a valid result
    if best_bary is None:
        best_bary = generate_random_init(seed=1)
        
    # Final clamp
    best_bary = clamp_bary(best_bary)
    
    # Convert to Cartesian
    cart_pts = bary_to_cart(best_bary)
    
    return cart_pts

# Test run if executed directly
if __name__ == "__main__":
    pts = construct()
    print("Points:")
    print(pts)
    # Check validity
    for p in pts:
        if p[0] < -1e-9 or p[1] < -1e-9 or p[0] > 1 + 1e-9:
            print("Invalid point:", p)
        if p[1] > np.sqrt(3) * p[0] + 1e-9 or p[1] > np.sqrt(3) * (1 - p[0]) + 1e-9:
            print("Invalid point (triangle bound):", p)
    # Calculate score
    bary = np.zeros((N, 2))
    # Inverse of bary_to_cart for debugging score
    # x = v + 0.5*w, y = w*H => w = y/H, v = x - 0.5*w, u = 1 - v - w
    w = pts[:, 1] / H
    v = pts[:, 0] - 0.5 * w
    u = 1 - v - w
    bary = np.column_stack([u, v])
    areas = calc_areas_bary(bary)
    print("Min Area (normalized):", np.min(areas))
    print("Max Area (normalized):", np.max(areas))
