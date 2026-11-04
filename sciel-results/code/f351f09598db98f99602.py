"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(points):
    """Largest common radius: min of half pairwise distances and wall clearance."""
    # points is (N, 3)
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    # Get unique pairs
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min()
    clearance = (0.5 - np.abs(points)).min()
    return min(min_pair * 0.5, clearance)

def _energy_and_grad(x, target_r):
    """
    Energy function for BFGS optimization.
    x is a flat array of length 3N.
    target_r is the desired radius.
    Returns (energy, gradient).
    """
    c = x.reshape(N, 3)
    
    # --- Pairwise Part ---
    # diff[i, j] = c[i] - c[j]
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    # dists[i, j] = ||c[i] - c[j]||
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Active pairs: those closer than 2 * target_r
    active = dists < 2.0 * target_r
    # Ensure diagonal is not active
    active[np.diag_indices_from(active)] = False
    
    # Violations: (2r - d) for active pairs, 0 otherwise
    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    
    # Energy is sum of squared violations (0.5 factor for clean gradient)
    energy_pair = 0.5 * np.sum(violations**2)
    
    # Gradient calculation for pairs
    # d/dx_i [ 0.5 * (2r - ||x_i - x_j||)^2 ]
    # = (2r - d) * d/dx_i (2r - d)
    # = (2r - d) * (-(x_i - x_j) / d)
    # = - (2r - d) / d * (x_i - x_j)
    # Note: violations[i,j] = 2r - d. So we need - violations[i,j] / d * (x_i - x_j)
    # To avoid division by zero, we use safe_dists
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # weights[i,j] = violations[i,j]
    # grad_i = - sum_j weights[i,j] * unit_vecs[i,j]
    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # --- Wall Part ---
    # Distance to each wall
    # Walls are at -0.5 and 0.5 for each dimension
    # dist_wall_k = 0.5 - |c_k|
    wall_dists = 0.5 - np.abs(c)
    
    # Violations: target_r - wall_dist if positive
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    
    energy_wall = np.sum(wall_violations**2)
    
    # Gradient for walls
    # d/dc [ (r - (0.5 - |c|))^2 ]
    # = 2 * (r - (0.5 - |c|)) * d/dc (0.5 - |c|)
    # = 2 * wall_violation * (-sign(c))
    signs = np.sign(c)
    # Handle c=0 case: sign(0)=0, which is fine, gradient is 0 at center
    grad_wall = 2.0 * wall_violations * signs
    
    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()
    
    return total_energy, total_grad

def _optimize_bfgs(init_pts, target_r, max_iter=1000, gtol=1e-9, time_limit=50.0):
    """
    Run BFGS optimization with a time limit.
    Returns (optimized_pts, success).
    """
    t_start = time.time()
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)
    
    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': gtol, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts, result.success
    except TimeoutError:
        return init_pts, False
    except Exception:
        return init_pts, False

# Jellyfish representation: 5x5x5 base lattice + 8 corner anchors
# 5^3 = 125 lattice sites, take first 100
# 8 corners, each with 3 coords = 24 anchor params
# Total: 24 params for global shape morphing

_BASE_LATTICE = None
_CORNER_INDICES = None
_LATTICE_WEIGHTS = None

def _init_jellyfish():
    """Initialize jellyfish representation components."""
    global _BASE_LATTICE, _CORNER_INDICES, _LATTICE_WEIGHTS
    
    if _BASE_LATTICE is not None:
        return
    
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    # Generate 5x5x5 grid, take first 100
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    _BASE_LATTICE = grid[:N].copy()
    
    # 8 corners of the cube [-0.5, 0.5]^3
    corners = np.array([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
        [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
        [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5],
    ])
    _CORNER_INDICES = corners  # 8x3
    
    # For each lattice site, compute weights to the 8 corners
    # Weight is inverse-cubed distance, normalized
    # This gives smooth blending
    sites = _BASE_LATTICE  # (100, 3)
    corners = _CORNER_INDICES  # (8, 3)
    
    # dists[i, j] = ||site_i - corner_j||^3
    diff = sites[:, np.newaxis, :] - corners[np.newaxis, :, :]
    dists = np.sum(diff**2, axis=2) ** 1.5  # (100, 8)
    
    # Add small epsilon to avoid zero division
    dists += 1e-10
    
    # Weights: inverse distance
    weights = 1.0 / dists  # (100, 8)
    weights = weights / weights.sum(axis=1, keepdims=True)  # normalize to sum=1
    
    _LATTICE_WEIGHTS = weights  # (100, 8)

def _jellyfish_decode(params):
    """
    Decode 24 anchor parameters into 100 point positions.
    params: (24,) array of corner offsets
    Returns: (100, 3) array of positions
    
    Each corner has 3 offset components.
    The actual corner position = base_corner + offset.
    Each site is a weighted blend of the 8 modified corners.
    """
    _init_jellyfish()
    
    corners_base = _CORNER_INDICES  # (8, 3)
    offsets = params.reshape(8, 3)  # (8, 3)
    
    # Modified corner positions
    corners_mod = corners_base + offsets  # (8, 3)
    
    # Blend: positions[i] = sum_j weights[i,j] * corners_mod[j]
    weights = _LATTICE_WEIGHTS  # (100, 8)
    positions = weights @ corners_mod  # (100, 3)
    
    # Clip to stay inside cube
    positions = np.clip(positions, -0.499, 0.499)
    
    return positions

def _jellyfish_energy_and_grad(params, target_r):
    """
    Energy and gradient for jellyfish parameter space.
    params: (24,) array
    Returns: (energy, gradient of shape (24,))
    """
    positions = _jellyfish_decode(params)
    
    # Compute energy and gradient in position space
    energy, pos_grad = _energy_and_grad(positions.flatten(), target_r)
    pos_grad = pos_grad.reshape(N, 3)  # (100, 3)
    
    # Backpropagate gradient to corner parameters
    # positions[i] = sum_j weights[i,j] * corners_mod[j]
    # corners_mod[j] = corners_base[j] + offsets[j]
    # d(positions[i])/d(offsets[j]) = weights[i,j] * I_3
    
    weights = _LATTICE_WEIGHTS  # (100, 8)
    
    # grad_params[j, k] = sum_i pos_grad[i, k] * weights[i, j]
    # = (pos_grad.T @ weights)[k, j]
    grad_params = (pos_grad.T @ weights).T  # (8, 3)
    grad_params = grad_params.flatten()  # (24,)
    
    return energy, grad_params

def _optimize_jellyfish(init_params, target_r, max_iter=500, gtol=1e-9, time_limit=20.0):
    """
    Run BFGS optimization in jellyfish parameter space.
    Returns (optimized_params, success).
    """
    t_start = time.time()
    
    def objective(params):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _jellyfish_energy_and_grad(params, target_r)
    
    try:
        result = minimize(
            objective,
            init_params.copy(),
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': gtol, 'disp': False}
        )
        return result.x, result.success
    except TimeoutError:
        return init_params, False
    except Exception:
        return init_params, False

def _contact_graph_refine(pts, target_r, max_iter=200, time_limit=15.0):
    """
    Refine using only active (contact) pairs for faster convergence.
    """
    t_start = time.time()
    c = pts.copy()
    
    for _ in range(max_iter):
        if time.time() - t_start > time_limit:
            break
        
        # Compute distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0)
        
        # Active pairs: closer than 2*target_r
        active = dists < 2.0 * target_r
        active[np.diag_indices_from(active)] = False
        
        # Wall violations
        wall_dists = 0.5 - np.abs(c)
        wall_viol = np.maximum(0.0, target_r - wall_dists)
        
        # Check if converged
        max_viol = max(np.max(np.where(active, 2.0*target_r - dists, 0.0)), 
                       np.max(wall_viol))
        if max_viol < 1e-10:
            break
        
        # Gradient
        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        violations = np.where(active, 2.0*target_r - dists, 0.0)
        grad = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
        
        signs = np.sign(c)
        grad += 2.0 * wall_viol * signs
        
        # Line search
        best_alpha = 1.0
        best_energy = _energy_and_grad(c.flatten(), target_r)[0]
        
        for alpha in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
            c_new = c - alpha * grad
            c_new = np.clip(c_new, -0.499, 0.499)
            e_new = _energy_and_grad(c_new.flatten(), target_r)[0]
            if e_new < best_energy:
                best_energy = e_new
                best_alpha = alpha
                c = c_new
                break
    
    return c

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # Safety margin under 150s
    
    _init_jellyfish()
    rng = np.random.RandomState(42)
    
    # Initialize with zero offsets (pure lattice)
    zero_params = np.zeros(24)
    init_pts = _jellyfish_decode(zero_params)
    
    best_pts = init_pts.copy()
    best_r = _score(best_pts)
    
    # Phase 1: Jellyfish global optimization
    # Try different target radii with jellyfish
    targets = [0.105, 0.110, 0.112, 0.114, 0.116]
    
    for target_r in targets:
        if time.time() - t0 > DEADLINE - 60.0:
            break
        
        # Try multiple random initializations of offsets
        for restart in range(3):
            if time.time() - t0 > DEADLINE - 60.0:
                break
            
            noise_scale = 0.02 if restart == 0 else 0.01
            init_params = rng.normal(0, noise_scale, size=24)
            
            opt_params, success = _optimize_jellyfish(
                init_params,
                target_r,
                max_iter=300,
                gtol=1e-9,
                time_limit=DEADLINE - (time.time() - t0) - 30.0
            )
            
            if np.all(np.isfinite(opt_params)):
                opt_pts = _jellyfish_decode(opt_params)
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _score(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()
    
    # Phase 2: Full coordinate polishing
    # Use the best jellyfish result as starting point
    if time.time() - t0 < DEADLINE - 30.0:
        # Polish with increasing targets
        polish_targets = [best_r * 0.98, best_r, best_r * 1.005, best_r * 1.01]
        
        for target_r in polish_targets:
            if time.time() - t0 > DEADLINE - 15.0:
                break
            
            noise = rng.normal(0, 0.003, size=(N, 3))
            init_pts = np.clip(best_pts + noise, -0.49, 0.49)
            
            opt_pts, success = _optimize_bfgs(
                init_pts,
                target_r,
                max_iter=1500,
                gtol=1e-11,
                time_limit=DEADLINE - (time.time() - t0) - 10.0
            )
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _score(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()
    
    # Phase 3: Contact graph refinement
    if time.time() - t0 < DEADLINE - 10.0 and best_r > 0.1:
        refined = _contact_graph_refine(
            best_pts,
            best_r * 1.002,
            max_iter=300,
            time_limit=DEADLINE - (time.time() - t0) - 5.0
        )
        refined = np.clip(refined, -0.5 + 1e-9, 0.5 - 1e-9)
        r_ref = _score(refined)
        
        if r_ref > best_r:
            best_r = r_ref
            best_pts = refined.copy()
    
    # Phase 4: Final high-precision polish
    if time.time() - t0 < DEADLINE - 5.0 and best_r > 0.1:
        noise = rng.normal(0, 0.001, size=(N, 3))
        init_pts = np.clip(best_pts + noise, -0.49, 0.49)
        
        opt_pts, success = _optimize_bfgs(
            init_pts,
            best_r,
            max_iter=3000,
            gtol=1e-13,
            time_limit=DEADLINE - (time.time() - t0) - 2.0
        )
        
        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _score(opt_pts)
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
    
    # Final safety check
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _BASE_LATTICE.copy()
        
    return best_pts
# EVOLVE-BLOCK-END
