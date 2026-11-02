"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# Global time management
_START_TIME = None
_DEADLINE = None

def _init_time():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    # 150s limit, keep 5s safety margin
    _DEADLINE = _START_TIME + 145.0

def _time_remaining():
    if _DEADLINE is None:
        return 140.0
    return max(0.0, _DEADLINE - time.time())

# EVOLVE-BLOCK-START

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0) using log-sum-exp."""
    return np.logaddexp(0, z / eps) * eps

def _objective(params, penalty_weight, smooth_eps=1e-4):
    """
    Vectorized objective for N=26 circles.
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary violations (smooth)
    # x - r >= 0  => r - x <= 0
    # x + r <= 1  => x + r - 1 <= 0
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    
    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
    # Full pairwise overlaps
    # N=26 is small, O(N^2) is fine
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _validate(xyz, tol=1e-7):
    """Check validity using vectorized operations."""
    n = xyz.shape[0]
    x, y, r = xyz[:,0], xyz[:,1], xyz[:,2]
    
    # Check radii
    if np.any(r < -1e-9):
        return False
        
    # Check boundaries
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
        
    # Check overlaps
    # Create distance matrix
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx*dx + dy*dy)
    
    # Sum of radii matrix
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Check all pairs i != j
    # dist[i,j] >= r[i] + r[j] - tol
    # We only need to check upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist[mask] < r_sum[mask] - tol):
        return False
        
    return True

def _fix_boundaries(xyz):
    """Project circles inside unit square."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0, xyz[i,2])
        if r >= 0.5:
            xyz[i,0] = 0.5
            xyz[i,1] = 0.5
        else:
            xyz[i,0] = np.clip(xyz[i,0], r, 1-r)
            xyz[i,1] = np.clip(xyz[i,1], r, 1-r)
        xyz[i,2] = r
    return xyz

def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5/side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i+0.5)/side
            y = (j+0.5)/side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r*0.5])
    return np.array(positions[:N])

def _initial_hex():
    """Hexagonal packing initial config, optimized for 26 circles."""
    # 5x5 grid = 25 circles. Add 1 in center? 
    # Or 6x4 = 24 + 2?
    # Let's use a 5x5 grid with slightly larger radius, and add one small circle.
    # Or better, use a hexagonal lattice that naturally fits.
    # Hex packing density is ~0.906. 
    # Let's try a 5x5 grid of radius 0.1. 
    # Spacing = 2*r = 0.2. 
    # Width = 4*0.2 + 2*0.1 = 1.0. Fits perfectly.
    # So 25 circles of r=0.1 fit in a 5x5 grid.
    # The 26th circle must be smaller.
    
    positions = []
    r_main = 0.098 # Slightly less than 0.1 to allow for optimization slack
    # 5x5 grid
    for i in range(5):
        for j in range(5):
            x = 0.1 + i * 0.2
            y = 0.1 + j * 0.2
            positions.append([x, y, r_main])
    
    # Add 26th circle in the center, smaller
    positions.append([0.5, 0.5, 0.05])
    
    return np.array(positions[:N])

def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])

def _initial_ring(seed=0):
    """Ring-shaped initial config."""
    rng = np.random.default_rng(seed)
    positions = [[0.5, 0.5, 0.08]]
    for ring, n_circles, radius in [(1, 6, 0.10), (2, 10, 0.08), (3, 9, 0.07)]:
        for k in range(n_circles):
            angle = 2*np.pi*k/n_circles + rng.uniform(-0.1, 0.1)
            dist = ring * 0.14
            x = 0.5 + dist*np.cos(angle)
            y = 0.5 + dist*np.sin(angle)
            positions.append([x, y, radius])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1,0.9), rng.uniform(0.1,0.9), 0.06])
    return np.array(positions[:N])

def _optimize_single(xyz0, time_budget):
    """Optimize from a single initial config with adaptive penalty."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
        
    # List of (penalty, eps, max_iter)
    phases = [
        (5.0, 1e-2, 200),
        (50.0, 1e-3, 300),
        (500.0, 1e-4, 300),
        (5000.0, 1e-5, 300),
        (50000.0, 1e-6, 200) # Polish phase
    ]
    
    for penalty, eps, max_iter in phases:
        remaining = _time_remaining()
        # Ensure we have enough time for this phase
        # L-BFGS-B is fast, but let's be safe
        if remaining < 1.0:
            break
            
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, eps),
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            
            # Check if this result is better
            current_obj = _objective(params, penalty, eps)
            res_obj = res.fun
            # Note: The objective function includes the penalty.
            # We want to maximize sum(r), which is -objective_without_penalty.
            # However, if the penalty changes, comparing raw objective values is tricky.
            # Better to compare the actual sum of radii for valid solutions, 
            # or the objective value at the *same* penalty/eps if we trust the convergence.
            # Since we are increasing penalty, a lower objective value at a higher penalty 
            # is a strong indicator of a better solution (less penalty + better radius).
            
            if res.fun < current_obj:
                params = res.x
                # Update best if this looks better
                # We can't simply compare best_score (sum of radii) with res.fun
                # because res.fun includes penalty.
                # We should extract the radii and check validity?
                # For speed, we assume if res.fun < current_obj, it's an improvement in the optimization landscape.
                # We will do a final validation and score check in construct().
                # For now, just track params.
                pass 
                
        except Exception:
            pass
            
    return params.reshape(N, 3)

def construct():
    _init_time()
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate diverse initial configurations
    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
        _initial_ring(0),
        _initial_ring(1)
    ]
    
    # Optimize from each
    for xyz0 in inits:
        remaining = _time_remaining()
        if remaining < 10.0:
            break
            
        # Allocate time to this initialization
        # Simple heuristic: use most of the remaining time, but leave some for others?
        # Or just run until the global deadline? 
        # The _optimize_single function checks _time_remaining() internally.
        # So we can just call it. It will stop when time runs out.
        
        xyz_opt = _optimize_single(xyz0, remaining)
        
        # Validate and repair
        if not _validate(xyz_opt):
            xyz_fix = xyz_opt.copy()
            # Try to fix by scaling radii
            for _ in range(20):
                if _validate(xyz_fix):
                    break
                xyz_fix[:, 2] *= 0.95
                xyz_fix = _fix_boundaries(xyz_fix)
            xyz_opt = xyz_fix
            
        if not _validate(xyz_opt):
            continue
            
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
            
    # Local refinement from the best solution found
    remaining = _time_remaining()
    if remaining > 10.0:
        rng = np.random.default_rng(42)
        # Do a few perturbations
        max_perturbs = 5
        for _ in range(max_perturbs):
            if _time_remaining() < 5.0:
                break
            # Perturb the best solution
            perturbed = best_xyz.copy()
            noise = rng.normal(0, 0.01, perturbed.shape)
            noise[:, 2] *= 0.1
            perturbed += noise
            perturbed = _fix_boundaries(perturbed)
            
            xyz_opt = _optimize_single(perturbed, _time_remaining())
            
            if not _validate(xyz_opt):
                xyz_fix = xyz_opt.copy()
                for _ in range(20):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.95
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_opt = xyz_fix
                
            if not _validate(xyz_opt):
                continue
                
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
    
    # Final validation
    if not _validate(best_xyz):
        # Fallback to a known valid config
        best_xyz = _initial_grid()
        
    return best_xyz

# EVOLVE-BLOCK-END
