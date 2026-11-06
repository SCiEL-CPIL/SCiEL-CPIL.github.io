"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _radius_for_centers(centers):
    """Compute the radius: min of half pairwise distances and wall distances."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances: min over each coordinate of (0.5 - |c|)
    wall = 0.5 - np.abs(centers)
    min_wall = np.min(wall)
    if min_wall < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(min_wall, min_pair)

def _energy_and_grad(x):
    """
    Compute a smooth energy function and its gradient.
    Energy is designed to be minimized. Lower energy corresponds to larger minimum radius.
    We use a repulsive potential between spheres and a repulsive potential from walls.
    """
    centers = x.reshape(N, 3)
    
    # 1. Pairwise repulsion
    # Use a power law potential: V = 1/d^p. As d decreases, V increases sharply.
    # We want to maximize min distance, so minimizing sum of 1/d^p pushes points apart.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero
    
    # Use a high power to emphasize close pairs (approximate max-min behavior)
    p = 4.0 
    # dist = sqrt(dist_sq)
    # V_pair = 1 / (dist^p) = 1 / (dist_sq)^(p/2)
    # Let alpha = p/2 = 2
    alpha = p / 2.0
    V_pair = np.sum(1.0 / (dist_sq ** alpha))
    
    # Gradient of V_pair w.r.t centers
    # d/dx_i (1/d^p) = -p/d^(p+2) * (x_i - x_j)
    # Here d^2 = dist_sq. d/dx_i (dist_sq^-alpha) = -alpha * dist_sq^(-alpha-1) * 2*(x_i - x_j)
    # Sum over j != i
    # grad_i = -2 * alpha * sum_j (x_i - x_j) * dist_sq_ij^(-alpha - 1)
    
    # We compute this vectorized
    # diff[i,j] = x_i - x_j
    # factor[i,j] = -2 * alpha * dist_sq[i,j]^(-alpha - 1)
    factor = -2.0 * alpha * (dist_sq ** (-alpha - 1.0))
    np.fill_diagonal(factor, 0.0)
    
    grad_pair = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    # 2. Wall repulsion
    # We want to keep centers away from walls at +/- 0.5.
    # Let boundary be B = 0.5.
    # Distance to wall for coord k: w_k = B - |x_k|
    # We want to maximize min w_k.
    # Use a potential that increases as w_k decreases.
    # V_wall = sum (1/w_k)^m for w_k > 0. If w_k <= 0, add a large penalty.
    
    # To handle the absolute value smoothly, we can use softplus or just check signs.
    # Since we start inside and use bounds, we can assume centers stay roughly inside.
    # However, L-BFGS-B uses bounds, so it shouldn't go outside.
    # But the gradient should still push away from boundaries.
    
    # Let's use a potential based on distance to boundary.
    # d_wall = 0.5 - |x|
    # If d_wall < 0, penalty is huge.
    # V_wall = sum (1 / max(d_wall, eps))^m
    
    m = 4.0
    eps = 1e-6
    
    abs_centers = np.abs(centers)
    d_wall = 0.5 - abs_centers
    
    # Mask for inside
    valid = d_wall > 0
    d_safe = np.where(valid, d_wall, eps)
    
    V_wall = np.sum(1.0 / (d_safe ** m))
    
    # Gradient of V_wall
    # d/dx (1/(0.5-|x|)^m)
    # Let u = 0.5 - |x|. du/dx = -sign(x).
    # d/dx (u^-m) = -m * u^(-m-1) * du/dx = -m * u^(-m-1) * (-sign(x)) = m * sign(x) * u^(-m-1)
    
    signs = np.sign(centers)
    # If center is 0, sign is 0, gradient is 0. This is a saddle point, but rare.
    # We can handle 0 by perturbing or just letting it be.
    
    grad_wall = m * signs * (d_safe ** (-m - 1.0))
    
    # If d_wall <= 0, we are outside. The gradient should push inward strongly.
    # The formula above assumes d_wall > 0.
    # If outside, d_safe = eps. grad = m * sign(x) * eps^(-m-1).
    # This is a huge force pushing towards the boundary (and through it?).
    # Actually, if x > 0.5, sign(x)=1. grad > 0. This pushes x further up. Wrong.
    # We want to push x down.
    # If x > 0.5, we want force < 0.
    # The potential 1/(0.5-x)^m is only defined for x < 0.5.
    # For x > 0.5, we should use a different potential or just rely on bounds.
    # L-BFGS-B respects bounds, so it won't step outside.
    # However, the gradient at the boundary should point inward.
    
    # Let's refine the wall potential to be symmetric and smooth around the boundary?
    # No, the boundary is a hard constraint.
    # Better approach: Use a potential that grows as we approach the boundary from inside.
    # V = sum (1 / (0.5 - |x|)^m)
    # This works for |x| < 0.5.
    # If L-BFGS-B tries to go outside, it clips.
    # The gradient calculated above is for |x| < 0.5.
    # If |x| is very close to 0.5, the gradient is large and points inward.
    # e.g., x = 0.49. sign=1. d_wall = 0.01. grad = m * 1 * 0.01^(-m-1) > 0.
    # This pushes x to increase? No, gradient ascent increases x. Gradient descent decreases x.
    # minimize(V) -> x_new = x - lr * grad.
    # If grad > 0, x decreases. Correct.
    
    # What if x = -0.49? sign=-1. d_wall = 0.01. grad = m * (-1) * ... < 0.
    # x_new = x - lr * (negative) -> x increases. Correct (towards 0).
    
    total_energy = V_pair + V_wall
    total_grad = (grad_pair + grad_wall).flatten()
    
    return total_energy, total_grad

def _make_bcc_initial(seed=42):
    """
    Generate initial configuration based on a BCC superlattice.
    BCC has higher density than Simple Cubic.
    We create a 4x4x4 grid of BCC unit cells.
    Each cell has 2 atoms: one at corner, one at body center.
    Total points in 4x4x4 BCC grid: 4*4*4 * 2 = 128 points.
    We select 50 of these, prioritizing those with larger local spacing or just random selection.
    To ensure good coverage, we will take a subset of the BCC lattice points.
    """
    rng = np.random.RandomState(seed)
    
    # Define BCC lattice in a unit cube [0,1]^3
    # Scale to [-0.5, 0.5]^3 later
    
    m = 4 # 4x4x4 supercells
    points = []
    
    # Lattice constant a = 1/m
    a = 1.0 / m
    
    for iz in range(m):
        for iy in range(m):
            for ix in range(m):
                # Corner atom
                x = ix * a
                y = iy * a
                z = iz * a
                points.append([x, y, z])
                
                # Body center atom
                x_c = (ix + 0.5) * a
                y_c = (iy + 0.5) * a
                z_c = (iz + 0.5) * a
                points.append([x_c, y_c, z_c])
                
    points = np.array(points)
    
    # Shift to center at 0: [-0.5, 0.5]
    points -= 0.5
    
    # We have 128 points. We need 50.
    # Strategy: Randomly select 50, but add jitter to break symmetry and avoid exact lattice degeneracies.
    idx = rng.choice(len(points), size=N, replace=False)
    centers = points[idx].copy()
    
    # Add small jitter to help escape local minima of the lattice
    centers += rng.uniform(-0.01, 0.01, size=centers.shape)
    
    # Clip to valid range
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def _make_fcc_initial(seed=42):
    """
    Generate initial configuration based on an FCC superlattice.
    FCC has the same density as BCC (0.74) but different geometry.
    """
    rng = np.random.RandomState(seed)
    
    m = 4
    a = 1.0 / m
    points = []
    
    for iz in range(m):
        for iy in range(m):
            for ix in range(m):
                # Corner
                points.append([ix*a, iy*a, iz*a])
                
                # Face centers
                # xy face
                points.append([(ix+0.5)*a, (iy+0.5)*a, iz*a])
                # xz face
                points.append([(ix+0.5)*a, iy*a, (iz+0.5)*a])
                # yz face
                points.append([ix*a, (iy+0.5)*a, (iz+0.5)*a])
                
    points = np.array(points)
    points -= 0.5
    
    idx = rng.choice(len(points), size=N, replace=False)
    centers = points[idx].copy()
    centers += rng.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def _make_random_initial(seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def _optimize_smooth(centers, time_budget=10.0):
    """
    Optimize the smooth energy function.
    """
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.4999, 0.4999)] * (N * 3)
    
    best_x = x0
    best_radius = _radius_for_centers(x0.reshape(N, 3))
    
    # Try multiple L-BFGS runs with different tolerances/iterations
    # L-BFGS-B is good for bound-constrained problems
    
    options = {
        'maxiter': 2000,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'maxfun': 5000
    }
    
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True, # We provide the gradient
            bounds=bounds,
            options=options
        )
        
        if res.success or res.fun < _energy_and_grad(x0)[0]:
            new_centers = res.x.reshape(N, 3)
            new_radius = _radius_for_centers(new_centers)
            if new_radius > best_radius:
                best_x = res.x
                best_radius = new_radius
    except Exception as e:
        pass
        
    return best_x.reshape(N, 3), best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Initializations: BCC, FCC, Random
    initializations = [
        ("bcc", _make_bcc_initial(42)),
        ("bcc2", _make_bcc_initial(123)),
        ("fcc", _make_fcc_initial(42)),
        ("rand", _make_random_initial(1))
    ]
    
    for name, init_centers in initializations:
        if time.time() > deadline - 20.0:
            break
            
        current = init_centers.copy()
        current_radius = _radius_for_centers(current)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = current.copy()
            
        # Optimize
        opt_centers, opt_radius = _optimize_smooth(current, time_budget=30.0)
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers.copy()
            
    if best_centers is None:
        # Fallback
        best_centers = _make_bcc_initial(42)
        
    # Final polish: Run a few more optimization steps on the best result
    if time.time() < deadline - 5.0:
        # Re-optimize the best result with tighter tolerances
        final_centers, final_radius = _optimize_smooth(best_centers, time_budget=15.0)
        if final_radius > best_radius:
            best_centers = final_centers
            best_radius = final_radius
            
    # Ensure validity
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
