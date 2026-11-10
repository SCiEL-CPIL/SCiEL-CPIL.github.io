"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _compute_radius(pts):
    """Compute the maximum common radius for given sphere centers."""
    n = len(pts)
    if n < 2:
        return 0.5
    
    # Compute all pairwise distances efficiently
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Get minimum pairwise distance (excluding self)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    
    # Radius is half the min pair distance, or min wall distance
    # But wait, the definition is:
    # r = min( min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|) )
    return min(min_pair_dist / 2.0, min_wall_dist)

def _potential_and_grad(pts, power=2.0):
    """
    Compute the potential energy and its gradient for the inverse-power repulsion model.
    V = sum_{i<j} 1/d_ij^power + sum_i sum_k 1/w_ik^power
    """
    n = len(pts)
    V = 0.0
    grad = np.zeros_like(pts)
    
    # Pairwise interactions
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d2 = np.dot(diff, diff)
            if d2 < 1e-12:
                d2 = 1e-12
            d = np.sqrt(d2)
            # Term: d^{-power}
            term_val = d ** (-power)
            V += term_val
            
            # Gradient w.r.t pts[i]: -power * d^{-power-2} * diff
            # Gradient w.r.t pts[j]: +power * d^{-power-2} * diff
            coeff = -power * d ** (-power - 2.0)
            grad[i] += coeff * diff
            grad[j] -= coeff * diff # Note: diff is pts[i]-pts[j], so grad[j] gets -coeff*diff? 
            # Let's re-verify: V = (xi-xj)^2)^(-p/2). 
            # dV/dxi = -p/2 * ((xi-xj)^2)^(-p/2 - 1) * 2*(xi-xj) = -p * d^(-p-2) * (xi-xj).
            # So grad[i] += -p * d^(-p-2) * diff.
            # dV/dxj = -p * d^(-p-2) * (-diff) = p * d^(-p-2) * diff.
            # So grad[j] += p * d^(-p-2) * diff.
            # My code above: coeff = -p * d^(-p-2).
            # grad[i] += coeff * diff. Correct.
            # grad[j] -= coeff * diff. -> grad[j] += p * d^(-p-2) * diff. Correct.

    # Wall interactions
    for i in range(n):
        for k in range(3):
            # Distance to wall: 0.5 - |x_ik|
            # Let u = 0.5 - |x_ik|. We minimize 1/u^power.
            # This is only valid if u > 0. If u <= 0, we are outside.
            # We assume pts are initialized inside and constrained? 
            # L-BFGS-B doesn't enforce constraints on the potential itself, but we can add a large penalty if outside.
            # Better: Use a soft wall potential that goes to infinity at the wall.
            # V_wall = 1 / (0.5 - |x_ik|)^power.
            # This is smooth for |x_ik| < 0.5.
            
            x = pts[i, k]
            abs_x = np.abs(x)
            if abs_x >= 0.5 - 1e-6:
                # If we hit the wall, we need a strong repulsion.
                # The potential 1/(0.5-abs_x)^p blows up.
                # We should clip or handle this.
                # For gradient calculation, if we are very close, the force is huge.
                pass
            
            u = 0.5 - abs_x
            if u < 1e-6:
                u = 1e-6
                # If we are effectively at the wall, the gradient should push us in.
                # sign of x determines direction.
                # dV/dx = -dV/du * du/dx.
                # du/dx = -sign(x).
                # dV/du = -power * u^(-power-1).
                # dV/dx = -power * u^(-power-1) * (-sign(x)) = power * u^(-power-1) * sign(x).
                
            term_val = u ** (-power)
            V += term_val
            
            # Gradient
            sign_x = 1.0 if x > 0 else -1.0
            if x == 0: sign_x = 1.0 # Arbitrary, but gradient is 0 in x direction for |x| at 0? 
            # Actually d/dx |x| is undefined at 0. But we are unlikely to be exactly at 0.
            # Let's use subgradient or just assume x != 0.
            
            grad_term = power * u ** (-power - 1.0) * sign_x
            grad[i, k] += grad_term

    return V, grad

def _optimize_single_start(pts_init, time_limit=10.0):
    """Run L-BFGS-B optimization starting from pts_init."""
    start_time = time.time()
    
    # Reshape to 1D array for scipy
    x0 = pts_init.flatten()
    
    def fun(x):
        pts = x.reshape(-1, 3)
        val, _ = _potential_and_grad(pts, power=2.0)
        return val
        
    def jac(x):
        pts = x.reshape(-1, 3)
        _, grad = _potential_and_grad(pts, power=2.0)
        return grad.flatten()
        
    # Use L-BFGS-B
    try:
        res = minimize(
            fun, x0, jac=jac, method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
    except Exception as e:
        return pts_init, _compute_radius(pts_init)
        
    pts_opt = res.x.reshape(-1, 3)
    r_opt = _compute_radius(pts_opt)
    
    # Check time
    if time.time() - start_time > time_limit:
         # If it took too long, we might have wasted time, but we return the best found so far
         pass
         
    return pts_opt, r_opt

def construct():
    """Construct optimal sphere packing using gradient-based optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_pts = None
    best_r = -np.inf
    
    # Strategy: Multiple restarts with different initial configurations
    # 1. Grid-based
    # 2. Random
    
    rng = np.random.RandomState(42)
    
    # Initial Config 1: Simple Grid
    # 4x4x4 = 64 points, take 50
    m = 4
    grid_pts = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) / m - 0.5
                y = (j + 0.5) / m - 0.5
                z = (k + 0.5) / m - 0.5
                grid_pts.append([x, y, z])
    grid_pts = np.array(grid_pts)
    # Shuffle and take N
    idx = rng.permutation(len(grid_pts))
    pts_init_1 = grid_pts[idx[:N]]
    
    pts_opt, r_opt = _optimize_single_start(pts_init_1, time_limit=20.0)
    if r_opt > best_r:
        best_r = r_opt
        best_pts = pts_opt.copy()
        
    if time.time() > deadline - 20.0:
        return best_pts

    # Initial Config 2: BCC-like
    # BCC has 2 points per unit cell.
    # Let's use a 3x3x3 BCC grid = 27*2 = 54 points. Take 50.
    bcc_pts = []
    m_bcc = 3
    for i in range(m_bcc):
        for j in range(m_bcc):
            for k in range(m_bcc):
                # Corner
                x = (i + 0.5) / m_bcc - 0.5
                y = (j + 0.5) / m_bcc - 0.5
                z = (k + 0.5) / m_bcc - 0.5
                bcc_pts.append([x, y, z])
                # Body Center
                if i < m_bcc - 1 and j < m_bcc - 1 and k < m_bcc - 1:
                    bx = (i + 1.0) / m_bcc - 0.5
                    by = (j + 1.0) / m_bcc - 0.5
                    bz = (k + 1.0) / m_bcc - 0.5
                    bcc_pts.append([bx, by, bz])
    bcc_pts = np.array(bcc_pts)
    if len(bcc_pts) >= N:
        idx = rng.permutation(len(bcc_pts))
        pts_init_2 = bcc_pts[idx[:N]]
        pts_opt, r_opt = _optimize_single_start(pts_init_2, time_limit=20.0)
        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts_opt.copy()

    if time.time() > deadline - 20.0:
        return best_pts

    # Initial Config 3-5: Random
    for i in range(5):
        if time.time() > deadline - 10.0:
            break
        pts_rand = rng.uniform(-0.45, 0.45, size=(N, 3))
        # Ensure no overlaps in initial config? Not strictly necessary for L-BFGS-B if potential is smooth,
        # but very close points can cause numerical issues with 1/d^2.
        # Let's add a small jitter to ensure minimum separation in init
        # Simple check: if any dist < 0.01, re-generate or shift.
        # For now, just run it.
        
        pts_opt, r_opt = _optimize_single_start(pts_rand, time_limit=15.0)
        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts_opt.copy()
            
    # Final Polish: 
    # The L-BFGS-B result is a local minimum of the potential.
    # We can try to improve it by a few steps of exact metric optimization.
    # However, the potential minimum is usually very close to the max-min distance solution.
    # Let's do a quick local search on the best_pts using small random perturbations and exact metric evaluation.
    
    if best_pts is not None and time.time() < deadline - 5.0:
        current_pts = best_pts.copy()
        current_r = best_r
        T = 1e-4
        for _ in range(1000):
            if time.time() > deadline - 2.0:
                break
            idx = rng.randint(N)
            step = T * rng.randn(3)
            new_pts = current_pts.copy()
            new_pts[idx] += step
            new_pts[idx] = np.clip(new_pts[idx], -0.499, 0.499)
            
            new_r = _compute_radius(new_pts)
            if new_r > current_r:
                current_pts = new_pts
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_pts = current_pts.copy()
            T *= 0.99

    return best_pts
# EVOLVE-BLOCK-END
