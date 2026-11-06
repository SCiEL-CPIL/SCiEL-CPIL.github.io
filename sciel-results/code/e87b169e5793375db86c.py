"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _exact_radius(c):
    """Compute the exact packing radius for centers c."""
    c_sq = np.sum(c * c, axis=1)
    d2 = c_sq[:, None] + c_sq[None, :] - 2.0 * (c @ c.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 1e30)
    min_d2 = np.min(d2)
    min_dist = np.sqrt(min_d2)
    min_dist_to_wall = np.min(0.5 - np.abs(c))
    return min(min_dist / 2.0, min_dist_to_wall)

def _soft_min_radius_and_grad(c, beta):
    """
    Compute a soft-min surrogate for the minimum distance and its gradient.
    Uses log-sum-exp: soft_min(x) = -(1/beta) * log(sum(exp(-beta * x)))
    Here x are the distances (pairwise and to walls).
    """
    # Pairwise distances
    c_sq = np.sum(c * c, axis=1)
    d2 = c_sq[:, None] + c_sq[None, :] - 2.0 * (c @ c.T)
    np.maximum(d2, 1e-12, out=d2)
    np.fill_diagonal(d2, 1e30)
    d = np.sqrt(d2)
    
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]
    
    # Wall distances: 0.5 - |c_ik|
    wall_dists = 0.5 - np.abs(c)  # shape (N, 3)
    d_walls = wall_dists.ravel()
    
    # All distances
    all_d = np.concatenate([d_pairs, d_walls])
    
    # Soft-min: -1/beta * log(sum(exp(-beta * d)))
    # Numerically stable: subtract max
    max_d = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - max_d))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = -(1.0 / beta) * (log_sum - beta * max_d)
    
    # Gradient with respect to c
    # d(soft_min)/d(c_ik) = (1/beta) * sum_j [w_j * d(d_j)/d(c_ik)]
    # where w_j = exp(-beta*d_j) / sum(exp(-beta*d_j))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros_like(c)
    
    # Pairwise part
    w_pairs = weights[:len(d_pairs)]
    # For each pair (i,j), d/d(c_i) d_ij = (c_i - c_j)/d_ij
    # d/d(c_j) d_ij = (c_j - c_i)/d_ij
    # We need to accumulate gradients
    # iu[0] and iu[1] give the indices
    ii = iu[0]
    jj = iu[1]
    
    # grad contribution from pair (i,j): w_ij * (c_i - c_j) / d_ij for c_i
    # and w_ij * (c_j - c_i) / d_ij for c_j
    # Using np.add.at for accumulation
    diff = c[ii] - c[jj]  # shape (len, 3)
    inv_d = w_pairs[:, None] / d_pairs[:, None]  # shape (len, 3)
    contrib = diff * inv_d  # shape (len, 3)
    
    # Add to grad[i] and subtract from grad[j]
    np.add.at(grad, ii, contrib)
    np.add.at(grad, jj, -contrib)
    
    # Wall part
    w_walls = weights[len(d_pairs):]
    # Wall distance for coordinate k of sphere i: 0.5 - |c_ik|
    # d/d(c_ik) (0.5 - |c_ik|) = -sign(c_ik)
    # So grad contribution: w * (-sign(c_ik))
    wall_grad = w_walls.reshape(N, 3) * (-np.sign(c))
    grad += wall_grad
    
    return soft_min_val, grad

def _init_grid():
    """Create a grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]

def _init_bcc():
    """Create a BCC-like initialization."""
    # 4x4x4 grid of unit cells, each with corner + body center
    # But we need exactly 50 points
    # Use a 3x4x5 grid of points
    nx, ny, nz = 3, 4, 5
    x = (np.arange(nx) + 0.5) / nx - 0.5
    y = (np.arange(ny) + 0.5) / ny - 0.5
    z = (np.arange(nz) + 0.5) / nz - 0.5
    pts = []
    for zi in z:
        for yi in y:
            for xi in x:
                pts.append([xi, yi, zi])
    pts = np.array(pts[:N])
    # Add some random perturbation
    pts += np.random.normal(0, 0.01, size=pts.shape)
    return pts

def _init_random():
    """Random initialization inside the cube."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def compute_energy(c):
        """
        Computes the energy to be minimized.
        Minimizing sum(1/d_ij^2) encourages maximizing the minimum distance.
        Boundary penalty keeps spheres inside [-0.5, 0.5].
        """
        c_sq = np.sum(c * c, axis=1)
        d2 = c_sq[:, np.newaxis] + c_sq[np.newaxis, :] - 2.0 * (c @ c.T)
        np.maximum(d2, 1e-12, out=d2)
        np.fill_diagonal(d2, 1.0)
        e_pair = np.sum(1.0 / d2) - N
        abs_c = np.abs(c)
        violations = np.maximum(0.0, abs_c - 0.5)
        e_boundary = np.sum(violations * violations) * 100.0
        return e_pair + e_boundary

    def sa_run(init, time_budget):
        """
        Runs Simulated Annealing starting from init.
        Returns the best center found.
        """
        current_c = init.copy()
        current_e = compute_energy(current_c)
        best_c = current_c.copy()
        best_e = current_e
        best_r = _exact_radius(best_c)
        
        T = 0.1
        cooling_rate = 0.99995
        min_T = 1e-8
        step_size = 0.03
        
        while time.time() < deadline - 10.0 and T > min_T:
            # Propose move
            i = np.random.randint(N)
            delta = np.random.normal(0, step_size, size=3)
            new_c = current_c.copy()
            new_c[i] += delta
            
            new_e = compute_energy(new_c)
            delta_e = new_e - current_e
            
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                current_c = new_c
                current_e = new_e
                
                if current_e < best_e:
                    best_c = current_c.copy()
                    best_e = current_e
                    r = _exact_radius(best_c)
                    if r > best_r:
                        best_r = r
            
            T *= cooling_rate
            # Occasionally adjust step size
            if np.random.random() < 0.01:
                step_size *= np.random.uniform(0.8, 1.2)
                step_size = np.clip(step_size, 0.005, 0.1)
            
        return best_c, best_r

    # Initialize multiple configurations
    inits = []
    inits.append(_init_grid())
    inits.append(_init_bcc())
    inits.append(_init_random())
    inits.append(_init_random())
    inits.append(_init_random())
    
    best_overall_c = inits[0].copy()
    best_overall_r = _exact_radius(best_overall_c)
    
    # Run SA on each init
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        c, r = sa_run(init, 0)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_c = c.copy()
            
    # Final Polish using Nelder-Mead
    if time.time() < deadline - 5.0:
        def obj(x):
            c = x.reshape(N, 3)
            return compute_energy(c)
        
        x0 = best_overall_c.flatten()
        
        try:
            res = minimize(obj, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
            if res.success or time.time() < deadline - 2.0:
                c_polished = res.x.reshape(N, 3)
                r_polished = _exact_radius(c_polished)
                if r_polished > best_overall_r:
                    best_overall_c = c_polished
                    best_overall_r = r_polished
        except Exception:
            pass

    # One more polish with tighter tolerance if time permits
    if time.time() < deadline - 2.0:
        def obj2(x):
            c = x.reshape(N, 3)
            return compute_energy(c)
        
        x0 = best_overall_c.flatten()
        
        try:
            res = minimize(obj2, x0, method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
            c_polished = res.x.reshape(N, 3)
            r_polished = _exact_radius(c_polished)
            if r_polished > best_overall_r:
                best_overall_c = c_polished
                best_overall_r = r_polished
        except Exception:
            pass

    return best_overall_c

# EVOLVE-BLOCK-END
