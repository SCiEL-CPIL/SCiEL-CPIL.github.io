# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline management
_START_TIME = time.time()
_TIMEOUT = 200.0  # Leave 10s margin for the 210s limit

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product for g(x) = P(x) exp(-x^2/2).
    P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    Constraint: g(0) = 0 => a12 = -(a0 + 3*a4 + 105*a8)/945
    
    Returns sqrt(<x^2>/<1> * <xi^2>/<1>)
    """
    # Coefficients of P(x) in power basis x^(2k)
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # p_k is coeff of x^(2k) in P(x)
    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    
    # Precomputed integrals I[n] = int x^(2n) exp(-x^2) dx
    # I[n] = (2n-1)!! * sqrt(pi) / 2^n
    # We need I[0] to I[12] (since max power in |P|^2 is x^24, i.e., n=12)
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        I[n] = I[n-1] * (2*n - 1) / 2.0
        
    # m[k] = int x^(2k) |P(x)|^2 exp(-x^2) dx
    # |P|^2 = sum_i sum_j p_i p_j x^(2(i+j))
    # m[k] = sum_i sum_j p_i p_j I[k+i+j]
    
    # Vectorized computation of m
    # p is shape (7,). We need sum over i,j for k=0,1,2
    # Let's just loop, it's small enough
    m0 = 0.0
    m1 = 0.0
    m2 = 0.0
    
    for i in range(7):
        pi = p[i]
        if pi == 0: continue
        for j in range(7):
            pp = pi * p[j]
            if pp == 0: continue
            idx0 = i + j
            idx1 = i + j + 1
            idx2 = i + j + 2
            if idx0 < 13: m0 += pp * I[idx0]
            if idx1 < 13: m1 += pp * I[idx1]
            if idx2 < 13: m2 += pp * I[idx2]
            
    if m0 <= 0 or not np.isfinite(m0):
        return 1e18
        
    var_x = m1 / m0
    if var_x <= 0:
        return 1e18

    # Fourier transform
    # G(xi) = sqrt(2*pi) P(-i xi) exp(-xi^2/2)
    # P(-i xi) = sum p_k (-i)^(2k) xi^(2k) = sum p_k (-1)^k xi^(2k)
    # Let q_k = p_k * (-1)^k
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # n[k] = int xi^(2k) |G(xi)|^2 dxi
    # |G|^2 = 2*pi |P(-i xi)|^2 exp(-xi^2)
    # n[k] = 2*pi sum_i sum_j q_i q_j I[k+i+j]
    
    n0 = 0.0
    n1 = 0.0
    
    for i in range(7):
        qi = q[i]
        if qi == 0: continue
        for j in range(7):
            qq = qi * q[j]
            if qq == 0: continue
            idx0 = i + j
            idx1 = i + j + 1
            if idx0 < 13: n0 += qq * I[idx0]
            if idx1 < 13: n1 += qq * I[idx1]
            
    n0 *= 2.0 * np.pi
    n1 *= 2.0 * np.pi
    
    if n0 <= 0 or not np.isfinite(n0):
        return 1e18
        
    var_xi = n1 / n0
    if var_xi <= 0:
        return 1e18

    return np.sqrt(var_x * var_xi)

def _objective_sphere(uv):
    """
    Objective function for Nelder-Mead.
    Input: u, v in R^2
    Maps to sphere:
    theta1 = u
    theta2 = v (clipped)
    Returns bound.
    """
    t1 = uv[0]
    t2 = uv[1]
    
    # Clip t2 to avoid singularities
    t2 = np.clip(t2, 1e-6, np.pi - 1e-6)
    
    a0 = np.sin(t2) * np.cos(t1)
    a4 = np.sin(t2) * np.sin(t1)
    a8 = np.cos(t2)
    
    return _compute_bound(a0, a4, a8)

def construct():
    """
    Find coefficients (a0, a4, a8) minimizing the uncertainty bound.
    Uses multi-start Nelder-Mead on the sphere.
    """
    # Seed values
    seed = np.array([-1.13, 0.04, 0.000308642])
    norm_seed = np.linalg.norm(seed)
    seed_dir = seed / norm_seed
    
    # Convert seed to angles
    theta1_seed = np.arctan2(seed_dir[1], seed_dir[0])
    theta2_seed = np.arccos(np.clip(seed_dir[2], -1.0, 1.0))
    
    best_bound = _compute_bound(*seed_dir)
    best_uv = np.array([theta1_seed, theta2_seed])
    
    # Generate random starting points
    rng = np.random.default_rng(42)
    
    # Add seed and some fixed good points if known, plus randoms
    # Let's sample 100 random points on the sphere
    # z ~ U(-1,1), phi ~ U(0, 2pi)
    # t2 = acos(z), t1 = phi
    
    starts = [best_uv.copy()]
    
    for _ in range(100):
        z = rng.uniform(-1, 1)
        phi = rng.uniform(0, 2 * np.pi)
        t2 = np.arccos(z)
        t1 = phi
        starts.append(np.array([t1, t2]))
        
    # Also add points near the seed
    for _ in range(20):
        noise = rng.normal(0, 0.1, 2)
        starts.append(best_uv + noise)
        
    # Run Nelder-Mead for each start
    # We need to be careful with time. 
    # Nelder-Mead is fast for 2D. 120 starts * ~100 iterations * ~10us/eval = 1.2s
    # This is very safe.
    
    options = {
        'xatol': 1e-10,
        'fatol': 1e-10,
        'maxiter': 200,
        'maxfev': 200
    }
    
    for i, start in enumerate(starts):
        if time.time() - _START_TIME > _TIMEOUT - 10:
            break
            
        # Wrap t1 to [-pi, pi]
        start[0] = (start[0] + np.pi) % (2 * np.pi) - np.pi
        start[1] = np.clip(start[1], 1e-6, np.pi - 1e-6)
        
        try:
            res = minimize(_objective_sphere, start, method='Nelder-Mead', options=options)
            if res.success or np.isfinite(res.fun):
                if res.fun < best_bound:
                    best_bound = res.fun
                    best_uv = res.x
        except Exception:
            pass
            
    # Final refinement with tighter tolerances if time permits
    if time.time() - _START_TIME < _TIMEOUT - 5:
        options_fine = {
            'xatol': 1e-12,
            'fatol': 1e-12,
            'maxiter': 500,
            'maxfev': 500
        }
        try:
            res = minimize(_objective_sphere, best_uv, method='Nelder-Mead', options=options_fine)
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_uv = res.x
        except Exception:
            pass
            
    # Convert best_uv back to coefficients
    t1 = best_uv[0]
    t2 = best_uv[1]
    
    a0 = np.sin(t2) * np.cos(t1)
    a4 = np.sin(t2) * np.sin(t1)
    a8 = np.cos(t2)
    
    result = np.array([a0, a4, a8])
    
    # Verify
    final_bound = _compute_bound(*result)
    if not np.isfinite(final_bound) or final_bound > best_bound + 1e-9:
        # Fallback to seed if something went wrong
        return seed_dir
        
    return result
# EVOLVE-BLOCK-END
