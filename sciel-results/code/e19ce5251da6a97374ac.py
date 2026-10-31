import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 200
_START_TIME = time.time()

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Hermite polynomial coefficients in power basis x^{2k}
H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

# a12 = -(a0 + 3*a4 + 105*a8) / 945
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2))

def _get_p(a):
    return A_mat @ a

def _logsign_to_linear(z):
    """Convert log-magnitude + sign parameters to linear coefficients.
    z = [log|a0|, log|a4|, log|a8|, s0, s4, s8]
    where a_i = sign(s_i) * exp(log|a_i|)
    """
    a = np.zeros(3)
    a[0] = np.tanh(z[3]) * np.exp(z[0])
    a[1] = np.tanh(z[4]) * np.exp(z[1])
    a[2] = np.tanh(z[5]) * np.exp(z[2])
    return a

def _linear_to_logsign(a):
    """Convert linear coefficients to log-magnitude + sign parameters."""
    z = np.zeros(6)
    for i in range(3):
        if abs(a[i]) < 1e-30:
            z[i] = -30.0
            z[i+3] = 0.0
        else:
            z[i] = np.log(abs(a[i]))
            z[i+3] = np.sign(a[i])
    return z

def _compute_bound(a):
    """Compute the uncertainty bound for given coefficients a = [a0, a4, a8]."""
    p = _get_p(a)
    
    # q_k = p_k * (-1)^k
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0 and qi == 0:
            continue
        for j in range(7):
            Mx0 += pi * p[j] * I[i+j]
            Mx1 += pi * p[j] * I[i+j+1]
            Mxi0 += qi * q[j] * I[i+j]
            Mxi1 += qi * q[j] * I[i+j+1]
    
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18
        
    return bound

def _compute_bound_logsign(z):
    """Compute the uncertainty bound in log-sign parameter space."""
    a = _logsign_to_linear(z)
    return _compute_bound(a)

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound and its analytic gradient.
    a: array of shape (3,)
    Returns: (bound, grad)
    """
    p = _get_p(a)
    
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0 and qi == 0:
            continue
        for j in range(7):
            Mx0 += pi * p[j] * I[i+j]
            Mx1 += pi * p[j] * I[i+j+1]
            Mxi0 += qi * q[j] * I[i+j]
            Mxi1 += qi * q[j] * I[i+j+1]
            
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient computation
    Sx = np.zeros((2, 7))
    Sxi = np.zeros((2, 7))
    
    for n in range(2):
        for i in range(7):
            Sx[n, i] = np.dot(p, I[i+n:i+n+7])
            Sxi[n, i] = np.dot(q, I[i+n:i+n+7])
            
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])
            
    dvx = np.zeros(3)
    dvxi = np.zeros(3)
    
    for m in range(3):
        dvx[m] = (dMx[1, m]*Mx0 - Mx1*dMx[0, m]) / (Mx0*Mx0)
        dvxi[m] = (dMxi[1, m]*Mxi0 - Mxi1*dMxi[0, m]) / (Mxi0*Mxi0)
        
    grad = np.zeros(3)
    factor = 1.0 / (2.0 * bound)
    for m in range(3):
        grad[m] = factor * (vxi*dvx[m] + vx*dvxi[m])
        
    return bound, grad

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val = _compute_bound(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 10  # seconds
    
    # Convert seed to log-sign space
    seed_z = _linear_to_logsign(seed_params)
    
    # Define bounds in log-sign space
    # log|a0|: -5 to 2 (a0 from ~0.007 to ~7.4)
    # log|a4|: -8 to 2 (a4 from ~0.00000001 to ~7.4)
    # log|a8|: -15 to 0 (a8 from ~3e-7 to 1)
    # signs: -1 to 1 (tanh maps to (-1,1))
    z_bounds = [(-5.0, 2.0), (-8.0, 2.0), (-15.0, 0.0),
                (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    
    # Differential evolution in log-sign space
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = differential_evolution(_compute_bound_logsign, z_bounds,
                                         maxiter=300, popsize=40,
                                         tol=1e-10, atol=1e-12,
                                         seed=42, init='latinhypercube',
                                         mutation=(0.5, 1.0), recombination=0.7,
                                         polish=True, updating='deferred')
            
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = _logsign_to_linear(res.x)
        except Exception:
            pass
    
    # Local polish in log-sign space
    if time.time() < _DEADLINE - deadline_margin:
        try:
            z_best = _linear_to_logsign(best_params)
            res = minimize(_compute_bound_logsign, z_best, method='L-BFGS-B',
                          bounds=z_bounds,
                          options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = _logsign_to_linear(res.x)
        except Exception:
            pass
    
    # Multiple random restarts in log-sign space
    if time.time() < _DEADLINE - deadline_margin:
        rng = np.random.default_rng(123)
        
        for _ in range(30):
            if time.time() > _DEADLINE - deadline_margin:
                break
                
            z_start = np.array([
                rng.uniform(z_bounds[0][0], z_bounds[0][1]),
                rng.uniform(z_bounds[1][0], z_bounds[1][1]),
                rng.uniform(z_bounds[2][0], z_bounds[2][1]),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0)
            ])
            
            try:
                res = minimize(_compute_bound_logsign, z_start, method='L-BFGS-B',
                              bounds=z_bounds,
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = res.fun
                    best_params = _logsign_to_linear(res.x)
            except Exception:
                pass
    
    # Final high-precision polish
    if time.time() < _DEADLINE - deadline_margin:
        try:
            z_best = _linear_to_logsign(best_params)
            res = minimize(_compute_bound_logsign, z_best, method='L-BFGS-B',
                          bounds=z_bounds,
                          options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})
            
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = _logsign_to_linear(res.x)
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
