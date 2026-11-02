import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import time

def _hermite_phys(n, x):
    """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    H_nm1 = np.ones_like(x)
    H_n = 2 * x
    for k in range(2, n + 1):
        H_np1 = 2 * x * H_n - 2 * (k - 1) * H_nm1
        H_nm1 = H_n
        H_n = H_np1
    return H_n

def _compute_integrals(c0, c4, c8, c12):
    # Coefficients for P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # P(x) is even.
    # We need to compute N = int P^2 exp(-x^2/2)
    # M2 = int x^2 P^2 exp(-x^2/2)
    # K = int Q^2 exp(-x^2/2) where Q = P' - x/2 P
    
    # Monomial coefficients for H_n
    max_n = 13
    coeffs = [[0.0] * (n + 1) for n in range(max_n + 1)]
    coeffs[0][0] = 1.0
    coeffs[1][1] = 2.0
    for n in range(2, max_n + 1):
        # H_n = 2x H_{n-1} - 2(n-1) H_{n-2}
        for k in range(n + 1):
            val = 0.0
            if k - 1 >= 0:
                val += 2.0 * coeffs[n-1][k-1]
            if n - 2 >= 0 and k <= n - 2:
                val -= 2.0 * (n - 1) * coeffs[n-2][k]
            coeffs[n][k] = val
            
    # P(x) coefficients
    p_coeffs = [0.0] * 13
    for k in range(13):
        p_coeffs[k] = c0 * coeffs[0][k] + c4 * coeffs[4][k] + c8 * coeffs[8][k] + c12 * coeffs[12][k]
        
    # Q(x) coefficients
    # Q = P' - x/2 P
    q_coeffs = [0.0] * 14 
    for k in range(13):
        if k + 1 < 13:
            q_coeffs[k] += (k + 1) * p_coeffs[k + 1]
        if k < 14:
            q_coeffs[k + 1] -= 0.5 * p_coeffs[k]
            
    # Precompute moments M_n = int x^n exp(-x^2/2) dx
    M_n = [0.0] * 27
    for n in range(27):
        if n % 2 == 1:
            M_n[n] = 0.0
        else:
            k = n // 2
            M_n[n] = factorial(2*k) / factorial(k) * np.sqrt(np.pi / 2.0)
            
    def calc_int(p1, p2, m):
        total = 0.0
        max_deg1 = len(p1) - 1
        max_deg2 = len(p2) - 1
        for i in range(max_deg1 + 1):
            if p1[i] == 0: continue
            for j in range(max_deg2 + 1):
                if p2[j] == 0: continue
                n = i + j + m
                if n < 27:
                    total += p1[i] * p2[j] * M_n[n]
        return total
        
    N = calc_int(p_coeffs, p_coeffs, 0)
    M2 = calc_int(p_coeffs, p_coeffs, 2)
    K = calc_int(q_coeffs, q_coeffs, 0)
    
    if N <= 0:
        return float('inf')
        
    return (M2 * K) / (N * N)

def construct():
    start_time = time.time()
    deadline = start_time + 180.0 # Safety margin
    
    # Seed
    c0_seed, c4_seed, c8_seed = -113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0
    
    # Constraint: g(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
    # c0 + 12 c4 + 1680 c8 + 665280 c12 = 0
    # c12 = -(c0 + 12 c4 + 1680 c8) / 665280
    
    def objective(vars):
        c0, c4, c8 = vars
        c12 = -(c0 + 12.0 * c4 + 1680.0 * c8) / 665280.0
        return _compute_integrals(c0, c4, c8, c12)
        
    x0 = np.array([c0_seed, c4_seed, c8_seed])
    
    best_c = x0.copy()
    best_val = objective(x0)
    
    # Multi-start strategy with L-BFGS-B
    # Generate a few random starting points and the seed
    np.random.seed(42)
    start_points = [x0]
    
    # Add some random perturbations
    for _ in range(5):
        perturbation = np.random.normal(0, 1, size=3)
        start_points.append(x0 + perturbation)
    
    # Also try some structured points
    start_points.append(np.array([-1.0, 0.0, 0.0]))
    start_points.append(np.array([0.0, 0.0, 0.0]))
    
    for sp in start_points:
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(objective, sp, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
                
        except Exception:
            continue
            
    # Final polish with high precision
    if time.time() < deadline - 10:
        try:
            res2 = minimize(objective, best_c, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10})
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x.copy()
        except Exception:
            pass
            
    return np.array(best_c)
