import numpy as np
from scipy.optimize import minimize
import time

# Global deadline to ensure we return before the 210s limit
# We use a safety margin of 20 seconds
_START_TIME = time.time()
_DEADLINE = _START_TIME + 190

def _fact_ratio(n):
    """
    Compute (2n)! / (4^n * n!) * sqrt(pi)
    This is the integral of x^(2n) * exp(-x^2) dx over R.
    """
    if n < 0:
        return 0.0
    # Using log to avoid overflow for large n, though n is small here (max ~12)
    # (2n)! = prod_{k=1}^{2n} k
    # 4^n n! = 4^n * prod_{k=1}^n k
    # Result = prod_{k=1}^{2n} k / (4^n * prod_{k=1}^n k) * sqrt(pi)
    # = (prod_{k=n+1}^{2n} k / 4^n) * sqrt(pi)
    # Let's just use math.factorial for simplicity since n <= 12
    import math
    if n == 0:
        return np.sqrt(np.pi)
    num = math.factorial(2 * n)
    den = (4 ** n) * math.factorial(n)
    return (num / den) * np.sqrt(np.pi)

# Precompute integrals I_n = int x^(2n) e^(-x^2) dx for n=0..13
# Max degree of P(x)^2 is 24 (since P is degree 12), so we need I_0 to I_12
# Wait, P(x) is degree 12. P(x)^2 is degree 24.
# We need moments m_k = int x^(2k) |P|^2 e^(-x^2) dx.
# |P|^2 = sum p_i p_j x^(2i+2j).
# m_k = sum p_i p_j int x^(2k+2i+2j) e^(-x^2) dx = sum p_i p_j I_{k+i+j}.
# Max k=2 (for x^2 moment), max i=6, max j=6.
# Max index = 2 + 6 + 6 = 14.
# So we need I_0 to I_14.
_I_TABLE = np.array([_fact_ratio(n) for n in range(15)])

def _compute_bound_and_grad(params):
    """
    Compute the uncertainty bound sqrt(<x^2><xi^2>) and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 is determined by g(0)=0.
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    # Hermite polynomials at 0 (Physicists' convention):
    # H0(0) = 1
    # H4(0) = 12
    # H8(0) = 1680
    # H12(0) = 665280
    # g(0) = a0*1 + a4*12 + a8*1680 + a12*665280 = 0
    # a12 = -(a0 + 12*a4 + 1680*a8) / 665280
    
    # Note: The inspiration used a different normalization/convention for H_n(0)
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945.
    # Let's check the convention.
    # Physicists: H_n(x) = (-1)^n e^(x^2) d^n/dx^n e^(-x^2)
    # H0 = 1
    # H1 = 2x
    # H2 = 4x^2 - 2
    # H3 = 8x^3 - 12x
    # H4 = 16x^4 - 48x^2 + 12 -> H4(0) = 12.
    # The inspiration used "Probabilists" or a scaled version?
    # Probabilists He_n: He_4(x) = x^4 - 6x^2 + 3 -> He_4(0)=3.
    # The task says "Hermite-polynomial". Usually implies Physicists in this context unless specified.
    # However, the coefficients of the polynomial basis don't change the minimization landscape,
    # they just scale the parameters.
    # Let's stick to the Physicists convention as it is standard in QM uncertainty contexts,
    # BUT we must be consistent with the polynomial expansion.
    
    # Let's define P(x) = sum c_n H_n(x).
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 3072x^6 + 13440x^4 - 13440x^2 + 1680
    # H12(x) = 4096x^12 - 98304x^10 + 1290240x^8 - 6451200x^6 + 12902400x^4 - 6451200x^2 + 665280
    
    # Coefficients of x^(2k) in H_n(x):
    # H0: [1]
    # H4: [12, -48, 16] -> p0+=12, p1+=-48, p2+=16
    # H8: [1680, -13440, 13440, -3072, 256] -> p0+=1680, p1+=-13440, p2+=13440, p3+=-3072, p4+=256
    # H12: [665280, -6451200, 12902400, -6451200, 1290240, -98304, 4096]
    #       p0+=665280, p1+=-6451200, p2+=12902400, p3+=-6451200, p4+=1290240, p5+=-98304, p6+=4096
    
    # Let's build the polynomial coefficients p[k] for x^(2k)
    p = np.zeros(7)
    
    # H0
    p[0] += a0
    
    # H4
    p[0] += 12.0 * a4
    p[1] += -48.0 * a4
    p[2] += 16.0 * a4
    
    # H8
    p[0] += 1680.0 * a8
    p[1] += -13440.0 * a8
    p[2] += 13440.0 * a8
    p[3] += -3072.0 * a8
    p[4] += 256.0 * a8
    
    # H12
    # a12 = -(a0 + 12*a4 + 1680*a8) / 665280
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    
    p[0] += 665280.0 * a12
    p[1] += -6451200.0 * a12
    p[2] += 12902400.0 * a12
    p[3] += -6451200.0 * a12
    p[4] += 1290240.0 * a12
    p[5] += -98304.0 * a12
    p[6] += 4096.0 * a12
    
    # Compute spatial moments m_k = int x^(2k) |P(x)|^2 e^(-x^2) dx
    # |P|^2 = sum_i sum_j p_i p_j x^(2i+2j)
    # m_k = sum_i sum_j p_i p_j I_{k+i+j}
    
    m = np.zeros(3) # m0, m2 (k=0,1), we don't need m4 for the product, but let's compute up to k=2 just in case? No, only m0, m2 needed.
    for i in range(7):
        if p[i] == 0: continue
        for j in range(7):
            if p[j] == 0: continue
            pp = p[i] * p[j]
            for k in range(2): # k=0 for m0, k=1 for m2
                idx = k + i + j
                if idx < 15:
                    m[k] += pp * _I_TABLE[idx]
    
    # Compute frequency moments n_k
    # G(xi) = sqrt(2*pi) * P(-i*xi) * e^(-xi^2/2)
    # P(-i*xi) = sum p_k (-i)^(2k) xi^(2k) = sum p_k (-1)^k xi^(2k)
    # Let q_k = p_k * (-1)^k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0)**k)
    
    # |G|^2 = 2*pi * |Q(xi)|^2 * e^(-xi^2)
    # n_k = 2*pi * sum_i sum_j q_i q_j I_{k+i+j}
    
    n = np.zeros(2)
    for i in range(7):
        if q[i] == 0: continue
        for j in range(7):
            if q[j] == 0: continue
            qq = q[i] * q[j]
            for k in range(2):
                idx = k + i + j
                if idx < 15:
                    n[k] += 2.0 * np.pi * qq * _I_TABLE[idx]
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Analytic Gradient
    # d(bound)/da_c = (1/(2*bound)) * (vF * d(vf)/da_c + vf * d(vF)/da_c)
    # d(vf)/da_c = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/da_c = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    
    # Derivatives of a12 with respect to a0, a4, a8
    da12_da0 = -1.0 / 665280.0
    da12_da4 = -12.0 / 665280.0
    da12_da8 = -1680.0 / 665280.0
    da12_vec = np.array([da12_da0, da12_da4, da12_da8])
    
    grad = np.zeros(3)
    
    # Precompute derivatives of p_k with respect to a_c
    # p_k = sum c_n * coeff_{n,k}
    # coeff_{0,0}=1, others 0
    # coeff_{4,0}=12, coeff_{4,1}=-48, coeff_{4,2}=16
    # coeff_{8,0}=1680, coeff_{8,1}=-13440, coeff_{8,2}=13440, coeff_{8,3}=-3072, coeff_{8,4}=256
    # coeff_{12,0}=665280, coeff_{12,1}=-6451200, coeff_{12,2}=12902400, coeff_{12,3}=-6451200, coeff_{12,4}=1290240, coeff_{12,5}=-98304, coeff_{12,6}=4096
    
    coeff_matrix = np.zeros((4, 7)) # rows: 0,4,8,12; cols: 0..6
    coeff_matrix[0, 0] = 1.0
    coeff_matrix[1, 0] = 12.0; coeff_matrix[1, 1] = -48.0; coeff_matrix[1, 2] = 16.0
    coeff_matrix[2, 0] = 1680.0; coeff_matrix[2, 1] = -13440.0; coeff_matrix[2, 2] = 13440.0; coeff_matrix[2, 3] = -3072.0; coeff_matrix[2, 4] = 256.0
    coeff_matrix[3, 0] = 665280.0; coeff_matrix[3, 1] = -6451200.0; coeff_matrix[3, 2] = 12902400.0; coeff_matrix[3, 3] = -6451200.0; coeff_matrix[3, 4] = 1290240.0; coeff_matrix[3, 5] = -98304.0; coeff_matrix[3, 6] = 4096.0
    
    # For each c in {0, 4, 8}:
    # dp_k/da_c = coeff_matrix[c_idx, k] + (da12/da_c) * coeff_matrix[3, k]
    
    for c_idx in range(3):
        dp = np.zeros(7)
        for k in range(7):
            dp[k] = coeff_matrix[c_idx, k] + da12_vec[c_idx] * coeff_matrix[3, k]
        
        # dm_k/da_c = 2 * sum_i dp_i * sum_j p_j I_{k+i+j}
        dm = np.zeros(2)
        for i in range(7):
            if dp[i] == 0: continue
            for k in range(2):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 15:
                        s += p[j] * _I_TABLE[idx]
                dm[k] += 2.0 * dp[i] * s
        
        # dq_k/da_c = dp_k * (-1)^k
        dq = np.zeros(7)
        for k in range(7):
            dq[k] = dp[k] * ((-1.0)**k)
            
        dn = np.zeros(2)
        for i in range(7):
            if dq[i] == 0: continue
            for k in range(2):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 15:
                        s += q[j] * _I_TABLE[idx]
                dn[k] += 2.0 * 2.0 * np.pi * dq[i] * s
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c_idx] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
        
    return bound, grad

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    return val, grad

def construct():
    # Multi-start optimization
    # The previous best was around [-1.13, 0.04, 0.0003]
    # Let's try a variety of starts.
    
    starts = [
        np.array([-1.13, 0.04, 0.000308642]), # Previous best
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, -0.1, -0.001]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([-1.2, 0.045, 0.00035]),
    ]
    
    # Add some random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead if time permits
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
        
    return best_params
