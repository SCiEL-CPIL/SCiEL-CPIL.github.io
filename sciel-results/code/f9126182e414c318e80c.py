"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    
    A(f)^2 = <x^2>/<1> for |g|^2
    A(F)^2 = <xi^2>/<1> for |G|^2
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    # p_k depends on a0, a4, a8
    p0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p1 = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p2 = a4 + 210.0*a8 + 13860.0*a12
    p3 = -28.0*a8 - 6930.0*a12
    p4 = a8 + 1155.0*a12
    p5 = -66.0*a12
    p6 = a12
    p = np.array([p0, p1, p2, p3, p4, p5, p6])
    
    # Precompute I_n = integral x^{2n} e^{-x^2} dx
    sp = np.sqrt(np.pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) / (2*n) * I[n-1]
    
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        for j in range(7):
            pp = p[i] * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pp * I[idx]
    
    # For Fourier transform: Q(xi) = sum q_k xi^{2k} where q_k = p_k * (-1)^k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0)**k)
    
    # |G|^2 = 2*pi * |Q|^2 * e^{-xi^2}
    n = np.zeros(3)
    for i in range(7):
        for j in range(7):
            qq = q[i] * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0*np.pi * qq * I[idx]
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.zeros(3)
    
    val = np.sqrt(vf * vF)
    
    # Gradient calculation
    # val = sqrt(vf * vF)
    # d(val)/d(a) = 1/(2*val) * (vF * d(vf)/d(a) + vf * d(vF)/d(a))
    
    # vf = m2/m0 -> d(vf)/d(a) = (dm2/d(a)*m0 - m2*dm0/d(a)) / m0^2
    # vF = n2/n0 -> d(vF)/d(a) = (dn2/d(a)*n0 - n2*dn0/d(a)) / n0^2
    
    # We need dm_k/d(a_j) and dn_k/d(a_j) for j in {0, 4, 8}
    # m_k = sum p_i p_j I[k+i+j]
    # dm_k/d(a_alpha) = sum_i (dp_i/da_alpha) p_j I[k+i+j] + sum_j p_i (dp_j/da_alpha) I[k+i+j]
    #                  = 2 * sum_{i,j} (dp_i/da_alpha) p_j I[k+i+j]  (since sum is symmetric)
    
    # dp_i/d(a_alpha)
    # p0 = a0 + 3a4 + 105a8 + 945a12
    # p1 = -6a4 - 420a8 - 7560a12
    # p2 = a4 + 210a8 + 13860a12
    # p3 = -28a8 - 6930a12
    # p4 = a8 + 1155a12
    # p5 = -66a12
    # p6 = a12
    # a12 = -(a0 + 3a4 + 105a8)/945
    # da12/da0 = -1/945
    # da12/da4 = -3/945
    # da12/da8 = -105/945
    
    # Let's compute dp/da matrix (7x3)
    dp_da = np.zeros((7, 3))
    # Column 0: da0
    dp_da[0, 0] = 1.0 + 945.0 * (-1.0/945.0) # 1 - 1 = 0
    # p1..p6 depend on a12 only (and a4, a8)
    dp_da[1, 0] = -7560.0 * (-1.0/945.0)
    dp_da[2, 0] = 13860.0 * (-1.0/945.0)
    dp_da[3, 0] = -6930.0 * (-1.0/945.0)
    dp_da[4, 0] = 1155.0 * (-1.0/945.0)
    dp_da[5, 0] = -66.0 * (-1.0/945.0)
    dp_da[6, 0] = 1.0 * (-1.0/945.0)
    
    # Column 1: da4
    dp_da[0, 1] = 3.0 + 945.0 * (-3.0/945.0) # 3 - 3 = 0
    dp_da[1, 1] = -6.0 - 7560.0 * (-3.0/945.0)
    dp_da[2, 1] = 1.0 + 13860.0 * (-3.0/945.0)
    dp_da[3, 1] = -6930.0 * (-3.0/945.0)
    dp_da[4, 1] = 1155.0 * (-3.0/945.0)
    dp_da[5, 1] = -66.0 * (-3.0/945.0)
    dp_da[6, 1] = 1.0 * (-3.0/945.0)
    
    # Column 2: da8
    dp_da[0, 2] = 105.0 + 945.0 * (-105.0/945.0) # 105 - 105 = 0
    dp_da[1, 2] = -420.0 - 7560.0 * (-105.0/945.0)
    dp_da[2, 2] = 210.0 + 13860.0 * (-105.0/945.0)
    dp_da[3, 2] = -28.0 - 6930.0 * (-105.0/945.0)
    dp_da[4, 2] = 1.0 + 1155.0 * (-105.0/945.0)
    dp_da[5, 2] = -66.0 * (-105.0/945.0)
    dp_da[6, 2] = 1.0 * (-105.0/945.0)
    
    # dm_k/d(a_alpha) = 2 * sum_{i,j} dp_da[i, alpha] * p[j] * I[k+i+j]
    dm_da = np.zeros((3, 3)) # dm[k]/da[alpha]
    dn_da = np.zeros((3, 3)) # dn[k]/da[alpha]
    
    for k in range(3):
        for alpha in range(3):
            sum_m = 0.0
            sum_n = 0.0
            for i in range(7):
                dpi = dp_da[i, alpha]
                if dpi == 0.0: continue
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        sum_m += dpi * p[j] * I[idx]
                        sum_n += dpi * q[j] * 2.0 * np.pi * I[idx]
            dm_da[k, alpha] = 2.0 * sum_m
            dn_da[k, alpha] = 2.0 * sum_n
    
    # d(vf)/da = (dm2/da * m0 - m2 * dm0/da) / m0^2
    dvf_da = (dm_da[1] * m0 - m2 * dm_da[0]) / (m0 * m0)
    dvF_da = (dn_da[1] * n0 - n2 * dn_da[0]) / (n0 * n0)
    
    grad_val = (vF * dvf_da + vf * dvF_da) / (2.0 * val)
    
    return val, grad_val


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    _, grad = _bound_and_grad(params)
    return grad

def construct():
    rng = np.random.default_rng(42)
    
    best_val = 1e18
    best_params = np.zeros(3)
    
    # Multi-start L-BFGS-B
    # Define a set of initial guesses
    starts = []
    # The seed from the previous best
    starts.append(np.array([-1.13, 0.04, 0.000308642]))
    
    # Random starts
    for i in range(50):
        # Scale based on expected magnitude
        scale = 10.0 ** rng.uniform(-2, 2)
        p = rng.normal(0, 1, 3) * scale
        starts.append(p)
        
        # Also try some structured starts
        p2 = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)])
        starts.append(p2)

    for start in starts:
        if time.time() > _DEADLINE - 5:
            break
            
        try:
            res = minimize(_objective, start, jac=_objective_grad, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success or res.fun < best_val:
                if np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x
        except Exception:
            continue
            
    return best_params
# EVOLVE-BLOCK-END
