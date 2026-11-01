"""Optimized uncertainty bound minimizer with precomputed moments and efficient search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195.0  # Safety margin before 210s kill

# Precompute integration weights I_n = integral x^{2n} e^{-x^2} dx
# I_n = (2n)! / (4^n n!) * sqrt(pi)
# We need n up to 13 (since max index in variance calc is 1+6+6=13)
_N_MAX = 14
_I = np.zeros(_N_MAX)
for n in range(_N_MAX):
    if n == 0:
        _I[n] = np.sqrt(np.pi)
    else:
        # Recurrence: I_n = (2n-1) * I_{n-1} / (2 * 4) = (2n-1) * I_{n-1} / 8
        _I[n] = (2*n - 1) * _I[n-1] / 8.0

def _compute_bound_and_grad(params):
    """
    Computes the uncertainty bound and its gradient for given a0, a4, a8.
    a12 is determined by g(0)=0.
    """
    a0, a4, a8 = params
    # a12 = -(a0 + 3*a4 + 105*a8) / 945
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^6 p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    p = np.zeros(7)
    # p0 (x^0)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    # p1 (x^2)
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    # p2 (x^4)
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    # p3 (x^6)
    p[3] = -28.0*a8 - 6930.0*a12
    # p4 (x^8)
    p[4] = a8 + 1155.0*a12
    # p5 (x^10)
    p[5] = -66.0*a12
    # p6 (x^12)
    p[6] = a12
    
    # Moments of |g|^2: m_k = integral x^{2k} |P(x)|^2 e^{-x^2} dx
    # |P|^2 = sum_{i,j} p_i p_j x^{2(i+j)}
    # m_k = sum_{i,j} p_i p_j I_{k+i+j}
    
    # We need m_0, m_2 (for variance) -> k=0, 1 in the formula above if we map k to power/2
    # Wait, variance is <x^2> = m_2 / m_0.
    # m_0 = sum_{i,j} p_i p_j I_{i+j}
    # m_2 = sum_{i,j} p_i p_j I_{1+i+j} (since x^2 * x^{2(i+j)} = x^{2(i+j+1)})
    
    # Vectorized computation
    # p_i p_j matrix
    P_mat = np.outer(p, p)
    
    # m_0: sum I_{i+j} * P_mat[i,j]
    # Indices i,j in 0..6. i+j in 0..12.
    # Create index matrix
    idx = np.add.outer(np.arange(7), np.arange(7))
    m0 = np.sum(P_mat * _I[idx])
    
    # m_2: sum I_{1+i+j} * P_mat[i,j]
    # Indices 1+i+j in 1..13.
    # _I has length 13 (indices 0..12). We need up to 1+6+6=13.
    # My _I array only goes up to 12. I need I_13.
    # Let's check max index: i=6, j=6 -> i+j=12. For m_2, we need I_{1+12}=I_{13}.
    # I need to extend _I to 14 elements (0..13).
    
    # Recalculate I_13
    # (26)! / (4^13 * 13!) * sqrt(pi)
    # I will just compute it on the fly or extend the array.
    # For efficiency, I'll assume the caller handles it or I extend it here.
    # Actually, I defined _I with size 13 (0..12). I need 14 (0..13).
    # Let's just compute I_13 explicitly in this function or fix the global.
    # Better: fix the global array size to 14.
    
    # Let's re-verify the global _I size.
    # I used range(13) which is 0..12.
    # I need I_13.
    # I will compute it here to be safe, or better, I should have made _I size 14.
    # I'll compute I_13 locally if needed, but it's better to just make the global array bigger.
    # Since I can't change the global easily without rewriting the whole block, 
    # I will compute I_13 here.
    
    # Actually, let's just make the global array size 14 in the rewritten code.
    # For now, in this function, I'll assume I have access to I up to 13.
    # I will modify the global _I to size 14 in the final code.
    
    # Assuming _I is extended to 14:
    I13 = _I[13] if len(_I) > 13 else np.exp(np.sum(np.log(np.arange(1, 27))) - 13*np.log(4) - np.sum(np.log(np.arange(1, 14))) + 0.5*np.log(np.pi))
    
    # To be robust, let's just use a local array of size 14.
    # No, let's stick to the global but ensure it's big enough.
    # I will rewrite the global init to size 14.
    
    # Let's assume _I is correct for 0..12. I need I_13.
    # I will compute I_13 here.
    log_I13 = 0.0
    for m in range(1, 27): log_I13 += np.log(m)
    log_I13 -= 13*np.log(4.0)
    for m in range(1, 14): log_I13 -= np.log(m)
    log_I13 += 0.5*np.log(np.pi)
    I13 = np.exp(log_I13)
    
    # Construct full I array for this call
    I_full = np.append(_I, [I13])
    
    m2 = np.sum(P_mat * I_full[1 + idx])
    
    # Fourier side
    # Q(x) = P(ix) ? No.
    # G(xi) = sqrt(2pi) P(-i xi) e^{-xi^2/2}
    # P(-i xi) = sum p_k (-i)^{2k} xi^{2k} = sum p_k (-1)^k xi^{2k}
    # So q_k = p_k * (-1)^k
    q = p * ((-1.0) ** np.arange(7))
    
    Q_mat = np.outer(q, q)
    
    # n_0 = 2pi * sum q_i q_j I_{i+j}
    n0 = 2.0 * np.pi * np.sum(Q_mat * I_full[idx])
    
    # n_2 = 2pi * sum q_i q_j I_{1+i+j}
    n2 = 2.0 * np.pi * np.sum(Q_mat * I_full[1 + idx])
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vf * vF)
    
    # Gradient Calculation
    # We need dp/da, dq/da
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    da12_dac = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp_k/da_c
    # p0 = a0 + 3a4 + 105a8 + 945a12
    # p1 = -6a4 - 420a8 - 7560a12
    # p2 = a4 + 210a8 + 13860a12
    # p3 = -28a8 - 6930a12
    # p4 = a8 + 1155a12
    # p5 = -66a12
    # p6 = a12
    
    # Matrix of derivatives of p_k w.r.t a_c (3x7)
    # Rows: a0, a4, a8. Cols: p0..p6
    Dp_da = np.zeros((3, 7))
    # Row 0 (a0)
    Dp_da[0, 0] = 1.0 + 945.0 * da12_dac[0]
    Dp_da[0, 1] = -7560.0 * da12_dac[0]
    Dp_da[0, 2] = 13860.0 * da12_dac[0]
    Dp_da[0, 3] = -6930.0 * da12_dac[0]
    Dp_da[0, 4] = 1155.0 * da12_dac[0]
    Dp_da[0, 5] = -66.0 * da12_dac[0]
    Dp_da[0, 6] = 1.0 * da12_dac[0]
    
    # Row 1 (a4)
    Dp_da[1, 0] = 3.0 + 945.0 * da12_dac[1]
    Dp_da[1, 1] = -6.0 - 7560.0 * da12_dac[1]
    Dp_da[1, 2] = 1.0 + 13860.0 * da12_dac[1]
    Dp_da[1, 3] = -6930.0 * da12_dac[1]
    Dp_da[1, 4] = 1155.0 * da12_dac[1]
    Dp_da[1, 5] = -66.0 * da12_dac[1]
    Dp_da[1, 6] = 1.0 * da12_dac[1]
    
    # Row 2 (a8)
    Dp_da[2, 0] = 105.0 + 945.0 * da12_dac[2]
    Dp_da[2, 1] = -420.0 - 7560.0 * da12_dac[2]
    Dp_da[2, 2] = 210.0 + 13860.0 * da12_dac[2]
    Dp_da[2, 3] = -28.0 - 6930.0 * da12_dac[2]
    Dp_da[2, 4] = 1.0 + 1155.0 * da12_dac[2]
    Dp_da[2, 5] = -66.0 * da12_dac[2]
    Dp_da[2, 6] = 1.0 * da12_dac[2]
    
    # dq_k/da_c = Dp_k/da_c * (-1)^k
    Dq_da = Dp_da * ((-1.0) ** np.arange(7))[np.newaxis, :]
    
    # dm0/da_c = 2 * sum_{i,j} (dp_i p_j + p_i dp_j) I_{i+j} / 2 ?
    # m0 = sum_{i,j} p_i p_j I_{i+j}
    # dm0 = sum_{i,j} (dp_i p_j + p_i dp_j) I_{i+j}
    # Due to symmetry of I and p, dm0 = 2 * sum_{i,j} dp_i p_j I_{i+j}
    
    # We can compute this for all 3 parameters at once.
    # dm0_da[c] = 2 * sum_{i,j} Dp_da[c,i] * p[j] * I_full[i+j]
    # Let S_p[c, i] = sum_j p[j] I_full[i+j]
    # Then dm0_da[c] = 2 * sum_i Dp_da[c, i] * S_p[c, i]
    
    # S_p shape (7,). Same for all c? No, p is fixed.
    # S_p[i] = sum_j p[j] I_full[i+j]
    S_p = np.zeros(7)
    for i in range(7):
        S_p[i] = np.dot(p, I_full[i + np.arange(7)])
        
    dm0_da = 2.0 * (Dp_da * S_p[np.newaxis, :]).sum(axis=1)
    
    # dm2_da[c] = 2 * sum_{i,j} Dp_da[c,i] p[j] I_full[1+i+j]
    S_p2 = np.zeros(7)
    for i in range(7):
        S_p2[i] = np.dot(p, I_full[1 + i + np.arange(7)])
        
    dm2_da = 2.0 * (Dp_da * S_p2[np.newaxis, :]).sum(axis=1)
    
    # Similarly for n
    S_q = np.zeros(7)
    for i in range(7):
        S_q[i] = np.dot(q, I_full[i + np.arange(7)])
        
    dn0_da = 2.0 * 2.0 * np.pi * (Dq_da * S_q[np.newaxis, :]).sum(axis=1)
    
    S_q2 = np.zeros(7)
    for i in range(7):
        S_q2[i] = np.dot(q, I_full[1 + i + np.arange(7)])
        
    dn2_da = 2.0 * 2.0 * np.pi * (Dq_da * S_q2[np.newaxis, :]).sum(axis=1)
    
    # d(vf)/da = (dm2 m0 - m2 dm0) / m0^2
    d_vf = (dm2_da * m0 - m2 * dm0_da) / (m0 * m0)
    
    # d(vF)/da = (dn2 n0 - n2 dn0) / n0^2
    d_vF = (dn2_da * n0 - n2 * dn0_da) / (n0 * n0)
    
    # d(bound)/da = (vF d_vf + vf d_vF) / (2 bound)
    grad = (vF * d_vf + vf * d_vF) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    return val, grad

def construct():
    # Start from the known good seed
    best_params = np.array([-1.13, 0.04, 0.000308642])
    best_val, _ = _compute_bound_and_grad(best_params)
    
    # Multi-start local optimization from diverse initial points
    initial_points = [
        [-1.13, 0.04, 0.000308642],
        [-1.0, 0.05, 0.0003],
        [-1.2, 0.03, 0.0004],
        [-0.9, 0.06, 0.0002],
        [-1.1, 0.045, 0.00035],
        [-1.15, 0.035, 0.00025],
        [-1.05, 0.055, 0.00025],
        [-1.25, 0.025, 0.00045],
    ]
    
    for x0 in initial_points:
        if time.time() > _DEADLINE - 15.0:
            break
        try:
            res = minimize(
                _objective_grad, 
                np.array(x0), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final polish
    if time.time() < _DEADLINE - 5.0:
        try:
            res = minimize(
                _objective_grad, 
                best_params, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
