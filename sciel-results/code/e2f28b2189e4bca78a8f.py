"""Uncertainty bound minimization via analytic gradient and multi-start L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195.0

# Precompute integrals I[n] = integral x^{2n} exp(-x^2) dx
_SP = np.sqrt(np.pi)
_I = np.zeros(13)
_I[0] = _SP
for _n in range(1, 13):
    _I[_n] = _I[_n - 1] * (2 * _n - 1) / 2.0

# Precompute sign array for Fourier transform
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Hermite polynomial coefficients: H_k(x) = sum_{j=0}^{k//2} c_{k,j} x^{2j}
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# Coefficient matrix: row k (0..3) corresponds to H_{4k}, columns 0..k are powers x^{2j}
# H0: [1]
# H4: [3, -6, 1]
# H8: [105, -420, 210, -28, 1]
# H12: [945, -7560, 13860, -6930, 1155, -66, 1]

def _p_from_coeffs(a0, a4, a8):
    """Compute polynomial coefficients p[0..6] for P(x) = sum p_k x^{2k}."""
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12
    return np.array([p0, p1, p2, p3, p4, p5, p6])


def _moments(p):
    """Compute m0 = sum p_i p_j I[i+j] and m1 = sum p_i p_j I[i+j+1]."""
    # Use matrix form for efficiency
    # m_k = sum_{i,j} p_i p_j I[k+i+j]
    m0 = 0.0
    m1 = 0.0
    for i in range(7):
        pi = p[i]
        for j in range(7):
            val = pi * p[j]
            s = i + j
            if s < 13:
                m0 += val * _I[s]
            if s + 1 < 13:
                m1 += val * _I[s + 1]
    return m0, m1


def _bound_and_grad(a0, a4, a8):
    """Compute the uncertainty bound and its gradient with respect to (a0, a4, a8)."""
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    p = _p_from_coeffs(a0, a4, a8)
    
    # Fourier: q = p * signs
    q = p * _SIGNS
    
    m0, m1 = _moments(p)
    n0, n1 = _moments(q)
    n0 *= 2.0 * np.pi
    n1 *= 2.0 * np.pi
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)
    
    vf = m1 / m0
    vF = n1 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.zeros(3)
    
    bound = np.sqrt(vf * vF)
    
    # Gradient computation via chain rule
    # bound = sqrt(vf * vF)
    # dbound/da = 0.5 / sqrt(vf*vF) * (vF * dvf/da + vf * dvF/da)
    
    # dvf/da = (dm1/da * m0 - m1 * dm0/da) / m0^2
    # dvF/da = (dn1/da * n0 - n1 * dn0/da) / n0^2
    
    # dm0/da_k = 2 * sum_j p_j * dp_k/da * I[k+j]  (for k=0..6, but we need w.r.t. a0,a4,a8)
    # Actually, let's compute dp/da_j for j=0,1,2 (a0, a4, a8)
    
    # p as function of (a0, a4, a8):
    # dp/da0: [1, 0, 0, 0, 0, 0, 0]
    # dp/da4: [3, -6, 1, 0, 0, 0, 0]
    # dp/da8: [105, -420, 210, -28, 1, 0, 0]
    # dp/da12: [945, -7560, 13860, -6930, 1155, -66, 1]
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    
    dp_da0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dp_da4 = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    dp_da8 = np.array([105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0])
    
    dp12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    da12_da0 = -1.0 / 945.0
    da12_da4 = -3.0 / 945.0
    da12_da8 = -105.0 / 945.0
    
    dp_da0_full = dp_da0 + dp12 * da12_da0
    dp_da4_full = dp_da4 + dp12 * da12_da4
    dp_da8_full = dp_da8 + dp12 * da12_da8
    
    # For each parameter a_k, compute dm0/da_k, dm1/da_k, dn0/da_k, dn1/da_k
    # dm0/da_k = 2 * sum_{i,j} (dp_i/da_k * p_j + p_i * dp_j/da_k) * I[i+j] / 2
    #          = 2 * sum_j dp_j/da_k * (sum_i p_i I[i+j])
    # Let S[j] = sum_i p_i I[i+j], then dm0/da_k = 2 * sum_j dp_j/da_k * S[j]
    
    def _dm_dparam(dp_da, p_arr, I_arr):
        """Compute dm0/da and dm1/da given dp/da and p."""
        # S0[j] = sum_i p_i I[i+j]
        # S1[j] = sum_i p_i I[i+j+1]
        S0 = np.zeros(7)
        S1 = np.zeros(7)
        for j in range(7):
            for i in range(7):
                s = i + j
                if s < 13:
                    S0[j] += p_arr[i] * I_arr[s]
                if s + 1 < 13:
                    S1[j] += p_arr[i] * I_arr[s + 1]
        
        dm0 = 2.0 * np.dot(dp_da, S0)
        dm1 = 2.0 * np.dot(dp_da, S1)
        return dm0, dm1
    
    dm0_da0, dm1_da0 = _dm_dparam(dp_da0_full, p, _I)
    dm0_da4, dm1_da4 = _dm_dparam(dp_da4_full, p, _I)
    dm0_da8, dm1_da8 = _dm_dparam(dp_da8_full, p, _I)
    
    # For Fourier, q = p * signs, so dq/da_k = (dp/da_k) * signs
    dq_da0 = dp_da0_full * _SIGNS
    dq_da4 = dp_da4_full * _SIGNS
    dq_da8 = dp_da8_full * _SIGNS
    
    dn0_da0, dn1_da0 = _dm_dparam(dq_da0, q, _I)
    dn0_da4, dn1_da4 = _dm_dparam(dq_da4, q, _I)
    dn0_da8, dn1_da8 = _dm_dparam(dq_da8, q, _I)
    
    # Scale n by 2*pi
    dn0_da0 *= 2.0 * np.pi
    dn1_da0 *= 2.0 * np.pi
    dn0_da4 *= 2.0 * np.pi
    dn1_da4 *= 2.0 * np.pi
    dn0_da8 *= 2.0 * np.pi
    dn1_da8 *= 2.0 * np.pi
    
    # dvf/da_k = (dm1_da_k * m0 - m1 * dm0_da_k) / m0^2
    # dvF/da_k = (dn1_da_k * n0 - n1 * dn0_da_k) / n0^2
    
    dvf_da0 = (dm1_da0 * m0 - m1 * dm0_da0) / (m0 * m0)
    dvf_da4 = (dm1_da4 * m0 - m1 * dm0_da4) / (m0 * m0)
    dvf_da8 = (dm1_da8 * m0 - m1 * dm0_da8) / (m0 * m0)
    
    dvF_da0 = (dn1_da0 * n0 - n1 * dn0_da0) / (n0 * n0)
    dvF_da4 = (dn1_da4 * n0 - n1 * dn0_da4) / (n0 * n0)
    dvF_da8 = (dn1_da8 * n0 - n1 * dn0_da8) / (n0 * n0)
    
    # dbound/da_k = 0.5 / sqrt(vf*vF) * (vF * dvf_da_k + vf * dvF_da_k)
    factor = 0.5 / bound
    
    grad0 = factor * (vF * dvf_da0 + vf * dvF_da0)
    grad4 = factor * (vF * dvf_da4 + vf * dvF_da4)
    grad8 = factor * (vF * dvf_da8 + vf * dvF_da8)
    
    return bound, np.array([grad0, grad4, grad8])


def _objective(params):
    """Objective function for optimizer."""
    bound, _ = _bound_and_grad(params[0], params[1], params[2])
    return bound


def _objective_grad(params):
    """Objective function and gradient for optimizer."""
    bound, grad = _bound_and_grad(params[0], params[1], params[2])
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
    return bound, grad


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses multi-start L-BFGS-B with analytic gradients.
    """
    # Known good starting point from seed
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound, _ = _bound_and_grad(seed[0], seed[1], seed[2])
    
    best_bound = seed_bound
    best_params = seed.copy()
    
    # Generate diverse starting points
    rng = np.random.default_rng(12345)
    
    starts = [seed.copy()]
    
    # Add random perturbations around seed
    for scale in [0.1, 0.3, 0.5, 1.0, 2.0]:
        for _ in range(10):
            pert = rng.normal(0, scale, 3)
            starts.append(seed + pert)
    
    # Add completely random points
    for _ in range(30):
        starts.append(rng.normal(0, 2.0, 3))
    
    # Add points along specific directions
    for direction in [
        np.array([1, 0, 0]), np.array([-1, 0, 0]),
        np.array([0, 1, 0]), np.array([0, -1, 0]),
        np.array([0, 0, 1]), np.array([0, 0, -1]),
        np.array([1, 1, 1]) / np.sqrt(3),
        np.array([-1, 1, 1]) / np.sqrt(3),
        np.array([1, -1, 1]) / np.sqrt(3),
        np.array([1, 1, -1]) / np.sqrt(3),
    ]:
        for r in [0.5, 1.0, 2.0, 4.0]:
            starts.append(seed + r * direction)
    
    # Remove duplicates (approximately)
    unique_starts = [starts[0]]
    for s in starts[1:]:
        if all(np.linalg.norm(s - u) > 1e-6 for u in unique_starts):
            unique_starts.append(s)
    starts = unique_starts
    
    # Run L-BFGS-B from each starting point
    options = {
        'maxiter': 500,
        'ftol': 1e-12,
        'gtol': 1e-10,
    }
    
    for i, x0 in enumerate(starts):
        if time.time() > _DEADLINE - 30:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                jac=_objective_grad,
                options=options
            )
            
            if result.success or result.fun < best_bound:
                b, _ = _bound_and_grad(result.x[0], result.x[1], result.x[2])
                if np.isfinite(b) and b < best_bound:
                    best_bound = b
                    best_params = result.x.copy()
        except Exception:
            continue
    
    # Final verification
    final_bound, _ = _bound_and_grad(best_params[0], best_params[1], best_params[2])
    if not np.isfinite(final_bound) or final_bound > best_bound + 1e-10:
        return seed.copy()
    
    return best_params
# EVOLVE-BLOCK-END
