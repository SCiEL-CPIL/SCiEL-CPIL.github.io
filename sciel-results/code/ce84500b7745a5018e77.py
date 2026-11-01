import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 205.0  # Safety margin before 210s

# Precompute Hermite polynomial values and derivatives at key points
# H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# We need H_0, H_4, H_8, H_12

# Values at x=0:
# H_0(0) = 1
# H_4(0) = 3
# H_8(0) = 105
# H_12(0) = 945
H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 945.0

# Precompute integral values: I[n] = integral x^{2n} exp(-x^2) dx from -inf to inf
# I[n] = sqrt(pi) * (2n)! / (4^n * n!)
# I[0] = sqrt(pi)
# I[n] = I[n-1] * (2n-1) / (2*n) * ... wait, let me use recurrence
# I[n] = sqrt(pi) * prod_{k=1}^{n} (2k-1)/(2k) * ... 
# Actually: I[n] = I[n-1] * (2n-1)/(2n) is wrong.
# I[n] = sqrt(pi) * (2n)!/(4^n * n!)
# I[n]/I[n-1] = (2n)!/(4^n*n!) * (4^{n-1}*(n-1)!)/(2n-2)!
# = (2n)(2n-1)/(4n) = (2n-1)/2
# So I[n] = I[n-1] * (2n-1)/2
_I_table = np.zeros(30)
_I_table[0] = np.sqrt(np.pi)
for _n in range(1, 30):
    _I_table[_n] = _I_table[_n-1] * (2*_n - 1) / 2.0

# For the uncertainty bound, we need to compute:
# A(g) = sup_x |x * g(x)|  and  A(hat{g}) = sup_x |xi * hat{g}(xi)|
# The product A(g)*A(hat{g}) is bounded by C4 * (some expression)
# 
# For a Hermite-Gaussian function g(x) = P(x) * exp(-x^2/2),
# where P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12,
# the Fourier transform is also a Hermite-Gaussian.
#
# The key insight is that for such functions, the uncertainty product
# can be computed analytically using the moments of |g|^2 and |hat{g}|^2.

def _hermite_coeffs():
    """Return coefficients of H0, H4, H8, H12 in power basis."""
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H_12(x) = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 151200x^4 - 226800x^2 + 945
    # 
    # In the basis [x^0, x^2, x^4, x^6, x^8, x^10, x^12]:
    h0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    h4 = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    h8 = np.array([105.0, -420.0, 210.0, 0.0, 1.0, 0.0, 0.0])
    h12 = np.array([945.0, -226800.0, 151200.0, -30240.0, 2520.0, -84.0, 1.0])
    return h0, h4, h8, h12

# Precompute hermite coefficients once
_H0, _H4, _H8, _H12 = _hermite_coeffs()

def _compute_moments(c0, c4, c8):
    """
    Compute the moments needed for the uncertainty bound.
    
    For g(x) = P(x) * exp(-x^2/2), we need:
    - ||g||^2 = integral of |g(x)|^2 dx
    - ||x*g||^2 = integral of x^2 * |g(x)|^2 dx
    - ||hat{g}||^2 = ||g||^2 (Plancherel)
    - ||xi*hat{g}||^2 = ||g'||^2 (Fourier derivative property)
    
    The uncertainty product is bounded by:
    A(g)*A(hat{g}) <= C4 * (||x*g||^2 / ||g||^2)^(1/2) * (||g'||^2 / ||g||^2)^(1/2)
    
    Actually, the standard uncertainty relation gives:
    (||x*f||^2 / ||f||^2) * (||f'||^2 / ||f||^2) >= 1/4
    
    For our bound, we need to compute these ratios.
    """
    h0, h4, h8, h12 = _hermite_coeffs()
    
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # c12 = -(c0 + 3*c4 + 105*c8) / 945
    c12 = -(c0 + 3.0*c4 + 105.0*c8) / 945.0
    
    # P(x) in power basis [x^0, x^2, x^4, x^6, x^8, x^10, x^12]
    P = c0*h0 + c4*h4 + c8*h8 + c12*h12
    # P = [P0, P2, P4, P6, P8, P10, P12]
    
    # |g(x)|^2 = P(x)^2 * exp(-x^2)
    # We need integral of P(x)^2 * exp(-x^2) dx from -inf to inf
    # and integral of x^2 * P(x)^2 * exp(-x^2) dx
    # and integral of (P'(x))^2 * exp(-x^2) dx
    
    # P(x)^2 is a polynomial of degree 24
    # Let's compute it coefficient by coefficient
    # P(x) = sum_{k=0}^{6} P[2k] * x^{2k}
    # P(x)^2 = sum_{i,j} P[2i]*P[2j] * x^{2(i+j)}
    
    # Let Q[m] be the coefficient of x^{2m} in P(x)^2
    Q = np.zeros(13)  # degrees 0 to 24
    for i in range(7):
        for j in range(7):
            Q[i+j] += P[i] * P[j]
    
    # integral of x^{2n} * exp(-x^2) dx from -inf to inf = sqrt(pi) * (2n-1)!! / 2^n
    # = sqrt(pi) * Gamma(n + 1/2) / Gamma(1/2) * 2^n / 2^n... 
    # Actually: integral x^{2n} e^{-x^2} dx = sqrt(pi) * (2n)! / (4^n * n!)
    
    def integral_x2n(n):
        """Integral of x^{2n} * exp(-x^2) dx from -inf to inf"""
        if n == 0:
            return np.sqrt(np.pi)
        # (2n)! / (4^n * n!) * sqrt(pi)
        # Use log to avoid overflow
        log_val = np.sum(np.log(np.arange(1, 2*n+1))) - n*np.log(4) - np.sum(np.log(np.arange(1, n+1))) + 0.5*np.log(np.pi)
        return np.exp(log_val)
    
    # ||g||^2 = sum_{m=0}^{12} Q[m] * integral_x2n(m)
    norm_sq = 0.0
    for m in range(13):
        norm_sq += Q[m] * integral_x2n(m)
    
    # ||x*g||^2 = integral x^2 * P(x)^2 * exp(-x^2) dx
    # = sum_{m=0}^{12} Q[m] * integral x^{2(m+1)} exp(-x^2) dx
    x_norm_sq = 0.0
    for m in range(13):
        x_norm_sq += Q[m] * integral_x2n(m+1)
    
    # Now compute ||g'||^2
    # g(x) = P(x) * exp(-x^2/2)
    # g'(x) = P'(x) * exp(-x^2/2) - x * P(x) * exp(-x^2/2)
    # g'(x) = [P'(x) - x*P(x)] * exp(-x^2/2)
    
    # P'(x) = sum_{k=1}^{6} 2k * P[2k] * x^{2k-1}
    # x*P(x) = sum_{k=0}^{6} P[2k] * x^{2k+1}
    
    # Let R(x) = P'(x) - x*P(x)
    # R(x) is an odd polynomial
    
    # R(x) = sum_{k=1}^{6} 2k*P[2k]*x^{2k-1} - sum_{k=0}^{6} P[2k]*x^{2k+1}
    # = sum_{k=1}^{6} (2k*P[2k] - P[2k-1])*x^{2k-1} - P[6]*x^{13}
    # Wait, let me be more careful.
    
    # P'(x) = 2*P[1]*x + 4*P[2]*x^3 + 6*P[3]*x^5 + 8*P[4]*x^7 + 10*P[5]*x^9 + 12*P[6]*x^11
    # x*P(x) = P[0]*x + P[1]*x^3 + P[2]*x^5 + P[3]*x^7 + P[4]*x^9 + P[5]*x^11 + P[6]*x^13
    
    # R(x) = (2*P[1]-P[0])*x + (4*P[2]-P[1])*x^3 + (6*P[3]-P[2])*x^5 
    #        + (8*P[4]-P[3])*x^7 + (10*P[5]-P[4])*x^9 + (12*P[6]-P[5])*x^11 - P[6]*x^13
    
    # R(x) = sum_{k=0}^{6} R[2k+1] * x^{2k+1}
    R = np.zeros(7)  # coefficients for x^1, x^3, ..., x^13
    for k in range(6):
        R[k] = 2*(k+1)*P[k+1] - P[k]
    R[6] = -P[6]
    
    # |g'(x)|^2 = R(x)^2 * exp(-x^2)
    # R(x)^2 is an even polynomial of degree 26
    # Let S[m] be the coefficient of x^{2m} in R(x)^2
    S = np.zeros(14)  # degrees 0 to 26
    for i in range(7):
        for j in range(7):
            S[i+j] += R[i] * R[j]
    
    # ||g'||^2 = sum_{m=0}^{13} S[m] * integral_x2n(m)
    deriv_norm_sq = 0.0
    for m in range(14):
        deriv_norm_sq += S[m] * integral_x2n(m)
    
    # The uncertainty product bound
    # A(g) * A(hat{g}) <= C4 * sqrt(||x*g||^2 / ||g||^2) * sqrt(||g'||^2 / ||g||^2)
    # = C4 * sqrt(||x*g||^2 * ||g'||^2 / ||g||^2^2)
    
    if norm_sq <= 0:
        return 0.0, 0.0, 0.0  # Degenerate case
    
    ratio_x = x_norm_sq / norm_sq
    ratio_d = deriv_norm_sq / norm_sq
    
    bound = np.sqrt(ratio_x * ratio_d)
    
    return bound, norm_sq, x_norm_sq

def _bound_and_grad(coeffs):
    """
    Compute the uncertainty bound and its gradient with respect to (c0, c4, c8).
    Uses analytic gradient via chain rule.
    """
    c0, c4, c8 = coeffs
    bound, norm_sq, x_norm_sq = _compute_moments(c0, c4, c8)
    
    if norm_sq <= 0:
        return bound, np.zeros(3)
    
    # Recompute intermediate values for gradient
    c12 = -(c0 + 3.0*c4 + 105.0*c8) / 945.0
    P = c0*_H0 + c4*_H4 + c8*_H8 + c12*_H12
    
    Q = np.zeros(13)
    for i in range(7):
        Q[i:i+7] += P[i] * P[:7-i] if i < 7 else 0
    
    R = np.zeros(7)
    for k in range(6):
        R[k] = 2*(k+1)*P[k+1] - P[k]
    R[6] = -P[6]
    
    S = np.zeros(14)
    for i in range(7):
        S[i:i+7] += R[i] * R[:7-i] if i < 7 else 0
    
    # d(norm_sq)/dP_j = 2 * sum_m Q_partial[m] * I[m] where Q_partial[m] = sum_i (P[i]*delta_{i+j,m} + P[m-i]*delta_{i+j,m})
    # Actually: dQ[m]/dP[j] = 2*P[m-j] for j<=m, and also P[j-m]... 
    # Q[m] = sum_{i+j=m} P[i]*P[j], so dQ[m]/dP[j] = 2*P[m-j] if j<=m, else 0
    # But we need dQ[m]/dP[j] for the gradient.
    
    # Let's compute d(norm_sq)/dP[j] = 2 * sum_m P[m-j] * I[m] for j<=m
    # More carefully: norm_sq = sum_m Q[m]*I[m] = sum_m sum_{i+j=m} P[i]*P[j]*I[m]
    # d(norm_sq)/dP[j] = 2 * sum_{m>=j} P[m-j] * I[m]
    
    d_norm_dP = np.zeros(7)
    for j in range(7):
        s = 0.0
        for m in range(j, 13):
            s += P[m-j] * _I_table[m]
        d_norm_dP[j] = 2.0 * s
    
    # d(x_norm_sq)/dP[j] = 2 * sum_{m>=j} P[m-j] * I[m+1]
    d_xnorm_dP = np.zeros(7)
    for j in range(7):
        s = 0.0
        for m in range(j, 13):
            s += P[m-j] * _I_table[m+1]
        d_xnorm_dP[j] = 2.0 * s
    
    # d(deriv_norm_sq)/dP[j]: more complex because R depends on P
    # R[k] = 2*(k+1)*P[k+1] - P[k] for k=0..5, R[6] = -P[6]
    # S[m] = sum_{i+j=m} R[i]*R[j]
    # deriv_norm_sq = sum_m S[m]*I[m]
    # d(deriv_norm_sq)/dP[j] = 2 * sum_m (dS[m]/dP[j]) * I[m]
    # dS[m]/dP[j] = sum_i (dR[i]/dP[j]*R[m-i] + R[i]*dR[m-i]/dP[j])
    # dR[i]/dP[j]: R[i] = 2*(i+1)*P[i+1] - P[i] for i<6, R[6]=-P[6]
    # So dR[i]/dP[j] = 2*(i+1)*delta_{j,i+1} - delta_{j,i} for i<6
    #                 = -delta_{j,6} for i=6
    
    d_deriv_dP = np.zeros(7)
    for j in range(7):
        s = 0.0
        for m in range(14):
            ds_dm = 0.0
            for i in range(7):
                if i+j == m:
                    # dR[i]/dP[j]
                    if i < 6:
                        dr_di = 0.0
                        if j == i+1:
                            dr_di += 2.0*(i+1)
                        if j == i:
                            dr_di -= 1.0
                    else:  # i == 6
                        dr_di = 0.0
                        if j == 6:
                            dr_di -= 1.0
                    
                    # dR[m-i]/dP[j]
                    mi = m - i
                    if 0 <= mi < 7:
                        if mi < 6:
                            dr_mi = 0.0
                            if j == mi+1:
                                dr_mi += 2.0*(mi+1)
                            if j == mi:
                                dr_mi -= 1.0
                        else:  # mi == 6
                            dr_mi = 0.0
                            if j == 6:
                                dr_mi -= 1.0
                        ds_dm += dr_di * R[mi] + R[i] * dr_mi
            s += ds_dm * _I_table[m]
        d_deriv_dP[j] = s
    
    # Now chain rule: P depends on c0, c4, c8
    # P = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # c12 = -(c0 + 3*c4 + 105*c8)/945
    # dP/dc0 = H0 + dc12/dc0 * H12 = H0 - (1/945)*H12
    # dP/dc4 = H4 + dc12/dc4 * H12 = H4 - (3/945)*H12
    # dP/dc8 = H8 + dc12/dc8 * H12 = H8 - (105/945)*H12
    
    dP_dc0 = _H0 - (1.0/945.0)*_H12
    dP_dc4 = _H4 - (3.0/945.0)*_H12
    dP_dc8 = _H8 - (105.0/945.0)*_H12
    
    # d(bound)/dc = 0.5 * bound^(-1) * d(bound^2)/dc
    # bound^2 = (x_norm_sq/norm_sq) * (deriv_norm_sq/norm_sq)
    # Let A = x_norm_sq, B = deriv_norm_sq, C = norm_sq
    # bound^2 = A*B/C^2
    # d(bound^2)/dc = (dA/dc)*B/C^2 + A*dB/dc/C^2 - 2*A*B*dC/dc/C^3
    #              = (B*dA + A*dB - 2*A*B*dC/C)/C^2
    
    A = x_norm_sq
    B = deriv_norm_sq
    C = norm_sq
    
    dA_dc0 = np.dot(d_norm_dP, dP_dc0)  # Wait, this is wrong
    # Actually: dA/dc0 = sum_j dA_dP[j] * dP_dc0[j]
    dA_dc0 = np.dot(d_xnorm_dP, dP_dc0)
    dA_dc4 = np.dot(d_xnorm_dP, dP_dc4)
    dA_dc8 = np.dot(d_xnorm_dP, dP_dc8)
    
    dB_dc0 = np.dot(d_deriv_dP, dP_dc0)
    dB_dc4 = np.dot(d_deriv_dP, dP_dc4)
    dB_dc8 = np.dot(d_deriv_dP, dP_dc8)
    
    dC_dc0 = np.dot(d_norm_dP, dP_dc0)
    dC_dc4 = np.dot(d_norm_dP, dP_dc4)
    dC_dc8 = np.dot(d_norm_dP, dP_dc8)
    
    bound_sq = A * B / (C * C)
    if bound_sq <= 0:
        return bound, np.zeros(3)
    
    factor = 0.5 / bound
    
    d_bound_sq_dc0 = (B*dA_dc0 + A*dB_dc0 - 2.0*A*B*dC_dc0/C) / (C*C)
    d_bound_sq_dc4 = (B*dA_dc4 + A*dB_dc4 - 2.0*A*B*dC_dc4/C) / (C*C)
    d_bound_sq_dc8 = (B*dA_dc8 + A*dB_dc8 - 2.0*A*B*dC_dc8/C) / (C*C)
    
    grad = factor * np.array([d_bound_sq_dc0, d_bound_sq_dc4, d_bound_sq_dc8])
    
    return bound, grad

def _objective(coeffs):
    """Objective function to minimize."""
    bound, _, _ = _compute_moments(coeffs[0], coeffs[1], coeffs[2])
    return bound

def _objective_grad(coeffs):
    """Objective function and gradient."""
    bound, grad = _bound_and_grad(coeffs)
    return bound, grad

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    # Seed initial guess
    seed = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score = _objective(seed)
    best_coeffs = seed.copy()
    
    # Multi-start optimization with L-BFGS-B
    np.random.seed(42)
    
    bounds = [(-5.0, 5.0), (-2.0, 2.0), (-0.5, 0.5)]
    
    num_starts = 50
    
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 30:
            break
        
        # Generate random initial guess
        if i == 0:
            x0 = seed.copy()
        else:
            x0 = np.array([
                np.random.uniform(bounds[0][0], bounds[0][1]),
                np.random.uniform(bounds[1][0], bounds[1][1]),
                np.random.uniform(bounds[2][0], bounds[2][1])
            ])
        
        try:
            res = minimize(_objective_grad, x0, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Try more targeted starts near the best solution
    for i in range(20):
        if time.time() > TIME_LIMIT - 15:
            break
        
        # Perturb the best solution
        x0 = best_coeffs + np.random.normal(0, 0.05, size=3)
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        try:
            res = minimize(_objective_grad, x0, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < TIME_LIMIT - 10:
        try:
            res = minimize(_objective_grad, best_coeffs, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return best_coeffs
