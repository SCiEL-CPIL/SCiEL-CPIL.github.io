import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

START_TIME = time.time()
TIME_LIMIT = 205.0

# Hermite polynomial coefficients in power basis [x^0, x^2, x^4, x^6, x^8, x^10, x^12]
_H0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_H4 = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
_H8 = np.array([105.0, -420.0, 210.0, 0.0, 1.0, 0.0, 0.0])
_H12 = np.array([945.0, -226800.0, 151200.0, -30240.0, 2520.0, -84.0, 1.0])

# Precompute integral table: I[n] = integral x^{2n} exp(-x^2) dx from -inf to inf
_I_table = np.zeros(30)
_I_table[0] = np.sqrt(np.pi)
for _n in range(1, 30):
    _I_table[_n] = _I_table[_n-1] * (2*_n - 1) / 2.0

def _get_P(c0, c4, c8):
    """Get P coefficients given c0, c4, c8 with c12 derived from P(0)=0."""
    c12 = -(c0 + 3.0*c4 + 105.0*c8) / 945.0
    P = c0*_H0 + c4*_H4 + c8*_H8 + c12*_H12
    return P

def _compute_bound(c0, c4, c8):
    """Compute the uncertainty bound sqrt(||xg||^2 * ||g'||^2 / ||g||^4)."""
    P = _get_P(c0, c4, c8)
    
    # Q[m] = coefficient of x^{2m} in P(x)^2
    Q = np.zeros(13)
    for i in range(7):
        for j in range(7):
            Q[i+j] += P[i] * P[j]
    
    # norm_sq = sum Q[m] * I[m]
    norm_sq = np.dot(Q, _I_table[:13])
    
    # x_norm_sq = sum Q[m] * I[m+1]
    x_norm_sq = np.dot(Q, _I_table[1:14])
    
    # R[k] = coefficient of x^{2k+1} in P'(x) - x*P(x)
    R = np.zeros(7)
    for k in range(6):
        R[k] = 2*(k+1)*P[k+1] - P[k]
    R[6] = -P[6]
    
    # S[m] = coefficient of x^{2m} in R(x)^2
    S = np.zeros(14)
    for i in range(7):
        for j in range(7):
            S[i+j] += R[i] * R[j]
    
    # deriv_norm_sq = sum S[m] * I[m]
    deriv_norm_sq = np.dot(S, _I_table[:14])
    
    if norm_sq <= 0:
        return 0.0
    
    bound = np.sqrt(x_norm_sq * deriv_norm_sq) / norm_sq
    return bound

def _compute_bound_grad(c0, c4, c8):
    """Compute bound and its gradient."""
    P = _get_P(c0, c4, c8)
    
    Q = np.zeros(13)
    for i in range(7):
        for j in range(7):
            Q[i+j] += P[i] * P[j]
    
    norm_sq = np.dot(Q, _I_table[:13])
    x_norm_sq = np.dot(Q, _I_table[1:14])
    
    R = np.zeros(7)
    for k in range(6):
        R[k] = 2*(k+1)*P[k+1] - P[k]
    R[6] = -P[6]
    
    S = np.zeros(14)
    for i in range(7):
        for j in range(7):
            S[i+j] += R[i] * R[j]
    
    deriv_norm_sq = np.dot(S, _I_table[:14])
    
    if norm_sq <= 0:
        return 0.0, np.zeros(3)
    
    bound = np.sqrt(x_norm_sq * deriv_norm_sq) / norm_sq
    if bound <= 0:
        return bound, np.zeros(3)
    
    # dP/dc0, dP/dc4, dP/dc8
    dP_dc0 = _H0 - (1.0/945.0)*_H12
    dP_dc4 = _H4 - (3.0/945.0)*_H12
    dP_dc8 = _H8 - (105.0/945.0)*_H12
    
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
    
    # d(deriv_norm_sq)/dP[j]
    d_deriv_dP = np.zeros(7)
    for j in range(7):
        s = 0.0
        for m in range(14):
            ds_dm = 0.0
            for i in range(7):
                if i + j == m:
                    # dR[i]/dP[j]
                    if i < 6:
                        dr_di = 0.0
                        if j == i+1:
                            dr_di += 2.0*(i+1)
                        if j == i:
                            dr_di -= 1.0
                    else:
                        dr_di = 0.0
                        if j == 6:
                            dr_di -= 1.0
                    
                    mi = m - i
                    if 0 <= mi < 7:
                        if mi < 6:
                            dr_mi = 0.0
                            if j == mi+1:
                                dr_mi += 2.0*(mi+1)
                            if j == mi:
                                dr_mi -= 1.0
                        else:
                            dr_mi = 0.0
                            if j == 6:
                                dr_mi -= 1.0
                        ds_dm += dr_di * R[mi] + R[i] * dr_mi
            s += ds_dm * _I_table[m]
        d_deriv_dP[j] = s
    
    # Chain rule
    A = x_norm_sq
    B = deriv_norm_sq
    C = norm_sq
    
    dA_dc0 = np.dot(d_xnorm_dP, dP_dc0)
    dA_dc4 = np.dot(d_xnorm_dP, dP_dc4)
    dA_dc8 = np.dot(d_xnorm_dP, dP_dc8)
    
    dB_dc0 = np.dot(d_deriv_dP, dP_dc0)
    dB_dc4 = np.dot(d_deriv_dP, dP_dc4)
    dB_dc8 = np.dot(d_deriv_dP, dP_dc8)
    
    dC_dc0 = np.dot(d_norm_dP, dP_dc0)
    dC_dc4 = np.dot(d_norm_dP, dP_dc4)
    dC_dc8 = np.dot(d_norm_dP, dP_dc8)
    
    # bound^2 = A*B/C^2
    # d(bound^2)/dc = (B*dA + A*dB - 2*A*B*dC/C)/C^2
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
    return _compute_bound(coeffs[0], coeffs[1], coeffs[2])

def _objective_grad(coeffs):
    return _compute_bound_grad(coeffs[0], coeffs[1], coeffs[2])

def construct():
    seed = np.array([-1.13, 0.04, 1.0/3240.0])
    best_score = _objective(seed)
    best_coeffs = seed.copy()
    
    np.random.seed(42)
    bounds = [(-10.0, 10.0), (-5.0, 5.0), (-1.0, 1.0)]
    
    # Phase 1: Multi-start L-BFGS-B with broad random initializations
    num_starts = 80
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 40:
            break
        
        if i == 0:
            x0 = seed.copy()
        elif i == 1:
            x0 = np.array([0.0, 0.0, 0.0])
        elif i == 2:
            x0 = np.array([1.0, -0.1, 0.001])
        elif i == 3:
            x0 = np.array([-1.0, 0.1, -0.001])
        else:
            x0 = np.array([
                np.random.uniform(bounds[0][0], bounds[0][1]),
                np.random.uniform(bounds[1][0], bounds[1][1]),
                np.random.uniform(bounds[2][0], bounds[2][1])
            ])
        
        try:
            res = minimize(_objective_grad, x0, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement around best solution
    for i in range(30):
        if time.time() > TIME_LIMIT - 20:
            break
        
        sigma = 0.1 * (1.0 - i/30.0) + 0.001
        x0 = best_coeffs + np.random.normal(0, sigma, size=3)
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        try:
            res = minimize(_objective_grad, x0, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: 1D line searches along each coordinate for final polish
    if time.time() < TIME_LIMIT - 10:
        for coord in range(3):
            if time.time() > TIME_LIMIT - 5:
                break
            def line_obj(t):
                x = best_coeffs.copy()
                x[coord] = t
                return _objective(x)
            
            try:
                res = minimize_scalar(line_obj, bounds=(best_coeffs[coord]-0.5, best_coeffs[coord]+0.5),
                                      method='bounded', options={'xatol': 1e-14})
                if res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x if isinstance(res.x, np.ndarray) else np.array([res.x])
                    if not isinstance(res.x, np.ndarray):
                        # minimize_scalar returns scalar x
                        new_x = best_coeffs.copy()
                        new_x[coord] = res.x
                        best_coeffs = new_x
                        best_score = _objective(best_coeffs)
            except Exception:
                pass
    
    # Final high-precision L-BFGS-B
    if time.time() < TIME_LIMIT - 5:
        try:
            res = minimize(_objective_grad, best_coeffs, method='L-BFGS-B',
                           jac=True, bounds=bounds,
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return best_coeffs
