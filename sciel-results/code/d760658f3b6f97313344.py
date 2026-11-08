import numpy as np
from scipy.optimize import minimize
import time

# Precompute Hermite polynomial coefficients (physicists' convention)
# H0 = 1
# H4 = 16x^4 - 48x^2 + 12
# H8 = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 3360
# H12 = 4096x^12 - 98304x^10 + 846720x^8 - 3932160x^6 + 8064000x^4 - 6048000x^2 + 1039500

H0_coeffs = np.array([1.0])
H4_coeffs = np.array([12.0, -48.0, 16.0])
H8_coeffs = np.array([3360.0, -13440.0, 11520.0, -3072.0, 256.0])
H12_coeffs = np.array([1039500.0, -6048000.0, 8064000.0, -3932160.0, 846720.0, -98304.0, 4096.0])

# Precompute integral tables
# I1[n] = ∫ x^(2n) * exp(-x²) dx for n = 0..24 (max needed: 4*6=24 for L4)
# I2[n] = ∫ x^(2n) * exp(-2x²) dx for n = 0..24

def _precompute_integrals():
    """Precompute integral tables for moments."""
    I1 = np.zeros(25)
    I2 = np.zeros(25)
    for n in range(25):
        if n == 0:
            I1[n] = np.sqrt(np.pi)
            I2[n] = np.sqrt(np.pi / 2.0)
        else:
            # (2n-1)!! = product of odd numbers from 1 to 2n-1
            double_fact = 1.0
            for k in range(n):
                double_fact *= (2*k + 1)
            I1[n] = double_fact * np.sqrt(np.pi) / (2.0 ** n)
            I2[n] = double_fact * np.sqrt(np.pi) / (2.0 ** (2*n + 0.5))
    return I1, I2

I1_table, I2_table = _precompute_integrals()

def _compute_P_coeffs(c0, c1, c2, c3):
    """Compute the monomial coefficients of P(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12."""
    P = np.zeros(7)
    P[0] += c0 * H0_coeffs[0]
    P[0] += c1 * H4_coeffs[0]
    P[1] += c1 * H4_coeffs[1]
    P[2] += c1 * H4_coeffs[2]
    for i in range(5):
        P[i] += c2 * H8_coeffs[i]
    for i in range(7):
        P[i] += c3 * H12_coeffs[i]
    return P

def _poly_square_moment_vec(P):
    """Compute ∫ P(x)² * exp(-x²) dx using precomputed table."""
    # P(x)² = sum_i sum_j P[i]*P[j]*x^(2(i+j))
    n = len(P)
    # Build index matrix
    idx = np.add.outer(np.arange(n), np.arange(n))
    return np.sum(P[:, None] * P[None, :] * I1_table[idx])

def _poly_fourth_moment_vec(P):
    """Compute ∫ P(x)⁴ * exp(-2x²) dx using precomputed table."""
    n = len(P)
    # P(x)⁴ = sum_{i,j,k,l} P[i]*P[j]*P[k]*P[l]*x^(2(i+j+k+l))
    # Vectorized: use outer products
    P2 = P[:, None] * P[None, :]  # P[i]*P[j]
    P4 = P2[:, :, None, None] * P2[None, None, :, :]  # P[i]*P[j]*P[k]*P[l]
    idx = np.add.outer(np.add.outer(np.arange(n), np.arange(n)), np.add.outer(np.arange(n), np.arange(n)))
    # idx[i,j,k,l] = i+j+k+l
    idx4 = np.zeros((n,n,n,n), dtype=int)
    for i in range(n):
        for j in range(n):
            idx4[i,j,:,:] = i + j + np.add.outer(np.arange(n), np.arange(n))
    return np.sum(P4 * I2_table[idx4])

def _objective_and_grad(c0, c1, c2):
    """Compute the objective (upper bound on C4) and its gradient."""
    c3 = -(c0 + 12*c1 + 3360*c2) / 1039500.0
    
    P = _compute_P_coeffs(c0, c1, c2, c3)
    
    L2_sq = _poly_square_moment_vec(P)
    L4_4th = _poly_fourth_moment_vec(P)
    
    if L2_sq <= 0:
        return np.inf, np.zeros(3)
    
    C4 = np.sqrt(L4_4th) / L2_sq
    
    # Derivatives of P with respect to c0, c1, c2
    dP_dc0 = _compute_P_coeffs(1, 0, 0, 0) + (-1/1039500.0) * _compute_P_coeffs(0, 0, 0, 1)
    dP_dc1 = _compute_P_coeffs(0, 1, 0, 0) + (-12/1039500.0) * _compute_P_coeffs(0, 0, 0, 1)
    dP_dc2 = _compute_P_coeffs(0, 0, 1, 0) + (-3360/1039500.0) * _compute_P_coeffs(0, 0, 0, 1)
    
    # dL2_sq/dc = 2 * sum_{i,j} dP[i] * P[j] * I1[i+j]
    n = len(P)
    idx2 = np.add.outer(np.arange(n), np.arange(n))
    
    def _dL2_sq(dP):
        return 2.0 * np.sum(dP[:, None] * P[None, :] * I1_table[idx2])
    
    # dL4_4th/dc = 4 * sum_{i,j,k,l} dP[i] * P[j]*P[k]*P[l] * I2[i+j+k+l]
    def _dL4_4th(dP):
        P3 = P[:, None, None] * P[None, :, None] * P[None, None, :]  # P[j]*P[k]*P[l]
        idx3 = np.zeros((n,n,n,n), dtype=int)
        for i in range(n):
            for j in range(n):
                idx3[i,j,:,:] = i + j + np.add.outer(np.arange(n), np.arange(n))
        return 4.0 * np.sum(dP[:, None, None, None] * P3[None, :, :, :] * I2_table[idx3])
    
    dL2_sq_dc0 = _dL2_sq(dP_dc0)
    dL2_sq_dc1 = _dL2_sq(dP_dc1)
    dL2_sq_dc2 = _dL2_sq(dP_dc2)
    
    dL4_4th_dc0 = _dL4_4th(dP_dc0)
    dL4_4th_dc1 = _dL4_4th(dP_dc1)
    dL4_4th_dc2 = _dL4_4th(dP_dc2)
    
    sqrt_L4 = np.sqrt(L4_4th)
    dC4_dc0 = (0.5 * dL4_4th_dc0 / sqrt_L4 - dL2_sq_dc0 / L2_sq) * C4
    dC4_dc1 = (0.5 * dL4_4th_dc1 / sqrt_L4 - dL2_sq_dc1 / L2_sq) * C4
    dC4_dc2 = (0.5 * dL4_4th_dc2 / sqrt_L4 - dL2_sq_dc2 / L2_sq) * C4
    
    return C4, np.array([dC4_dc0, dC4_dc1, dC4_dc2])

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 180s limit
    
    best_C4 = np.inf
    best_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    rng = np.random.RandomState(42)
    starts = [best_c.copy()]
    
    # Diverse starting points
    for scale in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
        for _ in range(4):
            perturbation = rng.normal(0, scale, 3)
            starts.append(best_c + perturbation)
    
    # Random wide-range starts
    for _ in range(30):
        starts.append(rng.uniform(-1000, 1000, 3))
    
    # Evaluate and sort starting points
    start_objs = []
    for s in starts:
        try:
            obj, _ = _objective_and_grad(s[0], s[1], s[2])
            start_objs.append((obj, s))
        except:
            start_objs.append((np.inf, s))
    start_objs.sort(key=lambda x: x[0])
    
    # Multi-start BFGS
    for obj, s in start_objs:
        if time.time() > deadline - 10.0:
            break
        if obj >= best_C4:
            continue
        
        try:
            result = minimize(
                lambda c: _objective_and_grad(c[0], c[1], c[2])[0],
                s,
                jac=lambda c: _objective_and_grad(c[0], c[1], c[2])[1],
                method='BFGS',
                options={'maxiter': 300, 'gtol': 1e-12}
            )
            if result.success and result.fun < best_C4:
                best_C4 = result.fun
                best_c = result.x
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                lambda c: _objective_and_grad(c[0], c[1], c[2])[0],
                best_c,
                jac=lambda c: _objective_and_grad(c[0], c[1], c[2])[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-15, 'ftol': 1e-15}
            )
            if result.success and result.fun < best_C4:
                best_C4 = result.fun
                best_c = result.x
        except Exception:
            pass
    
    return np.array(best_c)
