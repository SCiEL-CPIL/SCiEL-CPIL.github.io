"""Optimized uncertainty bound using analytical moment matrices and Newton's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 180

def _get_moments(max_power):
    """Compute I[2n] = int x^(2n) e^-x^2 dx for n=0..max_power//2"""
    moments = np.zeros(max_power + 1)
    moments[0] = np.sqrt(np.pi)
    for n in range(1, max_power // 2 + 1):
        moments[2*n] = moments[2*n-2] * (2*n - 1) / (2*n)
    return moments

# Max power in P^2 is 24. In Q^2, Q is degree 13, so Q^2 is degree 26.
# We need moments up to 26.
MOMENTS = _get_moments(26)

def _build_matrices():
    """
    Build matrices M0, M2, MQ such that:
    m0 = a^T M0 a
    m2 = a^T M2 a
    n2 = a^T MQ a
    where a = [a0, a1, a2]
    """
    # Basis polynomials in terms of powers of x:
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 420x^4 - 1680x^2 + 105
    # H12 = x^12 - 132x^10 + 7920x^8 - 174720x^6 + 1663200x^4 - 4665600x^2 + 10395
    
    # Coefficients for P(x) = a0*H0 + a1*H4 + a2*H8 + a3*H12
    # a3 = -(a0 + 3a1 + 105a2) / 945
    
    # We want to express P(x) = sum c_k x^k.
    # c_k = sum_j A[k, j] * a_j for j=0,1,2.
    # Note: a3 depends on a0,a1,a2.
    
    # Let's build the mapping A (13x3) for P coefficients.
    # Powers: 0, 1, ..., 12
    A = np.zeros((13, 3))
    
    # H0 contribution
    A[0, 0] += 1.0
    # H4 contribution
    A[4, 1] += 1.0
    A[2, 1] += -6.0
    A[0, 1] += 3.0
    # H8 contribution
    A[8, 2] += 1.0
    A[6, 2] += -28.0
    A[4, 2] += 420.0
    A[2, 2] += -1680.0
    A[0, 2] += 105.0
    # H12 contribution (scaled by a3)
    # a3 = -1/945 * a0 - 3/945 * a1 - 105/945 * a2
    # a3 = -1/945 * a0 - 1/315 * a1 - 1/9 * a2
    coeff_a3_0 = -1.0 / 945.0
    coeff_a3_1 = -3.0 / 945.0
    coeff_a3_2 = -105.0 / 945.0
    
    # H12 coeffs:
    # x^12: 1
    # x^10: -132
    # x^8: 7920
    # x^6: -174720
    # x^4: 1663200
    # x^2: -4665600
    # x^0: 10395
    
    A[12, 0] += 1.0 * coeff_a3_0
    A[12, 1] += 1.0 * coeff_a3_1
    A[12, 2] += 1.0 * coeff_a3_2
    
    A[10, 0] += -132.0 * coeff_a3_0
    A[10, 1] += -132.0 * coeff_a3_1
    A[10, 2] += -132.0 * coeff_a3_2
    
    A[8, 0] += 7920.0 * coeff_a3_0
    A[8, 1] += 7920.0 * coeff_a3_1
    A[8, 2] += 7920.0 * coeff_a3_2
    
    A[6, 0] += -174720.0 * coeff_a3_0
    A[6, 1] += -174720.0 * coeff_a3_1
    A[6, 2] += -174720.0 * coeff_a3_2
    
    A[4, 0] += 1663200.0 * coeff_a3_0
    A[4, 1] += 1663200.0 * coeff_a3_1
    A[4, 2] += 1663200.0 * coeff_a3_2
    
    A[2, 0] += -4665600.0 * coeff_a3_0
    A[2, 1] += -4665600.0 * coeff_a3_1
    A[2, 2] += -4665600.0 * coeff_a3_2
    
    A[0, 0] += 10395.0 * coeff_a3_0
    A[0, 1] += 10395.0 * coeff_a3_1
    A[0, 2] += 10395.0 * coeff_a3_2

    # Now build Moment Matrices G_k for P coefficients c
    # G_k[i, j] = I[i+j+k]
    # M0 = A^T G0 A
    # M2 = A^T G2 A
    # For Q(x) = P'(x) - x P(x)
    # Q coeffs:
    # P' = sum (i+1) c_{i+1} x^i
    # xP = sum c_i x^{i+1}
    # q_j = (j+1)c_{j+1} - c_{j-1} (with c_{-1}=0, c_{13}=0)
    # This is also linear in c, and thus in a.
    # Let B be the matrix mapping a to q coeffs (length 14, indices 0..13).
    B = np.zeros((14, 3))
    for j in range(14):
        # Term from P': (j+1) c_{j+1}
        if j + 1 < 13:
            B[j, :] += (j + 1) * A[j + 1, :]
        # Term from -xP: -c_{j-1}
        if j - 1 >= 0:
            B[j, :] -= A[j - 1, :]
            
    # MQ = B^T G0 B (since weight is 1 for Q^2 integral)
    
    def make_g_matrix(offset):
        G = np.zeros((13, 13))
        for i in range(13):
            for j in range(13):
                p = i + j + offset
                if p <= 26 and p % 2 == 0:
                    G[i, j] = MOMENTS[p]
        return G

    G0 = make_g_matrix(0)
    G2 = make_g_matrix(2)
    
    M0 = A.T @ G0 @ A
    M2 = A.T @ G2 @ A
    MQ = B.T @ G0 @ B
    
    return M0, M2, MQ

M0, M2, MQ = _build_matrices()

def _objective_and_grad(a):
    """
    Compute f(a) and its gradient.
    f(a) = sqrt( (a^T M2 a) (a^T MQ a) ) / (a^T M0 a)
    We minimize f(a)^2 to avoid sqrt in gradient? 
    Let F(a) = f(a)^2 = (u * v) / w^2
    u = a^T M2 a
    v = a^T MQ a
    w = a^T M0 a
    
    grad F = [ (v * grad u + u * grad v) * w^2 - u * v * 2 * w * grad w ] / w^4
            = [ (v * grad u + u * grad v) * w - 2 * u * v * grad w ] / w^3
    
    grad u = 2 M2 a
    grad v = 2 MQ a
    grad w = 2 M0 a
    """
    u = a @ M2 @ a
    v = a @ MQ @ a
    w = a @ M0 @ a
    
    if w <= 1e-12:
        return 1e18, np.zeros(3)
        
    F = (u * v) / (w * w)
    
    if not np.isfinite(F):
        return 1e18, np.zeros(3)
        
    grad_u = 2 * (M2 @ a)
    grad_v = 2 * (MQ @ a)
    grad_w = 2 * (M0 @ a)
    
    # grad F = (v * grad_u + u * grad_v) / w^2 - 2 * u * v * grad_w / w^3
    # = [ w * (v * grad_u + u * grad_v) - 2 * u * v * grad_w ] / w^3
    
    term1 = w * (v * grad_u + u * grad_v)
    term2 = 2 * u * v * grad_w
    
    grad_F = (term1 - term2) / (w * w * w)
    
    # We want to minimize f = sqrt(F).
    # grad f = grad F / (2 * sqrt(F))
    if F <= 0:
        return 1e18, np.zeros(3)
        
    f = np.sqrt(F)
    grad_f = grad_F / (2 * f)
    
    return f, grad_f

def construct():
    # Multi-start Newton/L-BFGS-B with analytic gradients
    # Initial guesses:
    # 1. Near vacuum (1,0,0)
    # 2. Small perturbations
    # 3. Random points
    
    best_val = 1e18
    best_a = np.array([1.0, 0.0, 0.0])
    
    starts = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.1, 0.0]),
        np.array([1.0, -0.1, 0.0]),
        np.array([1.0, 0.0, 0.1]),
        np.array([1.0, 0.0, -0.1]),
        np.array([0.5, 0.5, 0.5]),
        np.array([0.5, -0.5, 0.5]),
        np.array([1.0, 0.05, 0.05]),
        np.array([1.0, -0.05, -0.05]),
    ]
    
    # Add some random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        starts.append(rng.normal(1.0, 0.5, size=3))
        
    for start in starts:
        if time.time() > _DEADLINE - 5:
            break
            
        try:
            # Use L-BFGS-B with analytic gradient
            res = minimize(
                lambda x: _objective_and_grad(x)[0],
                start,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            
            val, _ = _objective_and_grad(res.x)
            if val < best_val and np.all(np.isfinite(res.x)):
                best_val = val
                best_a = res.x
                
        except Exception:
            continue
            
    # Final polish with Nelder-Mead if time permits
    if time.time() < _DEADLINE - 2:
        try:
            res = minimize(
                lambda x: _objective_and_grad(x)[0],
                best_a,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            val, _ = _objective_and_grad(res.x)
            if val < best_val and np.all(np.isfinite(res.x)):
                best_val = val
                best_a = res.x
        except Exception:
            pass

    if not np.all(np.isfinite(best_a)):
        return np.array([1.0, 0.0, 0.0])
        
    return best_a
# EVOLVE-BLOCK-END
