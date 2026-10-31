"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


# Precompute Gauss-Hermite nodes and weights (15 points is exact for degree <= 29)
_N_GQ = 15
_X_GQ, _W_GQ = np.polynomial.hermite.hermgauss(_N_GQ)
_X2 = _X_GQ * _X_GQ
_H0 = np.ones(_N_GQ)
_H4 = _X2**2 - 6 * _X2 + 3
_H8 = _X2**4 - 28 * _X2**3 + 210 * _X2**2 - 420 * _X2 + 105
_H12 = _X2**6 - 84 * _X2**5 + 1260 * _X2**4 - 12600 * _X2**3 + 55440 * _X2**2 - 100800 * _X2 + 10395

# Precompute for derivative: P'(x) terms
_X3 = _X_GQ**3
_X5 = _X_GQ**5
_X7 = _X_GQ**7
_X9 = _X_GQ**9
_X11 = _X_GQ**11
# Derivative of H4: 4x^3 - 12x
_DH4 = 4 * _X3 - 12 * _X_GQ
# Derivative of H8: 8x^7 - 168x^5 + 840x^3 - 840x
_DH8 = 8 * _X7 - 168 * _X5 + 840 * _X3 - 840 * _X_GQ
# Derivative of H12: 12x^11 - 840x^9 + 10080x^7 - 75600x^5 + 221760x^3 - 201600x
_DH12 = 12 * _X11 - 840 * _X9 + 10080 * _X7 - 75600 * _X5 + 221760 * _X3 - 201600 * _X_GQ

# Weights for integrals: w_i * 1, w_i * x_i^2
_W_I0 = _W_GQ
_W_I2 = _W_GQ * _X2


def _compute_integrals(a, b, c):
    """Compute I0, I2, J0 using precomputed Gauss-Hermite nodes.
    Returns (I0, I2, J0) and also P, P_prime for gradient computation."""
    d = -(a + 3 * b + 105 * c) / 10395.0
    
    P = a * _H0 + b * _H4 + c * _H8 + d * _H12
    I0 = np.dot(_W_I0, P * P)
    I2 = np.dot(_W_I2, P * P)
    
    P_prime = b * _DH4 + c * _DH8 + d * _DH12
    g_prime_poly = P_prime - _X_GQ * P
    J0 = np.dot(_W_I0, g_prime_poly * g_prime_poly)
    
    return I0, I2, J0, P, P_prime, d


def _build_quadratic_forms():
    """Build the 3x3 quadratic form matrices for I0, I2, and J0.
    
    The key insight: for Hermite-Gaussian functions, the Fourier transform
    of H_{2n}(x) e^{-x^2/2} is (-1)^n H_{2n}(k) e^{-k^2/2}, so the
    momentum-space integral |hat{g}(k)|^2 has the same form as |g(x)|^2.
    
    However, A(F)^2 involves <k^2> = integral k^2 |hat{g}|^2 dk / integral |hat{g}|^2 dk
    which equals I2/I0 (same as position). But the uncertainty product uses
    A(f)*A(F) where A(f) = sqrt(<x^2> - <x>^2) and A(F) = sqrt(<k^2> - <k>^2).
    
    Wait - let me reconsider. The task says A(f)*A(Fourier(f)). For the
    Hermite-Gaussian, the Fourier transform preserves the structure. The
    momentum variance <k^2> = integral k^2 |hat{g}(k)|^2 dk / integral |hat{g}(k)|^2 dk.
    
    Since hat{g}(k) = (-1)^n terms * g(k) (for even Hermite), |hat{g}(k)|^2 = |g(k)|^2,
    so <k^2> = <x^2> = I2/I0.
    
    But that would make A(f)*A(F) = I2/I0, not sqrt(I2*J0)/I0.
    
    Let me re-examine. The J0 in the seed is integral (P' - x*P)^2 e^{-x^2} dx,
    which is the variance of the derivative, not the momentum variance directly.
    
    Actually, for the uncertainty principle: A(f)^2 A(F)^2 >= 1/4.
    A(f)^2 = <x^2> - <x>^2 = I2/I0 (since <x>=0 by symmetry)
    A(F)^2 = <k^2> - <k>^2
    
    For g(x) = P(x)e^{-x^2/2}, the Fourier transform is:
    hat{g}(k) = (1/sqrt(2pi)) integral P(x) e^{-x^2/2} e^{-ikx} dx
    
    For even P, hat{g}(k) is real and even. The momentum variance:
    <k^2> = integral k^2 |hat{g}(k)|^2 dk / integral |hat{g}(k)|^2 dk
    
    Using the property that for Hermite polynomials, the Fourier transform
    relates to the same polynomial: if g = sum c_n H_{2n} e^{-x^2/2}, then
    hat{g} = sum (-1)^n c_n H_{2n} e^{-k^2/2} (up to normalization).
    
    So |hat{g}(k)|^2 involves cross terms c_n c_m (-1)^{n+m} H_{2n}(k) H_{2m}(k).
    And k^2 |hat{g}(k)|^2 involves k^2 H_{2n}(k) H_{2m}(k).
    
    This is NOT simply I2/I0 because of the (-1)^n factors creating sign changes
    in cross terms.
    
    So J0 as defined in the seed (integral (P'-xP)^2 e^{-x^2} dx) is actually
    related to the momentum variance through the relation:
    A(F)^2 = (1/I0) * integral (P'(x) - x P(x))^2 e^{-x^2} dx / (2 * something)
    
    Actually, let me just verify numerically that J0 = I2 for this setup, or not.
    """
    pass


def _compute_quadratic_form_matrices():
    """Compute the 3x3 symmetric matrices M0, M2, MJ such that:
    I0 = v^T M0 v, I2 = v^T M2 v, J0 = v^T MJ v
    where v = [a, b, c].
    
    Each integral is quadratic in (a,b,c) because:
    P = a*H0 + b*H4 + c*H8 + d*H12, d = -(a+3b+105c)/10395
    So P = sum_j v_j * H_j where H_j are the effective basis functions.
    """
    # Effective basis vectors: each is a function of x
    # H_eff_0 = H0 - H12/10395
    # H_eff_1 = H4 - 3*H12/10395
    # H_eff_2 = H8 - 105*H12/10395
    
    H_eff0 = _H0 - _H12 / 10395.0
    H_eff1 = _H4 - 3 * _H12 / 10395.0
    H_eff2 = _H8 - 105 * _H12 / 10395.0
    
    # I0 = sum w_i * (sum_j v_j * H_eff_j[i])^2 = v^T M0 v
    # M0[j,k] = sum_i w_i * H_eff_j[i] * H_eff_k[i]
    H_eff = np.column_stack([H_eff0, H_eff1, H_eff2])  # shape (N_GQ, 3)
    
    M0 = H_eff.T @ (_W_I0[:, None] * H_eff)  # M0[j,k] = sum_i W_I0[i] * H_eff[j,i] * H_eff[k,i]
    M2 = H_eff.T @ (_W_I2[:, None] * H_eff)
    
    # For J0: g_prime_poly = P_prime - x*P
    # P_prime = b*DH4 + c*DH8 + d*DH12
    # dP_prime/da = -DH12/10395, dP_prime/db = DH4 - 3*DH12/10395, dP_prime/dc = DH8 - 105*DH12/10395
    # x*P: x*H_eff0, x*H_eff1, x*H_eff2
    # g_prime_poly = sum_j v_j * (dPp_eff_j - x * H_eff_j)
    
    dPp_eff0 = -_DH12 / 10395.0
    dPp_eff1 = _DH4 - 3 * _DH12 / 10395.0
    dPp_eff2 = _DH8 - 105 * _DH12 / 10395.0
    
    gp_eff0 = dPp_eff0 - _X_GQ * H_eff0
    gp_eff1 = dPp_eff1 - _X_GQ * H_eff1
    gp_eff2 = dPp_eff2 - _X_GQ * H_eff2
    
    gp_eff = np.column_stack([gp_eff0, gp_eff1, gp_eff2])
    MJ = gp_eff.T @ (_W_I0[:, None] * gp_eff)
    
    # Symmetrize
    M0 = (M0 + M0.T) / 2
    M2 = (M2 + M2.T) / 2
    MJ = (MJ + MJ.T) / 2
    
    return M0, M2, MJ


def _objective_from_matrices(v, M0, M2, MJ):
    """Compute objective and gradient from precomputed quadratic form matrices."""
    I0 = v @ M0 @ v
    I2 = v @ M2 @ v
    J0 = v @ MJ @ v
    
    if I0 <= 1e-15 or J0 <= 1e-15 or I2 <= 1e-15:
        return 1e10, np.zeros(3)
    
    product = np.sqrt(I2 * J0) / I0
    
    # d(product)/dv = (dI2 * J0 + I2 * dJ0) / (2 * I2 * J0) - dI0 / I0
    dI0 = 2 * M0 @ v
    dI2 = 2 * M2 @ v
    dJ0 = 2 * MJ @ v
    
    grad = (dI2 * J0 + I2 * dJ0) / (2.0 * I2 * J0) - dI0 / I0
    
    return product, grad


def construct():
    start_time = time.time()
    deadline = 195.0  # seconds (safety margin before 210s kill)
    
    # Build quadratic form matrices once
    M0, M2, MJ = _compute_quadratic_form_matrices()
    
    # Seed value
    seed_result = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_params = seed_result.copy()
    best_score, _ = _objective_from_matrices(best_params, M0, M2, MJ)
    
    # Strategy 1: Solve the generalized eigenvalue problem
    # Minimize sqrt(I2 * J0) / I0 over v != 0
    # Equivalent to minimizing I2 * J0 / I0^2
    # At optimum: d(I2*J0/I0^2)/dv = 0
    # This leads to: (I2*MJ + J0*M2) v = 2*I2*J0/I0 * M0 v
    # Let lambda = I2*J0/I0^2, then: (I2*MJ + J0*M2) v = 2*lambda * I0 * M0 v
    # But I2 = v^T M2 v, J0 = v^T MJ v, I0 = v^T M0 v are not known a priori.
    # 
    # Alternative: minimize f = I2*J0/I0^2. Use the fact that at optimum,
    # the gradient is zero. We can use iterative refinement.
    #
    # Better approach: For a fixed ratio, we can solve a generalized eigenvalue problem.
    # Let's use the substitution: minimize I2*J0 subject to I0 = 1.
    # This is a non-convex problem, but we can use the L-BFGS approach with
    # the matrix-based gradient which is much faster.
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    start_points = [best_params.copy()]
    
    # Perturbed versions of the seed
    for i in range(30):
        perturbation = rng.randn(3) * 0.1
        start_points.append(seed_result + perturbation)
    
    # Random points in reasonable range
    for i in range(30):
        start_points.append(rng.randn(3) * 0.5)
    
    # Structured grid points
    for a_val in [-2, -1.5, -1, -0.5, -0.2, 0, 0.2, 0.5, 1]:
        for b_val in [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5]:
            start_points.append(np.array([a_val, b_val, 0.01]))
    
    # Additional structured points based on the seed ratios
    for scale in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
        start_points.append(seed_result * scale)
    
    # Optimize from each starting point using L-BFGS-B with analytic gradient
    for sp in start_points:
        if time.time() > deadline - 15:
            break
        
        try:
            result = minimize(
                lambda x: _objective_from_matrices(x, M0, M2, MJ)[0],
                sp,
                jac=lambda x: _objective_from_matrices(x, M0, M2, MJ)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Refine with tighter tolerances
    if time.time() < deadline - 20:
        try:
            result = minimize(
                lambda x: _objective_from_matrices(x, M0, M2, MJ)[0],
                best_params,
                jac=lambda x: _objective_from_matrices(x, M0, M2, MJ)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Additional refinement: try small perturbations around the best
    if time.time() < deadline - 10:
        for i in range(50):
            if time.time() > deadline - 5:
                break
            perturbation = rng.randn(3) * 1e-4
            candidate = best_params + perturbation
            score, _ = _objective_from_matrices(candidate, M0, M2, MJ)
            if score < best_score:
                best_score = score
                best_params = candidate.copy()
    
    # Final verification
    final_score, _ = _objective_from_matrices(best_params, M0, M2, MJ)
    if final_score < best_score:
        best_score = final_score
    
    # Ensure we don't return worse than seed
    seed_score, _ = _objective_from_matrices(seed_result, M0, M2, MJ)
    if best_score > seed_score:
        best_params = seed_result.copy()
    
    return best_params
# EVOLVE-BLOCK-END
