# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import time


def _build_harmonics_grid():
    """Build the 70x130 quadrature grid and precompute real orthonormal Y_m."""
    k = 10
    n_theta = 70
    n_phi = 130

    # Gauss-Legendre nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(x)
    # Surface measure: dOmega = dphi * sin(theta) dtheta = dphi * (-dx)
    # integral = sum_i w_i * sum_j f(theta_i, phi_j) * (2pi/n_phi)
    # Actually: integral f dOmega = integral_0^{2pi} dphi integral_{-1}^1 f(theta,phi) dx
    # so weight for each (i,j) is w_i * (2*pi/n_phi)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    # weights[i, j] = w_i * dphi
    W = np.outer(w, np.ones(n_phi) * dphi)
    W = W.ravel()  # shape (n_theta * n_phi,)

    # Build real orthonormal harmonics Y_m for m = -10..10
    # Y_0 = N_0 * P_10^0(cos theta)
    # Y_m = sqrt(2) * N_m * P_10^m(cos theta) * cos(m*phi) for m > 0
    # Y_m = sqrt(2) * N_|m| * P_10^|m|(cos theta) * sin(|m|*phi) for m < 0
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    # P^m = P_10^m with Condon-Shortley phase (scipy lpmv includes it)

    # cos_theta, sin_theta, phi arrays
    ct = x  # shape (n_theta,)
    st = np.sin(theta)  # shape (n_theta,)

    # Precompute factorials
    def fact(n):
        r = 1.0
        for i in range(2, n + 1):
            r *= i
        return r

    Y = np.zeros((21, n_theta * n_phi))

    for idx, m in enumerate(range(-10, 11)):
        am = abs(m)
        # N_m = sqrt(21/(4*pi) * (10-am)!/(10+am)!)
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * fact(10 - am) / fact(10 + am))

        # P_10^am(cos theta) with Condon-Shortley phase
        P = lpmv(am, 10, ct)  # shape (n_theta,)

        if m == 0:
            Ym_theta = N_m * P  # shape (n_theta,)
            Ym = np.tile(Ym_theta, n_phi)
        elif m > 0:
            Ym_theta = np.sqrt(2.0) * N_m * P  # shape (n_theta,)
            # Ym[i,j] = Ym_theta[i] * cos(m * phi[j])
            Ym = np.zeros((n_theta, n_phi))
            for j in range(n_phi):
                Ym[:, j] = Ym_theta * np.cos(m * phi[j])
            Ym = Ym.ravel()
        else:
            Ym_theta = np.sqrt(2.0) * N_m * P  # shape (n_theta,)
            # Ym[i,j] = Ym_theta[i] * sin(|m| * phi[j])
            Ym = np.zeros((n_theta, n_phi))
            for j in range(n_phi):
                Ym[:, j] = Ym_theta * np.sin(am * phi[j])
            Ym = Ym.ravel()

        Y[idx, :] = Ym

    return Y, W


def _compute_R_and_grad(coeffs, Y, W):
    """Compute R = ||f||_4 / ||f||_2 and its gradient w.r.t. coeffs."""
    f = Y @ coeffs  # shape (N,)
    
    # ||f||_2^2 = sum W_i * f_i^2
    f2 = f * f
    norm2_sq = np.sum(W * f2)
    
    # ||f||_4^4 = sum W_i * f_i^4
    f4 = f2 * f2
    norm4_4 = np.sum(W * f4)
    
    if norm2_sq < 1e-30 or norm4_4 < 1e-30:
        return 1e10, np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))  # (norm4_4)^(1/4)
    
    R = norm4 / norm2
    
    # Gradient of R = norm4/norm2 w.r.t. coeffs
    # dR/df = (1/norm2) * dnorm4/df - (norm4/norm2^2) * dnorm2/df
    # dnorm4/df: norm4 = (sum W f^4)^(1/4)
    # dnorm4/df_i = (1/4) * (sum W f^4)^(-3/4) * 4 * W_i * f_i^3 = W_i * f_i^3 / norm4_4^(3/4)
    # dnorm2/df_i = (1/(2*sqrt(sum W f^2))) * 2 * W_i * f_i = W_i * f_i / norm2
    
    norm4_4_34 = norm4_4 ** 0.75
    dnorm4_df = W * f * f2 / norm4_4_34
    dnorm2_df = W * f / norm2
    
    dR_df = dnorm4_df / norm2 - (norm4 / (norm2 * norm2)) * dnorm2_df
    
    # dR/coeffs = Y^T @ dR_df
    grad = Y.T @ dR_df
    
    return R, grad


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m=-10..10 of a degree-10 spherical harmonic
    that minimize R = ||f||_4 / ||f||_2.
    
    Returns:
        coeffs: np.ndarray of shape (21,)
    """
    t_start = time.time()
    deadline = t_start + 580.0  # safety margin before 630s kill
    
    # Build quadrature grid and harmonics
    Y, W = _build_harmonics_grid()
    N = Y.shape[1]
    
    # Initialize with all ones
    coeffs = np.ones(21, dtype=np.float64)
    
    # Normalize to have ||coeffs||_2 = 1 (scale invariance)
    coeffs /= np.linalg.norm(coeffs)
    
    best_R, best_coeffs = _compute_R_and_grad(coeffs, Y, W)
    
    # Gradient descent with adaptive step size
    lr = 0.1
    min_lr = 1e-8
    max_lr = 1.0
    patience = 0
    
    while time.time() < deadline - 30.0:  # leave 30s for polish
        R, grad = _compute_R_and_grad(coeffs, Y, W)
        
        # Normalize gradient to unit norm for stable updates
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        
        # Adaptive step: try to decrease R
        step = lr * grad / gnorm
        new_coeffs = coeffs - step
        new_coeffs /= np.linalg.norm(new_coeffs)
        
        new_R, _ = _compute_R_and_grad(new_coeffs, Y, W)
        
        if new_R < R:
            coeffs = new_coeffs
            best_R = min(best_R, new_R)
            best_coeffs = coeffs.copy()
            patience = 0
            # Increase lr slightly if making good progress
            if new_R < R * 0.999:
                lr = min(max_lr, lr * 1.05)
        else:
            patience += 1
            lr = max(min_lr, lr * 0.8)
            if patience > 50:
                # Try a larger random perturbation
                noise = np.random.randn(21) * 0.1
                perturbed = coeffs + noise
                perturbed /= np.linalg.norm(perturbed)
                pert_R, _ = _compute_R_and_grad(perturbed, Y, W)
                if pert_R < R:
                    coeffs = perturbed
                    best_R = min(best_R, pert_R)
                    best_coeffs = coeffs.copy()
                    patience = 0
                    lr = 0.1
                else:
                    break
    
    # Polish pass with smaller steps
    coeffs = best_coeffs.copy()
    lr = 1e-4
    for _ in range(2000):
        if time.time() > deadline:
            break
        R, grad = _compute_R_and_grad(coeffs, Y, W)
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-14:
            break
        step = lr * grad / gnorm
        new_coeffs = coeffs - step
        new_coeffs /= np.linalg.norm(new_coeffs)
        new_R, _ = _compute_R_and_grad(new_coeffs, Y, W)
        if new_R < R:
            coeffs = new_coeffs
            best_R = new_R
            best_coeffs = coeffs.copy()
        else:
            lr *= 0.9
    
    # Final normalization check
    best_coeffs = best_coeffs / np.linalg.norm(best_coeffs)
    
    # Ensure validity: max|c| between 1e-3 and 1e5
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs *= 1e-2 / max_abs
    elif max_abs > 1e5:
        best_coeffs *= 1e4 / max_abs
    
    return best_coeffs


# EVOLVE-BLOCK-END
