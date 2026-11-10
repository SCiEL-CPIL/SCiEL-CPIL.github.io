# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _build_basis():
    """Build the 70x130 quadrature grid and 21x9100 basis matrix."""
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    # Transform to [0, 1] for theta
    theta = np.arccos(x)
    w_theta = w * 0.5  # Jacobian of cos(theta) = -sin(theta), but we integrate over cos(theta)
    
    # Actually, the quadrature is over cos(theta) directly.
    # dOmega = sin(theta) dtheta dphi = -d(cos(theta)) dphi
    # So integral = sum_i w_i f(cos(theta_i), phi_j) dphi
    # With 130 equally spaced phi points, dphi = 2*pi/130
    
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Build grid
    mu = x  # cos(theta)
    # Weights for the mu integration
    w_mu = w * 0.5  # This is for integral over mu in [-1,1]
    
    # Total weight for each (mu_i, phi_j) point
    # Integral = sum_{i,j} w_mu[i] * dphi * f(mu_i, phi_j)
    # But wait, the standard surface measure dOmega = sin(theta)dtheta dphi
    # If we substitute mu = cos(theta), then dOmega = dmu dphi
    # So the integral over the sphere is ∫∫ f(mu, phi) dmu dphi
    # The Gauss-Legendre quadrature on [-1,1] gives ∫_{-1}^{1} g(mu) dmu ≈ sum_i w_i g(mu_i)
    # So we don't need the 0.5 factor if w_i are the standard GL weights.
    
    # Let me reconsider. np.polynomial.legendre.leggauss(n) returns nodes x and weights w
    # such that ∫_{-1}^{1} f(x) dx ≈ Σ w_i f(x_i)
    # So for our integral ∫_{-1}^{1} ∫_{0}^{2π} f(mu, phi) dphi dmu
    # ≈ Σ_i w_i Σ_j f(mu_i, phi_j) * dphi
    
    # So the weight for each point is w_i * dphi
    
    # Precompute basis functions
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m|(cos theta) sin(|m| phi) for m < 0
    
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    
    # Compute associated Legendre polynomials
    def assoc_legendre(l, m, x):
        """Compute P_l^m(x) with Condon-Shortley phase."""
        if m < 0:
            return 0
        if m > l:
            return 0
        
        # Compute P_m^m(x)
        P = np.ones_like(x)
        if m > 0:
            for i in range(1, m + 1):
                P *= -np.sqrt((2 * m - 1) ** 2 - x ** 2)
            P *= (-1) ** m
        
        if m == l:
            return P
        
        # Compute P_{m+1}^m(x)
        Pm1 = x * (2 * m + 1) * P
        
        if m + 1 == l:
            return Pm1
        
        # Compute higher orders
        for n in range(m + 2, l + 1):
            Pn = (x * (2 * n - 1) * Pm1 - (n + m - 1) * P) / (n - m)
            P = Pm1
            Pm1 = Pn
        
        return Pm1
    
    # Precompute all basis values on the grid
    # Shape: (21, 70, 130)
    basis = np.zeros((2 * k + 1, n_theta, n_phi))
    
    for m_idx in range(2 * k + 1):
        m = m_idx - k  # m ranges from -10 to 10
        abs_m = abs(m)
        
        # Compute normalization
        N_m = math.sqrt(21.0 / (4.0 * np.pi) * math.factorial(k - abs_m) / math.factorial(k + abs_m))
        
        # Compute P_10^{|m|}(mu)
        P = assoc_legendre(k, abs_m, mu)
        
        if m == 0:
            basis[m_idx] = N_m * P[:, np.newaxis] * np.ones((1, n_phi))
        elif m > 0:
            basis[m_idx] = np.sqrt(2) * N_m * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:  # m < 0
            basis[m_idx] = np.sqrt(2) * N_m * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    
    # Flatten to (21, 9100)
    basis_flat = basis.reshape(21, n_theta * n_phi)
    
    # Weights: w_i * dphi for each point
    w_flat = np.repeat(w_mu, n_phi) * dphi
    
    return basis_flat, w_flat


def _compute_R_and_grad(coeffs, basis, w):
    """Compute R = ||f||_4 / ||f||_2 and its gradient."""
    # f = sum_m c_m Y_m
    f = coeffs @ basis  # shape (9100,)
    
    # ||f||_2^2 = sum w_i f_i^2
    f2 = f * f
    norm2_sq = np.sum(w * f2)
    
    # ||f||_4^4 = sum w_i f_i^4
    f4 = f2 * f2
    norm4_4 = np.sum(w * f4)
    
    if norm2_sq <= 0 or norm4_4 <= 0:
        return np.inf, np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))  # ||f||_4 = (sum w f^4)^{1/4}
    
    R = norm4 / norm2
    
    # Gradient of R = dR/dc
    # R = norm4 / norm2
    # dR/dc = (1/norm2) * dnorm4/dc - (norm4/norm2^2) * dnorm2/dc
    # dnorm4/dc_m = (1/4) * (sum w f^4)^{-3/4} * sum w 4 f^3 Y_m
    #             = (sum w f^3 Y_m) / norm4^3
    # dnorm2/dc_m = (1/2) * (sum w f^2)^{-1/2} * sum w 2 f Y_m
    #             = (sum w f Y_m) / norm2
    
    # sum w f^3 Y_m for all m
    w_f3 = w * f * f2  # w * f^3
    grad_norm4 = basis.T @ w_f3 / (norm4 ** 3)
    
    # sum w f Y_m for all m
    w_f = w * f
    grad_norm2 = basis.T @ w_f / norm2
    
    grad_R = grad_norm4 / norm2 - (norm4 / norm2 ** 2) * grad_norm2
    
    return R, grad_R


def _adam_step(coeffs, grad, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer step."""
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    update = lr * m_hat / (np.sqrt(v_hat) + eps)
    return coeffs - update, m, v


def _gradient_descent_with_adam(coeffs_init, basis, w, max_steps=5000, time_limit=None, start_time=None):
    """Run Adam gradient descent with adaptive learning rate."""
    if start_time is None:
        start_time = time.time()
    
    coeffs = coeffs_init.copy()
    m = np.zeros_like(coeffs)
    v = np.zeros_like(coeffs)
    
    best_coeffs = coeffs.copy()
    best_R = _compute_R_and_grad(coeffs, basis, w)[0]
    
    lr = 0.01
    patience = 0
    max_patience = 200
    
    for step in range(max_steps):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        R, grad = _compute_R_and_grad(coeffs, basis, w)
        
        if np.isfinite(R) and R < best_R:
            best_R = R
            best_coeffs = coeffs.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            # Reduce learning rate
            lr *= 0.5
            patience = 0
            if lr < 1e-6:
                break
        
        coeffs, m, v = _adam_step(coeffs, grad, m, v, step + 1, lr=lr)
        
        # Clip coefficients to avoid extreme values
        max_abs = np.max(np.abs(coeffs))
        if max_abs > 1e5:
            coeffs = coeffs * (1e5 / max_abs)
        if max_abs < 1e-3 and max_abs > 0:
            coeffs = coeffs * (1e-3 / max_abs)
    
    return best_coeffs, best_R


def _newton_like_refinement(coeffs, basis, w, max_steps=100, time_limit=None, start_time=None):
    """Newton-like refinement using approximate Hessian."""
    if start_time is None:
        start_time = time.time()
    
    coeffs = coeffs.copy()
    best_coeffs = coeffs.copy()
    best_R = _compute_R_and_grad(coeffs, basis, w)[0]
    
    for step in range(max_steps):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        R, grad = _compute_R_and_grad(coeffs, basis, w)
        
        if not np.isfinite(R):
            break
        
        if R < best_R:
            best_R = R
            best_coeffs = coeffs.copy()
        
        # Simple momentum-based step with line search
        step_size = 0.01
        improved = False
        for _ in range(20):
            new_coeffs = coeffs - step_size * grad
            new_R, _ = _compute_R_and_grad(new_coeffs, basis, w)
            if np.isfinite(new_R) and new_R < R:
                coeffs = new_coeffs
                improved = True
                break
            step_size *= 0.5
        
        if not improved:
            break
    
    return best_coeffs, best_R


def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length
    2*k + 1 = 21.

    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients
            c_m, ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    start_time = time.time()
    time_limit = 600  # seconds, with safety margin
    
    # Build basis
    basis, w = _build_basis()
    
    k = 10
    n_coeffs = 2 * k + 1
    
    # Generate multiple initial guesses
    initial_guesses = []
    
    # 1. All ones
    initial_guesses.append(np.ones(n_coeffs))
    
    # 2. Only m=0
    c = np.zeros(n_coeffs)
    c[k] = 1.0
    initial_guesses.append(c)
    
    # 3. Only m=10
    c = np.zeros(n_coeffs)
    c[20] = 1.0
    initial_guesses.append(c)
    
    # 4. Only m=-10
    c = np.zeros(n_coeffs)
    c[0] = 1.0
    initial_guesses.append(c)
    
    # 5. Alternating signs
    c = np.ones(n_coeffs)
    c[::2] = -1.0
    initial_guesses.append(c)
    
    # 6. Random with fixed seed
    rng = np.random.RandomState(42)
    for _ in range(5):
        c = rng.randn(n_coeffs)
        c /= np.linalg.norm(c)
        initial_guesses.append(c)
    
    # 7. Structured: emphasize low m
    c = np.zeros(n_coeffs)
    for i, m in enumerate(range(-10, 11)):
        c[i] = 1.0 / (1.0 + abs(m) * 0.5)
    initial_guesses.append(c)
    
    # 8. Structured: emphasize high m
    c = np.zeros(n_coeffs)
    for i, m in enumerate(range(-10, 11)):
        c[i] = 1.0 / (1.0 + (10 - abs(m)) * 0.5)
    initial_guesses.append(c)
    
    # Run optimization from each initial guess
    best_coeffs = None
    best_R = np.inf
    
    for idx, c_init in enumerate(initial_guesses):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 10:  # Need at least 10 seconds
            break
        
        # Allocate time for this restart
        time_per_restart = remaining / (len(initial_guesses) - idx)
        adam_time = min(time_per_restart * 0.7, 300)
        refine_time = min(time_per_restart * 0.3, 100)
        
        # Adam gradient descent
        coeffs, R = _gradient_descent_with_adam(
            c_init, basis, w,
            max_steps=3000,
            time_limit=adam_time,
            start_time=time.time()
        )
        
        if R < best_R:
            best_R = R
            best_coeffs = coeffs.copy()
        
        # Newton-like refinement
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining > 5:
            coeffs, R = _newton_like_refinement(
                coeffs, basis, w,
                max_steps=100,
                time_limit=min(refine_time, remaining - 5),
                start_time=time.time()
            )
            
            if R < best_R:
                best_R = R
                best_coeffs = coeffs.copy()
    
    # If no valid result found, return all ones
    if best_coeffs is None or not np.isfinite(best_R):
        return np.ones(n_coeffs)
    
    # Validate
    if len(best_coeffs) != 21:
        return np.ones(n_coeffs)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(n_coeffs)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Rescale
        if max_abs > 0:
            target = 1.0
            best_coeffs = best_coeffs * (target / max_abs)
        else:
            return np.ones(n_coeffs)
    
    return best_coeffs


# EVOLVE-BLOCK-END
