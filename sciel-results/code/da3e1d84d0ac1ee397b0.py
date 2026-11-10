# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
import numpy.polynomial.legendre as npl
from scipy.optimize import minimize


def _build_basis():
    """Build real orthonormal degree-10 spherical harmonics on a quadrature grid."""
    n_theta = 70
    n_phi = 130
    
    mu, w = npl.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    basis = np.zeros((21, n_theta, n_phi), dtype=np.float64)
    
    for m_idx in range(21):
        m = m_idx - 10
        abs_m = abs(m)
        
        num = 1.0
        for i in range(10 - abs_m + 1, 11):
            num *= i
        den = 1.0
        for i in range(10 - abs_m + 1, 10 + abs_m + 1):
            den *= i
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * num / den)
        
        P = lpmv(abs_m, 10, mu)
        
        if m == 0:
            basis[m_idx] = N_m * P[:, np.newaxis] * np.ones((1, n_phi))
        elif m > 0:
            basis[m_idx] = np.sqrt(2.0) * N_m * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            basis[m_idx] = np.sqrt(2.0) * N_m * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    
    weights = np.outer(w, np.full(n_phi, dphi))
    
    return basis, weights


def _compute_R_and_grad(c, basis, weights):
    """Compute R and its gradient efficiently in one pass."""
    f = np.tensordot(c, basis, axes=(0, 0))
    
    f2 = f * f
    norm2_sq = np.sum(f2 * weights)
    
    if norm2_sq <= 0:
        return np.inf, np.zeros_like(c)
    
    f4 = f2 * f2
    norm4_4 = np.sum(f4 * weights)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    R = norm4 / norm2
    
    # Gradient: dR/dc_j
    # R = norm4 / norm2
    # dR/dc_j = (1/norm2) * d(norm4)/dc_j - (norm4/norm2^2) * d(norm2)/dc_j
    
    # d(norm4)/dc_j = (1/(4*norm4^3)) * d(norm4_4)/dc_j
    # d(norm4_4)/dc_j = 4 * sum(f^3 * Y_j * w)
    # d(norm2)/dc_j = (1/(2*norm2)) * d(norm2_sq)/dc_j
    # d(norm2_sq)/dc_j = 2 * sum(f * Y_j * w)
    
    f3 = f2 * f
    # Vectorized: int_f3_Y[j] = sum(f^3 * basis[j] * weights)
    # = sum over all grid points
    int_f3_Y = np.einsum('jkl,kl->j', basis, f3 * weights)
    int_f_Y = np.einsum('jkl,kl->j', basis, f * weights)
    
    d_norm4_4_dc = 4.0 * int_f3_Y
    d_norm4_dc = d_norm4_4_dc / (4.0 * norm4 ** 3)
    
    d_norm2_sq_dc = 2.0 * int_f_Y
    d_norm2_dc = d_norm2_sq_dc / (2.0 * norm2)
    
    grad = d_norm4_dc / norm2 - (norm4 / norm2 ** 2) * d_norm2_dc
    
    return R, grad


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing R = ||f||_4 / ||f||_2 for degree-10 spherical harmonics.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    
    basis, weights = _build_basis()
    
    def objective(c):
        R, _ = _compute_R_and_grad(c, basis, weights)
        return R
    
    def grad_fn(c):
        _, g = _compute_R_and_grad(c, basis, weights)
        return g
    
    # Multiple random restarts
    best_c = None
    best_R = np.inf
    
    # Try several initial points
    rng = np.random.default_rng(42)
    initial_guesses = [
        np.ones(21, dtype=np.float64),
        rng.standard_normal(21),
        np.zeros(21),
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(21)]),
        rng.standard_normal(21) * 0.5,
    ]
    
    # Add some structured guesses
    for m in range(-10, 11):
        guess = np.zeros(21)
        guess[m + 10] = 1.0
        initial_guesses.append(guess)
    
    for guess in initial_guesses:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                objective,
                guess,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_R:
                R_val = result.fun
                c_val = result.x
                if np.all(np.isfinite(c_val)):
                    best_R = R_val
                    best_c = c_val.copy()
        except Exception:
            continue
    
    # If no good result from L-BFGS, do a simple gradient descent
    if best_c is None or not np.all(np.isfinite(best_c)):
        c = np.ones(21, dtype=np.float64)
        lr = 0.05
        momentum = np.zeros(21, dtype=np.float64)
        beta = 0.9
        best_R = _compute_R_and_grad(c, basis, weights)[0]
        best_c = c.copy()
        
        while time.time() < deadline - 10:
            R, grad = _compute_R_and_grad(c, basis, weights)
            if R < best_R:
                best_R = R
                best_c = c.copy()
            
            momentum = beta * momentum - lr * grad
            c_new = c + momentum
            
            if not np.all(np.isfinite(c_new)):
                lr *= 0.5
                momentum *= 0.5
                continue
            
            R_new = _compute_R_and_grad(c_new, basis, weights)[0]
            if R_new < R:
                c = c_new
            else:
                lr *= 0.9
                momentum *= 0.5
    
    if best_c is None or not np.all(np.isfinite(best_c)):
        return np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1.0 / max(1e-3, max_abs))
    elif max_abs > 1e5:
        best_c = best_c * (1e5 / max_abs)
    
    return best_c


# EVOLVE-BLOCK-END
