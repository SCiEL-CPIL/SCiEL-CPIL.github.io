"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Pre-compute Gauss-Hermite quadrature nodes and weights
_N_Q = 300
_x_gh, _w_gh = np.polynomial.hermite.hermgauss(_N_Q)
_w_gh *= np.sqrt(np.pi)  # Scale for exp(-x^2) weight

# Hermite polynomials evaluated at quadrature nodes (pre-compute for speed)
_x2 = _x_gh ** 2
_H0 = np.ones_like(_x_gh)
_H4 = 16*_x2**2 - 48*_x2 + 12
_H8 = 256*_x2**4 - 3072*_x2**3 + 11520*_x2**2 - 13440*_x2 + 1680
_H12 = (4096*_x2**6 - 73728*_x2**5 + 580608*_x2**4 - 2419200*_x2**3 
        + 5443200*_x2**2 - 5443200*_x2 + 2032128)

# Pre-compute derivatives of Hermite polynomials at quadrature nodes
_dH4 = 64*_x_gh**3 - 96*_x_gh
_dH8 = 2048*_x_gh**7 - 8064*_x_gh**5 + 10752*_x_gh**3 - 1344*_x_gh
_dH12 = (49152*_x_gh**11 - 322560*_x_gh**9 + 677376*_x_gh**7 
         - 653184*_x_gh**5 + 258048*_x_gh**3 - 14336*_x_gh)

# Constraint: a12 = -(a0 + 12*a4 + 1680*a8) / 2032128
# P = a0*H0 + a4*H4 + a8*H8 + a12*H12
#   = a0*(H0 - H12/2032128) + a4*(H4 - 12*H12/2032128) + a8*(H8 - 1680*H12/2032128)
# Define effective basis functions:
_B0 = _H0 - (1.0/2032128.0)*_H12
_B4 = _H4 - (12.0/2032128.0)*_H12
_B8 = _H8 - (1680.0/2032128.0)*_H12

# Derivatives of effective basis:
_dB0 = -(1.0/2032128.0)*_dH12
_dB4 = _dH4 - (12.0/2032128.0)*_dH12
_dB8 = _dH8 - (1680.0/2032128.0)*_dH12

# For the Fourier side: f(x) = P(x)*exp(-x^2/2)
# f'(x) = (P'(x) - x*P(x))*exp(-x^2/2)
# The "momentum" operator on the polynomial part is: D = d/dx - x
# So for basis B_i: D(B_i) = dB_i - x*B_i
_DB0 = _dB0 - _x_gh*_B0
_DB4 = _dB4 - _x_gh*_B4
_DB8 = _dB8 - _x_gh*_B8

# Build Gram matrices using quadrature
# G_norm[i,j] = <B_i, B_j> = integral B_i * B_j * exp(-x^2) dx
# G_x[i,j] = <x*B_i, x*B_j> = integral x^2 * B_i * B_j * exp(-x^2) dx
# G_mom[i,j] = <D(B_i), D(B_j)> = integral (dB_i - x*B_i)*(dB_j - x*B_j)*exp(-x^2) dx

_B_mat = np.column_stack([_B0, _B4, _B8])  # shape (N_Q, 3)
_DB_mat = np.column_stack([_DB0, _DB4, _DB8])
_xB_mat = _x_gh[:, np.newaxis] * _B_mat  # shape (N_Q, 3)

G_norm = (_B_mat * _w_gh[:, np.newaxis]).T @ _B_mat
G_x = (_xB_mat * _w_gh[:, np.newaxis]).T @ _xB_mat
G_mom = (_DB_mat * _w_gh[:, np.newaxis]).T @ _DB_mat

def _compute_bound_from_coeffs(a0, a4, a8):
    """Compute the uncertainty product using Gram matrices."""
    a = np.array([a0, a4, a8])
    norm2 = a @ G_norm @ a
    if norm2 < 1e-30:
        return 1e10
    var_x = (a @ G_x @ a) / norm2
    var_p = (a @ G_mom @ a) / norm2
    if var_x < 0 or var_p < 0:
        return 1e10
    return np.sqrt(max(var_x, 0.0) * max(var_p, 0.0))

def _compute_bound(a0, a4, a8):
    """Compute the uncertainty product (upper bound on C4) for given coefficients."""
    return _compute_bound_from_coeffs(a0, a4, a8)

def _compute_bound_gradient(a0, a4, a8):
    """Compute the gradient of the bound with respect to (a0, a4, a8)."""
    a = np.array([a0, a4, a8])
    norm2 = a @ G_norm @ a
    if norm2 < 1e-30:
        return np.array([0.0, 0.0, 0.0])
    
    var_x = (a @ G_x @ a) / norm2
    var_p = (a @ G_mom @ a) / norm2
    
    if var_x < 0 or var_p < 0:
        return np.array([0.0, 0.0, 0.0])
    
    bound_val = np.sqrt(max(var_x, 1e-30) * max(var_p, 1e-30))
    if bound_val < 1e-30:
        return np.array([0.0, 0.0, 0.0])
    
    # d(bound)/da = (var_p * d(var_x) + var_x * d(var_p)) / (2 * bound_val)
    # var_x = (a' G_x a) / (a' G_norm a)
    # d(var_x)/da = [2 G_x a * norm2 - (a' G_x a) * 2 G_norm a] / norm2^2
    #            = 2 [G_x a - var_x * G_norm a] / norm2
    
    dvar_x = 2.0 * (G_x @ a - var_x * (G_norm @ a)) / norm2
    dvar_p = 2.0 * (G_mom @ a - var_p * (G_norm @ a)) / norm2
    
    grad = (var_p * dvar_x + var_x * dvar_p) / (2.0 * bound_val)
    
    return grad

def _find_rayleigh_stationary():
    """Find stationary points of the Rayleigh quotient sqrt((a'Gx a)(a'Gm a))/(a'Gn a)^2.
    
    At a stationary point, the gradient is zero:
    var_p * (G_x a - var_x * G_norm a) + var_x * (G_mom a - var_p * G_norm a) = 0
    
    This gives: var_p * G_x a + var_x * G_mom a - (var_p*var_x + var_x*var_p) G_norm a = 0
    i.e.: var_p * G_x a + var_x * G_mom a - 2*var_x*var_p * G_norm a = 0
    
    Let's parameterize differently. Let u = sqrt(var_x), v = sqrt(var_p).
    Then bound = u*v, and the stationarity condition involves the ratio.
    
    Alternative: minimize bound^2 = var_x * var_p subject to norm2 = 1.
    Using Lagrange multipliers:
    d/d a [ (a'Gx a)(a'Gm a)/(a'Gn a)^2 - lambda*(a'Gn a - 1) ] = 0
    
    This is a 4th-order eigenvalue problem. Instead, let's use a simpler approach:
    For a given ratio r = var_x/var_p, the optimal a minimizes the product.
    
    Actually, let's just find eigenvectors of the generalized problem.
    The stationary points satisfy:
    (var_p * G_x + var_x * G_mom - 2*var_x*var_p*G_norm) a = 0
    
    where var_x = a'Gx a / a'Gn a, var_p = a'Gm a / a'Gn a.
    
    This is implicit. Let's try a different parametrization.
    Let t = var_x / (var_x + var_p). Then we can search over t.
    
    For a fixed t, we want to minimize (1-t)*var_p + t*var_x... no.
    
    Let me try: the product var_x * var_p is minimized when the two are balanced
    in some sense. Let's use the fact that for a 3D problem, we can do a 
    generalized eigenvalue approach.
    
    Define: find a such that G_x a = mu * G_norm a and G_mom a = nu * G_norm a.
    This is over-determined in 3D. Instead, let's use the following:
    
    For any unit vector u (in G_norm metric), the bound is sqrt(u'Gx u * u'Gm u).
    We can parametrize u on the sphere and minimize.
    """
    # Normalize G_norm to get a metric
    # Find eigendecomposition of G_norm
    eigvals, eigvecs = np.linalg.eigh(G_norm)
    # Filter out near-zero eigenvalues
    mask = eigvals > 1e-15
    eigvals = eigvals[mask]
    eigvecs = eigvecs[:, mask]
    
    if len(eigvals) < 3:
        return []
    
    # Transform to orthonormal coordinates: a = eigvecs @ diag(1/sqrt(eigvals)) @ y
    # Then a' G_norm a = y'y = 1
    inv_sqrt_eig = 1.0 / np.sqrt(eigvals)
    
    # Transformed matrices
    Gx_t = (eigvecs * inv_sqrt_eig) @ G_x @ (eigvecs * inv_sqrt_eig)
    Gm_t = (eigvecs * inv_sqrt_eig) @ G_mom @ (eigvecs * inv_sqrt_eig)
    
    # Now minimize sqrt((y'Gx_t y)(y'Gm_t y)) subject to y'y = 1
    # In 3D, parametrize y on the unit sphere.
    # Stationary points: (Gx_t y * (y'Gm_t y) + Gm_t y * (y'Gx_t y)) / 2 = lambda * y
    # i.e.: (y'Gm_t y) * Gx_t y + (y'Gx_t y) * Gm_t y = 2*lambda*y
    
    # This is still implicit. Let's use a grid search on the sphere + local optimization.
    # Or better: use the fact that in 3D, we can reduce to 2D.
    
    # Alternative approach: use the secular equation.
    # For the product to be stationary, we need:
    # (y'Gm y) Gx y + (y'Gx y) Gm y = 2*lambda * y
    # Take dot product with y:
    # (y'Gm y)(y'Gx y) + (y'Gx y)(y'Gm y) = 2*lambda
    # => lambda = (y'Gx y)(y'Gm y) = bound^2
    
    # So the condition is: (y'Gm y) Gx y + (y'Gx y) Gm y = 2*(y'Gx y)(y'Gm y) y
    # Let alpha = y'Gx y, beta = y'Gm y.
    # beta * Gx y + alpha * Gm y = 2*alpha*beta * y
    
    # This is a nonlinear eigenvalue problem. Let's solve it iteratively.
    # Start from multiple initial points and use fixed-point / Newton.
    
    candidates = []
    rng = np.random.default_rng(123)
    
    # Generate initial points on the unit sphere in 3D
    for _ in range(20):
        y0 = rng.normal(size=3)
        y0 /= np.linalg.norm(y0)
        candidates.append(y0)
        
        # Also try points in the plane
        theta = rng.uniform(0, 2*np.pi)
        y0 = np.array([np.cos(theta), np.sin(theta), 0.0])
        candidates.append(y0)
    
    # For each candidate, do a few iterations of the nonlinear eigenvalue solve
    best_y = None
    best_val = 1e10
    
    for y0 in candidates:
        y = y0.copy()
        for _ in range(50):
            alpha = y @ Gx_t @ y
            beta = y @ Gm_t @ y
            if alpha < 0 or beta < 0:
                break
            # Residual: r = beta*Gx*y + alpha*Gm*y - 2*alpha*beta*y
            r = beta * (Gx_t @ y) + alpha * (Gm_t @ y) - 2.0*alpha*beta * y
            # Project onto tangent space: r_t = r - (y'r)*y
            ytr = y @ r
            r_t = r - ytr * y
            # Newton-like step: we need the Jacobian of the residual
            # For simplicity, use a damped step
            norm_r = np.linalg.norm(r_t)
            if norm_r < 1e-12:
                break
            # Step size
            step = 0.1
            y_new = y - step * r_t / (norm_r + 1e-15)
            y_new /= np.linalg.norm(y_new)
            # Check improvement
            val_old = np.sqrt(max(alpha, 0) * max(beta, 0))
            alpha_new = y_new @ Gx_t @ y_new
            beta_new = y_new @ Gm_t @ y_new
            val_new = np.sqrt(max(alpha_new, 0) * max(beta_new, 0))
            if val_new < val_old:
                y = y_new
            else:
                # Try smaller step
                y_new = y - 0.01 * r_t / (norm_r + 1e-15)
                y_new /= np.linalg.norm(y_new)
                alpha_new = y_new @ Gx_t @ y_new
                beta_new = y_new @ Gm_t @ y_new
                val_new = np.sqrt(max(alpha_new, 0) * max(beta_new, 0))
                if val_new < val_old:
                    y = y_new
                else:
                    break
        
        alpha = y @ Gx_t @ y
        beta = y @ Gm_t @ y
        val = np.sqrt(max(alpha, 0) * max(beta, 0))
        if val < best_val:
            best_val = val
            best_y = y.copy()
    
    if best_y is not None:
        # Transform back: a = eigvecs @ diag(1/sqrt(eigvals)) @ y
        a_opt = (eigvecs * inv_sqrt_eig) @ best_y
        return [a_opt]
    
    return []

def construct():
    start_time = time.time()
    deadline = 190.0  # Wall-clock limit with safety margin (evaluator kills at 210s)
    
    # Initial guess from the seed
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_x = x0.copy()
    best_score = _compute_bound(x0[0], x0[1], x0[2])
    
    def objective(a):
        return _compute_bound(a[0], a[1], a[2])
    
    def gradient(a):
        return _compute_bound_gradient(a[0], a[1], a[2])
    
    # Try Rayleigh stationary point approach
    if time.time() < deadline - 60.0:
        try:
            rayleigh_sols = _find_rayleigh_stationary()
            for sol in rayleigh_sols:
                val = _compute_bound(sol[0], sol[1], sol[2])
                if val < best_score:
                    best_score = val
                    best_x = sol.copy()
        except Exception:
            pass
    
    # Generate multiple starting points for global search
    rng = np.random.default_rng(42)
    
    starts = [x0, best_x]
    # Local perturbations around seed
    for _ in range(8):
        starts.append(x0 + rng.normal(0, 0.05, size=3))
    # Wider exploration
    for _ in range(5):
        starts.append(x0 + rng.normal(0, 0.5, size=3))
    # Random wide starts
    for _ in range(3):
        starts.append(rng.uniform(-3, 3, size=3))
    
    # Multi-start optimization with gradient information
    for i, start in enumerate(starts):
        if time.time() > deadline - 15.0:
            break
        
        try:
            res = minimize(objective, start, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 10.0:
        try:
            res = minimize(objective, best_x, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-16})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # One more polish round if time permits
    if time.time() < deadline - 5.0:
        try:
            res = minimize(objective, best_x, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 10000, 'gtol': 1e-15, 'ftol': 1e-16})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
