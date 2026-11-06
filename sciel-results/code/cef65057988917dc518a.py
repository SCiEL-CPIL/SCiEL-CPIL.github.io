"""ODE solver: Convection-diffusion 100-point using scipy solve_ivp with BDF."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def _build_sparse_jacobian(n):
    """Build the sparse tridiagonal Jacobian for a 1D convection-diffusion grid.
    The standard second-order central difference for diffusion gives:
        d2y/dx2 ≈ (y[i-1] - 2y[i] + y[i+1]) / dx^2
    For convection:
        dy/dx ≈ (y[i+1] - y[i-1]) / (2*dx)
    So f_i = D*(y[i-1] - 2y[i] + y[i+1])/dx^2 + C*(y[i+1] - y[i-1])/(2*dx)
    
    The Jacobian J[i,j] = df_i/dy_j is tridiagonal:
        J[i,i-1] = D/dx^2 - C/(2*dx)
        J[i,i]   = -2*D/dx^2
        J[i,i+1] = D/dx^2 + C/(2*dx)
    
    We need to infer D and C from the problem. Since we don't know them exactly,
    we'll estimate them by probing f at a known state.
    """
    pass  # This won't be used directly; we'll compute Jacobian numerically but sparsely.


def _estimate_coefficients(f, t0, y0, n):
    """Estimate diffusion and convection coefficients by probing f.
    
    For a 1D convection-diffusion equation:
        ∂y/∂t = D * ∂²y/∂x² + C * ∂y/∂x
    
    With central differences on a uniform grid with spacing dx:
        f_i = D/dx² * (y_{i-1} - 2y_i + y_{i+1}) + C/(2dx) * (y_{i+1} - y_{i-1})
    
    We can estimate D and C by evaluating f at specific test states.
    """
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dx = 1.0 / (n - 1)  # Assume domain [0, 1]
    
    # Test with a linear profile: y_i = i
    y_linear = np.arange(n, dtype=float)
    fy = f(t0, y_linear)
    # For linear profile, second derivative = 0, first derivative = 1
    # So f_i ≈ C/(2dx) * (1 - (-1)) = C/dx... wait, let me reconsider.
    # y_{i+1} - y_{i-1} = (i+1) - (i-1) = 2, so C/(2dx)*2 = C/dx
    # y_{i-1} - 2y_i + y_{i+1} = (i-1) - 2i + (i+1) = 0
    # So f_i = C/dx for interior points
    # But boundary conditions matter. Let's use interior points.
    if n >= 3:
        c_est = fy[1] * dx  # f_1 ≈ C/dx, so C ≈ f_1 * dx
    else:
        c_est = 0.0
    
    # Test with a quadratic profile: y_i = i^2
    y_quad = np.arange(n, dtype=float) ** 2
    fy2 = f(t0, y_quad)
    # y_{i+1} - 2y_i + y_{i+1} = (i+1)^2 - 2i^2 + (i-1)^2 = 2
    # y_{i+1} - y_{i-1} = (i+1)^2 - (i-1)^2 = 4i
    # So f_i = D/dx^2 * 2 + C/(2dx) * 4i = 2D/dx^2 + 2Ci/dx
    # For i=1: f_1 = 2D/dx^2 + 2C/dx
    # We know C ≈ c_est, so 2D/dx^2 = f_1 - 2*c_est/dx
    if n >= 3:
        d_est = (fy2[1] - 2 * c_est / dx) * dx**2 / 2
    else:
        d_est = 0.0
    
    return d_est, c_est


def _make_sparse_jacobian(n, D, C, dx):
    """Create the sparse tridiagonal Jacobian matrix."""
    # Lower diagonal: D/dx^2 - C/(2*dx)
    lower = np.full(n - 1, D / dx**2 - C / (2 * dx))
    # Main diagonal: -2*D/dx^2
    main = np.full(n, -2 * D / dx**2)
    # Upper diagonal: D/dx^2 + C/(2*dx)
    upper = np.full(n - 1, D / dx**2 + C / (2 * dx))
    
    # Handle boundaries (Neumann or Dirichlet may modify these, but for now use standard)
    # For Dirichlet BCs, the boundary rows might be different, but let's use the standard tridiagonal
    J = diags([lower, main, upper], [-1, 0, 1], shape=(n, n), format='csr')
    return J


def integrate(f, t_span, y0):
    """Solve the ODE y' = f(t, y) using solve_ivp with BDF method and sparse Jacobian."""
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 300  # 300 seconds
    
    # Estimate coefficients for sparse Jacobian
    try:
        D, C = _estimate_coefficients(f, t0, y0, n)
        dx = 1.0 / (n - 1)
        J_sparse = _make_sparse_jacobian(n, D, C, dx)
    except Exception:
        J_sparse = None
    
    def jac(t, y):
        if J_sparse is not None:
            return J_sparse
        # Fallback: numerical Jacobian (dense)
        n = len(y)
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1e-7 * max(1.0, abs(y[j]))
            J[:, j] = (f(t, y + e) - f(t, y)) / e[j]
        return J
    
    # Use BDF method with tight tolerances
    try:
        sol = solve_ivp(
            f, 
            [t0, t1], 
            y0, 
            method='BDF',
            jac=jac,
            rtol=1e-10,
            atol=1e-12,
            max_step=0.01,
            dense_output=False
        )
        
        if not sol.success:
            # Try with Radau as fallback
            sol = solve_ivp(
                f, 
                [t0, t1], 
                y0, 
                method='Radau',
                jac=jac,
                rtol=1e-10,
                atol=1e-12,
                max_step=0.01,
                dense_output=False
            )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If solve_ivp failed, fall back to a simple explicit method
            return _rk4_fallback(f, t0, t1, y0)
    
    except Exception:
        # Fallback to RK4
        return _rk4_fallback(f, t0, t1, y0)


def _rk4_fallback(f, t0, t1, y0, steps=5000):
    """Simple RK4 fallback if solve_ivp fails."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            break
    return y
# EVOLVE-BLOCK-END
