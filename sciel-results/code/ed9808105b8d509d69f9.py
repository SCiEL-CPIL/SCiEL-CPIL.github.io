"""ODE solver exploiting linearity of the convection-diffusion problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_matrix_A(f, t0, y0):
    """
    Build the system matrix A for the linear ODE y' = A @ y.
    Uses the linearity property: f(t, y) = A @ y.
    We call f with the identity matrix to get A directly.
    """
    n = len(y0)
    # f(t, I) should give A @ I = A if f is linear and time-invariant
    # But f takes (t, y) where y is a vector, not a matrix.
    # So we need to call f with each basis vector to get columns of A.
    # Actually, let's think: f(t, y) = A @ y.
    # If we call f(t, e_j) where e_j is the j-th basis vector, we get A @ e_j = column j of A.
    # That would take n calls.
    
    # Alternative: if f is truly linear, f(t, y1) + f(t, y2) = f(t, y1+y2)
    # And f(t, c*y) = c * f(t, y)
    
    # Let's just call f with each basis vector to get A.
    # But that's n calls. For n=100, that's 100 calls.
    
    # Actually, we can be smarter. Let's check if f is time-invariant first.
    # For a time-invariant linear system, f(t, y) = A @ y for all t.
    
    # Let's build A by calling f with basis vectors.
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t0, e)
    
    return A


def _solve_linear_ode_eig(A, t0, t1, y0):
    """
    Solve y' = A @ y using eigenvalue decomposition.
    y(t) = V @ exp(D*(t-t0)) @ V^{-1} @ y0
    where A = V @ D @ V^{-1}
    """
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    # Eigenvalue decomposition
    try:
        eigvals, V = np.linalg.eig(A)
    except np.linalg.LinAlgError:
        return None
    
    # Check if eigenvalues are all real (or nearly real)
    # For convection-diffusion, eigenvalues should be real and negative
    if np.max(np.abs(np.imag(eigvals))) > 1e-10 * np.max(np.abs(eigvals)):
        # Complex eigenvalues - this is still fine, we can handle it
        pass
    
    dt = t1 - t0
    
    # Compute exp(eigvals * dt)
    exp_eig = np.exp(eigvals * dt)
    
    # y(t1) = V @ diag(exp_eig) @ V^{-1} @ y0
    # = V @ (exp_eig * (V^{-1} @ y0))
    
    try:
        V_inv_y0 = np.linalg.solve(V, y0)
        y_end = V @ (exp_eig * V_inv_y0)
    except np.linalg.LinAlgError:
        return None
    
    return y_end


def _rk4(f, t0, t1, y0, steps):
    """Standard RK4 as fallback."""
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
            return None
    return y


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    Exploits linearity to build A and use eigendecomposition.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    deadline = time.time() + 280  # 280s safety margin
    
    # Try to exploit linearity
    try:
        A = _build_matrix_A(f, t0, y0)
        
        if time.time() > deadline:
            # Fallback to RK4 with few steps
            return _rk4(f, t0, t1, y0, 100)
        
        y_end = _solve_linear_ode_eig(A, t0, t1, y0)
        
        if y_end is not None and np.all(np.isfinite(y_end)):
            return y_end
    except Exception:
        pass
    
    # Fallback to RK4
    y_end = _rk4(f, t0, t1, y0, 2000)
    if y_end is not None:
        return y_end
    
    # Last resort: single step
    h = t1 - t0
    return y0 + h * f(t0, y0)
# EVOLVE-BLOCK-END
