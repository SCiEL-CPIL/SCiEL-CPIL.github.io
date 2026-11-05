"""ODE solver: 15-dimensional stiff linear system."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _estimate_jacobian(f, t, y):
    """Estimate the Jacobian J of f(t, y) using central finite differences."""
    n = len(y)
    J = np.empty((n, n))
    # Use a small step size
    h = 1e-7
    for j in range(n):
        y_plus = y.copy()
        y_minus = y.copy()
        step = h * max(1.0, abs(y[j]))
        y_plus[j] += step
        y_minus[j] -= step
        J[:, j] = (f(t, y_plus) - f(t, y_minus)) / (2.0 * step)
    return J


def _matrix_exp_linear(f, t0, t1, y0):
    """
    Solve y' = A y (linear, time-independent) using matrix exponential.
    We estimate A from f, then compute y(t1) = exp((t1-t0)*A) @ y0.
    This requires only O(n) calls to f for Jacobian estimation.
    """
    n = len(y0)
    y = np.array(y0, dtype=float)
    
    # Estimate the system matrix A from f(t, y) = A @ y
    # We need to find A such that f(t, y) = A @ y for all y
    # We can estimate A by evaluating f at basis vectors
    # But since f is linear, f(t, y) = A @ y
    # We can estimate A by: A = f(t, I) where I is identity, but that requires n calls
    # Actually, we can use the Jacobian: J = A (since f is linear in y)
    
    # Use a reference point to estimate A
    # Since f(t, y) = A @ y, the Jacobian is A regardless of y
    # Let's estimate A using the Jacobian at y0
    A = _estimate_jacobian(f, t0, y)
    
    # Compute matrix exponential: y(t1) = exp((t1-t0)*A) @ y0
    dt = t1 - t0
    # Use scipy's expm if available, otherwise use eigenvalue decomposition
    try:
        from scipy.linalg import expm
        expmA = expm(dt * A)
        y_end = expmA @ y
    except ImportError:
        # Fallback: eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(dt * A)
        exp_eigenvalues = np.exp(eigenvalues)
        # y_end = V @ diag(exp(lambda_i)) @ V^{-1} @ y
        V_inv = np.linalg.inv(eigenvectors)
        y_end = eigenvectors @ (exp_eigenvalues * (V_inv @ y))
    
    return y_end


def _backward_euler(f, t0, t1, y0, steps, max_newton=10):
    """Implicit Euler with Newton iteration for stiff problems."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    I = np.eye(n)
    
    for _ in range(steps):
        t_next = t + h
        z = y.copy()
        for _newton in range(max_newton):
            fz = f(t_next, z)
            J = _estimate_jacobian(f, t_next, z)
            # Solve (I - h*J) @ delta = -(z - y - h*fz)
            # Newton step: z_new = z + delta
            A_ne = I - h * J
            b_ne = -(z - y - h * fz)
            try:
                step = np.linalg.solve(A_ne, b_ne)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(A_ne, b_ne, rcond=None)[0]
            z = z + step
            if np.linalg.norm(step) <= 1e-12 * (1.0 + np.linalg.norm(z)):
                break
        y, t = z, t_next
    return y


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Strategy:
    1. Try the linear matrix exponential approach (most efficient for linear systems).
    2. If that fails or is inaccurate, fall back to implicit Euler.
    3. Use adaptive refinement if needed.
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    deadline = time.time() + 290.0  # Safety margin before 300s limit
    
    # First, try the linear matrix exponential approach
    # This is optimal for linear systems: only ~30 function calls for Jacobian
    try:
        y_linear = _matrix_exp_linear(f, t0, t1, y0_arr)
        if np.all(np.isfinite(y_linear)):
            # Verify by checking if f is actually linear: f(t, 2*y) should be 2*f(t, y)
            # Do a quick linearity check
            y_test = 2.0 * y0_arr
            f_test = f(t0, y_test)
            f_orig = f(t0, y0_arr)
            linearity_err = np.linalg.norm(f_test - 2.0 * f_orig) / (1.0 + np.linalg.norm(f_orig))
            
            if linearity_err < 1e-5:
                # System is linear, return the matrix exponential result
                return y_linear
            else:
                # Not strictly linear, fall through to other methods
                pass
    except Exception:
        pass
    
    # If we're getting close to the deadline, return what we have
    if time.time() > deadline - 50.0:
        # Return a basic result
        try:
            return _backward_euler(f, t0, t1, y0_arr, 1000, max_newton=5)
        except Exception:
            return y0_arr
    
    # For non-linear or stiff systems, use implicit Euler with adaptive steps
    # Start with a reasonable number of steps
    for steps in [1000, 5000, 20000, 100000]:
        if time.time() > deadline - 30.0:
            break
        try:
            y_result = _backward_euler(f, t0, t1, y0_arr, steps, max_newton=8)
            if np.all(np.isfinite(y_result)):
                # Check convergence by comparing with a slightly different step count
                if steps > 1000:
                    y_check = _backward_euler(f, t0, t1, y0_arr, steps // 2, max_newton=8)
                    if np.all(np.isfinite(y_check)):
                        rel_diff = np.linalg.norm(y_result - y_check) / (1.0 + np.linalg.norm(y_result))
                        if rel_diff < 1e-10:
                            return y_result
                else:
                    return y_result
        except Exception:
            continue
    
    # Final fallback: use a moderate number of steps
    try:
        return _backward_euler(f, t0, t1, y0_arr, 10000, max_newton=6)
    except Exception:
        return y0_arr

# EVOLVE-BLOCK-END
