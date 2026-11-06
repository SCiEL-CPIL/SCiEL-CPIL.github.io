import numpy as np
import time
from scipy.linalg import expm

_START_TIME = time.time()
_TIMEOUT = 250.0

def _get_spectral_solution(y0, dt, N=10):
    """
    Compute e^{A dt} y0 for the 2D heat equation on NxN grid with Dirichlet BCs.
    Assumes dx = 1.
    """
    y = y0.reshape(N, N)
    
    # Eigenvalues of 1D Laplacian: -4 * sin^2(k*pi/(2*(N+1)))
    k = np.arange(1, N + 1)
    lambda_1d = -4.0 * np.sin(np.pi * k / (2 * (N + 1)))**2
    
    # 2D eigenvalues are sums of 1D eigenvalues
    # mu_{j,k} = lambda_j + lambda_k
    # We can use the property that the transform is separable.
    # y_sol = V * diag(exp(lambda_1d * dt)) * V^T * y * V * diag(exp(lambda_1d * dt)) * V^T
    # where V is the DST-I matrix (sine transform).
    
    # DST-I can be computed via FFT or directly.
    # For small N, direct is fine, but let's use a simple sine transform.
    # V[i, k] = sqrt(2/(N+1)) * sin(pi * (i+1) * (k+1) / (N+1))
    
    # Construct V
    i_idx = np.arange(N)[:, None] + 1
    k_idx = np.arange(N)[None, :] + 1
    V = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * i_idx * k_idx / (N + 1))
    
    # Transform y to sine basis
    y_sine = V.T @ y @ V
    
    # Apply exponential decay
    # lambda_1d is a vector of length N
    # We need to multiply each element y_sine[j, k] by exp((lambda_1d[j] + lambda_1d[k]) * dt)
    # This is equivalent to:
    # y_sine * exp(lambda_1d[:, None] * dt) * exp(lambda_1d[None, :] * dt)
    decay_1d = np.exp(lambda_1d * dt)
    y_sine_decayed = y_sine * decay_1d[:, None] * decay_1d[None, :]
    
    # Transform back
    y_sol = V @ y_sine_decayed @ V.T
    
    return y_sol.flatten()

def _verify_spectral_operator(f, t0, y0, N=10, dx=1.0):
    """
    Verify if f(t, y) matches the standard 2D heat equation operator.
    Returns True if match is close, False otherwise.
    Also returns the estimated dx if possible, but we assume dx=1 for now.
    """
    # Predicted action of A on y0
    y = y0.reshape(N, N)
    # Interior points only for the Laplacian action? No, y0 is the full state vector.
    # The matrix A acts on the full vector.
    # A y = (L x I + I x L) y
    # We can compute this using the spectral method with dt=0? No, that's identity.
    # We need A y.
    # A = V * diag(lambda_2d) * V^T
    # So A y = V * diag(lambda_2d) * (V^T y)
    
    k = np.arange(1, N + 1)
    lambda_1d = -4.0 * np.sin(np.pi * k / (2 * (N + 1)))**2 / (dx*dx)
    
    i_idx = np.arange(N)[:, None] + 1
    k_idx = np.arange(N)[None, :] + 1
    V = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * i_idx * k_idx / (N + 1))
    
    y_sine = V.T @ y @ V
    # lambda_2d[j, k] = lambda_1d[j] + lambda_1d[k]
    lambda_2d = lambda_1d[:, None] + lambda_1d[None, :]
    
    y_sine_A = y_sine * lambda_2d
    A_y = (V @ y_sine_A @ V.T).flatten()
    
    # Compare with f(t0, y0)
    f_y = f(t0, y0)
    
    # Check relative error
    norm_Ay = np.linalg.norm(A_y)
    norm_diff = np.linalg.norm(A_y - f_y)
    
    if norm_Ay < 1e-10:
        # If A_y is zero, check if f_y is also zero
        return norm_diff < 1e-8
    else:
        return norm_diff / norm_Ay < 1e-5

def _rk4(f, t0, t1, y0, steps):
    """Standard RK4 solver."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + 4 * k4) / 6 # Wait, RK4 is (k1+2k2+2k3+k4)/6
        # Correction:
        y = y - h / 6 * (k1 + 2 * k2 + 2 * k3 + 4 * k4) # This is wrong logic in the loop above, let's rewrite cleanly
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y

def _rk4_clean(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    N = 10
    
    # Try spectral method
    try:
        if time.time() - _START_TIME < _TIMEOUT:
            # Verify if the operator matches the standard 2D heat equation with dx=1
            if _verify_spectral_operator(f, t0, y0, N, dx=1.0):
                y_sol = _get_spectral_solution(y0, dt, N)
                if np.all(np.isfinite(y_sol)):
                    return y_sol
            else:
                # Try dx=0.1 or other common values? 
                # The problem statement says "fixed 10x10", implying a specific standard setup.
                # Usually dx=1 is the default in such abstract problems unless specified.
                # If dx=1 fails, we might try to estimate dx.
                # Let's try to estimate dx by comparing norms.
                # A_y predicted with dx=1 has norm N1.
                # f_y has norm N2.
                # If A_y = c * A_y_dx1, then dx = 1/sqrt(c).
                
                # Re-compute A_y for dx=1 to get the ratio
                k = np.arange(1, N + 1)
                lambda_1d_1 = -4.0 * np.sin(np.pi * k / (2 * (N + 1)))**2
                i_idx = np.arange(N)[:, None] + 1
                k_idx = np.arange(N)[None, :] + 1
                V = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * i_idx * k_idx / (N + 1))
                y_mat = y0.reshape(N, N)
                y_sine = V.T @ y_mat @ V
                lambda_2d_1 = lambda_1d_1[:, None] + lambda_1d_1[None, :]
                A_y_dx1 = (V @ (y_sine * lambda_2d_1) @ V.T).flatten()
                
                f_y = f(t0, y0)
                
                norm_Ay1 = np.linalg.norm(A_y_dx1)
                norm_fy = np.linalg.norm(f_y)
                
                if norm_Ay1 > 1e-10 and norm_fy > 1e-10:
                    ratio = norm_fy / norm_Ay1
                    # A_y_dx = A_y_dx1 * (1/dx^2)
                    # So 1/dx^2 = ratio => dx = 1/sqrt(ratio)
                    dx_est = 1.0 / np.sqrt(ratio)
                    
                    # Verify with estimated dx
                    if _verify_spectral_operator(f, t0, y0, N, dx=dx_est):
                        y_sol = _get_spectral_solution(y0, dt, N) # Wait, _get_spectral_solution assumes dx=1
                        # We need a version that takes dx
                        # Let's refactor _get_spectral_solution to take dx
                        pass 
                    else:
                        # Fall back to RK4
                        y = _rk4_clean(f, t0, t1, y0, 2000)
                        if y is not None:
                            return y
                        y = _rk4_clean(f, t0, t1, y0, 10000)
                        return y
                else:
                    # Fall back to RK4
                    y = _rk4_clean(f, t0, t1, y0, 2000)
                    if y is not None:
                        return y
                    y = _rk4_clean(f, t0, t1, y0, 10000)
                    return y
    except Exception:
        pass
    
    # Final fallback
    y = _rk4_clean(f, t0, t1, y0, 2000)
    if y is not None:
        return y
    y = _rk4_clean(f, t0, t1, y0, 10000)
    return y

def _get_spectral_solution_dx(y0, dt, N=10, dx=1.0):
    """
    Compute e^{A dt} y0 for the 2D heat equation on NxN grid with Dirichlet BCs.
    Takes dx into account.
    """
    y = y0.reshape(N, N)
    
    k = np.arange(1, N + 1)
    lambda_1d = -4.0 * np.sin(np.pi * k / (2 * (N + 1)))**2 / (dx*dx)
    
    i_idx = np.arange(N)[:, None] + 1
    k_idx = np.arange(N)[None, :] + 1
    V = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * i_idx * k_idx / (N + 1))
    
    y_sine = V.T @ y @ V
    
    decay_1d = np.exp(lambda_1d * dt)
    y_sine_decayed = y_sine * decay_1d[:, None] * decay_1d[None, :]
    
    y_sol = V @ y_sine_decayed @ V.T
    
    return y_sol.flatten()

def _verify_spectral_operator_dx(f, t0, y0, N=10, dx=1.0):
    """
    Verify if f(t, y) matches the standard 2D heat equation operator with given dx.
    """
    y = y0.reshape(N, N)
    
    k = np.arange(1, N + 1)
    lambda_1d = -4.0 * np.sin(np.pi * k / (2 * (N + 1)))**2 / (dx*dx)
    
    i_idx = np.arange(N)[:, None] + 1
    k_idx = np.arange(N)[None, :] + 1
    V = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * i_idx * k_idx / (N + 1))
    
    y_sine = V.T @ y @ V
    lambda_2d = lambda_1d[:, None] + lambda_1d[None, :]
    
    y_sine_A = y_sine * lambda_2d
    A_y = (V @ y_sine_A @ V.T).flatten()
    
    f_y = f(t0, y0)
    
    norm_Ay = np.linalg.norm(A_y)
    norm_diff = np.linalg.norm(A_y - f_y)
    
    if norm_Ay < 1e-10:
        return norm_diff < 1e-8
    else:
        return norm_diff / norm_Ay < 1e-5

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    N = 10
    
    try:
        if time.time() - _START_TIME < _TIMEOUT:
            # First, assume dx=1
            if _verify_spectral_operator_dx(f, t0, y0, N, dx=1.0):
                y_sol = _get_spectral_solution_dx(y0, dt, N, dx=1.0)
                if np.all(np.isfinite(y_sol)):
                    return y_sol
            else:
                # Try to estimate dx
                k = np.arange(1, N + 1)
                lambda_1d_1 = -4.0 * np.sin(np.pi * k / (2 * (N + 1)))**2
                i_idx = np.arange(N)[:, None] + 1
                k_idx = np.arange(N)[None, :] + 1
                V = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * i_idx * k_idx / (N + 1))
                y_mat = y0.reshape(N, N)
                y_sine = V.T @ y_mat @ V
                lambda_2d_1 = lambda_1d_1[:, None] + lambda_1d_1[None, :]
                A_y_dx1 = (V @ (y_sine * lambda_2d_1) @ V.T).flatten()
                
                f_y = f(t0, y0)
                
                norm_Ay1 = np.linalg.norm(A_y_dx1)
                norm_fy = np.linalg.norm(f_y)
                
                if norm_Ay1 > 1e-10 and norm_fy > 1e-10:
                    ratio = norm_fy / norm_Ay1
                    dx_est = 1.0 / np.sqrt(ratio)
                    
                    if _verify_spectral_operator_dx(f, t0, y0, N, dx=dx_est):
                        y_sol = _get_spectral_solution_dx(y0, dt, N, dx=dx_est)
                        if np.all(np.isfinite(y_sol)):
                            return y_sol
    except Exception:
        pass
    
    # Fallback to RK4
    y = _rk4_clean(f, t0, t1, y0, 2000)
    if y is not None:
        return y
    y = _rk4_clean(f, t0, t1, y0, 10000)
    return y
