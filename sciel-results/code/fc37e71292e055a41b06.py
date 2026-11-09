import numpy as np
from scipy.linalg import expm, eigh
import time
import warnings

def integrate(f, t_span, y0):
    """
    Solves the linear autonomous ODE y' = A y for a 10x10 2D heat equation.
    1. Attempts to infer the matrix A structure (Dirichlet, Neumann, or Periodic)
       with 3 function calls.
    2. If inference fails, reconstructs the full matrix A with 100 function calls.
    3. Computes the exact solution using matrix exponential or eigen-decomposition.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    y0 = np.asarray(y0, dtype=float).flatten()
    N = len(y0)
    t0, t1 = t_span
    dt = t1 - t0

    if N == 0:
        return y0
    
    if dt == 0:
        return y0

    # If N is not 100, we can't assume 10x10 structure. 
    # Fallback to full reconstruction if N is small enough, else generic RK.
    if N != 100:
        return _general_solve(f, t0, t1, y0, deadline)

    # 1. Infer Structure with 3 calls
    # Indices: 
    # (0,0) -> 0 (Corner)
    # (0,1) -> 1 (Edge)
    # (5,5) -> 55 (Interior)
    
    e_cor = np.zeros(N)
    e_cor[0] = 1.0
    
    e_edge = np.zeros(N)
    e_edge[1] = 1.0
    
    e_int = np.zeros(N)
    e_int[55] = 1.0

    try:
        v_cor = np.asarray(f(0.0, e_cor), dtype=float).flatten()
        v_edge = np.asarray(f(0.0, e_edge), dtype=float).flatten()
        v_int = np.asarray(f(0.0, e_int), dtype=float).flatten()
    except Exception:
        return _general_solve(f, t0, t1, y0, deadline)

    if time.time() > deadline:
        return _fast_rk4(f, t0, t1, y0, 1000)

    # Extract weights from interior node (5,5)
    # Neighbors: 54, 56, 45, 65
    # Center: 55
    w_c = v_int[55]
    w_n = v_int[54] # Assuming symmetry and uniform coefficients
    
    # Check if weights are valid
    if abs(w_n) < 1e-12:
        # Cannot infer, fallback to full reconstruction
        A = _reconstruct_A(f, N, deadline)
        if A is None:
            return _fast_rk4(f, t0, t1, y0, 1000)
    else:
        # Construct candidate matrices
        A_dir = _build_laplacian_dirichlet(10, w_c, w_n)
        A_per = _build_laplacian_periodic(10, w_c, w_n)
        A_neu = _build_laplacian_neumann(10, w_c, w_n)

        # Evaluate which candidate fits the 3 observations best
        # We have v_cor, v_edge, v_int
        # Check residuals
        res_dir = np.linalg.norm(A_dir @ e_cor - v_cor) + np.linalg.norm(A_dir @ e_edge - v_edge) + np.linalg.norm(A_dir @ e_int - v_int)
        res_per = np.linalg.norm(A_per @ e_cor - v_cor) + np.linalg.norm(A_per @ e_edge - v_edge) + np.linalg.norm(A_per @ e_int - v_int)
        res_neu = np.linalg.norm(A_neu @ e_cor - v_cor) + np.linalg.norm(A_neu @ e_edge - v_edge) + np.linalg.norm(A_neu @ e_int - v_int)

        errors = [res_dir, res_per, res_neu]
        best_idx = np.argmin(errors)
        best_err = errors[best_idx]

        # Threshold for accepting inferred structure
        # Scale relative to vector norms to handle magnitude
        scale = np.linalg.norm(v_int) + np.linalg.norm(v_cor) + np.linalg.norm(v_edge)
        
        if scale > 0 and best_err / scale < 1e-12:
            A = [A_dir, A_per, A_neu][best_idx]
            # Success: 3 calls
        else:
            # Fallback to full reconstruction
            A = _reconstruct_A(f, N, deadline)
            if A is None:
                 return _fast_rk4(f, t0, t1, y0, 1000)

    if time.time() > deadline:
        return _fast_rk4(f, t0, t1, y0, 1000)

    # Compute exact solution using eigen-decomposition for symmetric matrices
    try:
        # Check if A is symmetric (Laplacians usually are)
        if np.allclose(A, A.T, rtol=1e-10, atol=1e-12):
            # Use eigen-decomposition for better stability and speed
            eigvals, eigvecs = eigh(A)
            # y_end = V @ exp(dt * eigvals) @ V^T @ y0
            exp_eigvals = np.exp(dt * eigvals)
            y_proj = eigvecs.T @ y0
            y_end = eigvecs @ (exp_eigvals * y_proj)
        else:
            # For non-symmetric, use matrix exponential
            expA = expm(dt * A)
            y_end = expA @ y0
    except Exception:
        # Fallback to high-accuracy RK4 with adaptive step
        y_end = _adaptive_rk4(f, t0, t1, y0, deadline)

    return y_end

def _build_laplacian_dirichlet(n, w_c, w_n):
    N = n * n
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_c
            
            # Left
            if j > 0:
                A[idx, i * n + (j - 1)] = w_n
            # Right
            if j < n - 1:
                A[idx, i * n + (j + 1)] = w_n
            # Up
            if i > 0:
                A[idx, (i - 1) * n + j] = w_n
            # Down
            if i < n - 1:
                A[idx, (i + 1) * n + j] = w_n
    return A

def _build_laplacian_periodic(n, w_c, w_n):
    N = n * n
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_c
            
            # Left (wrap)
            j_left = (j - 1) % n
            A[idx, i * n + j_left] = w_n
            # Right (wrap)
            j_right = (j + 1) % n
            A[idx, i * n + j_right] = w_n
            # Up (wrap)
            i_up = (i - 1) % n
            A[idx, i_up * n + j] = w_n
            # Down (wrap)
            i_down = (i + 1) % n
            A[idx, i_down * n + j] = w_n
    return A

def _build_laplacian_neumann(n, w_c, w_n):
    N = n * n
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_c
            
            # Left
            if j > 0:
                A[idx, i * n + (j - 1)] += w_n
            else:
                # Ghost left is (i, 1)
                A[idx, i * n + 1] += w_n
            
            # Right
            if j < n - 1:
                A[idx, i * n + (j + 1)] += w_n
            else:
                # Ghost right is (i, n-2)
                A[idx, i * n + (n - 2)] += w_n
            
            # Up
            if i > 0:
                A[idx, (i - 1) * n + j] += w_n
            else:
                # Ghost up is (1, j)
                A[idx, 1 * n + j] += w_n
            
            # Down
            if i < n - 1:
                A[idx, (i + 1) * n + j] += w_n
            else:
                # Ghost down is (n-2, j)
                A[idx, (n - 2) * n + j] += w_n
    return A

def _reconstruct_A(f, N, deadline):
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        if time.time() > deadline:
            return None
        e_i = np.zeros(N, dtype=float)
        e_i[i] = 1.0
        col = f(0.0, e_i)
        A[:, i] = np.asarray(col, dtype=float).flatten()
    return A

def _general_solve(f, t0, t1, y0, deadline):
    N = len(y0)
    if N <= 200 and time.time() < deadline - 10:
        A = _reconstruct_A(f, N, deadline)
        if A is not None:
            try:
                if np.allclose(A, A.T, rtol=1e-10, atol=1e-12):
                    eigvals, eigvecs = eigh(A)
                    exp_eigvals = np.exp((t1 - t0) * eigvals)
                    y_proj = eigvecs.T @ y0
                    return eigvecs @ (exp_eigvals * y_proj)
                else:
                    expA = expm((t1 - t0) * A)
                    return expA @ y0
            except Exception:
                pass
    return _adaptive_rk4(f, t0, t1, y0, deadline)

def _fast_rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.asarray(y0, dtype=float).flatten()
    
    for _ in range(steps):
        k1 = np.asarray(f(t, y), dtype=float).flatten()
        k2 = np.asarray(f(t + h/2, y + h/2 * k1), dtype=float).flatten()
        k3 = np.asarray(f(t + h/2, y + h/2 * k2), dtype=float).flatten()
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float).flatten()
        # Correct RK4 formula: k1 + 2k2 + 2k3 + k4
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        
    return y

def _adaptive_rk4(f, t0, t1, y0, deadline):
    """
    Adaptive RK4 with error estimation using step doubling.
    Uses RK4 with h and RK4 with h/2, compares results.
    """
    y = np.asarray(y0, dtype=float).flatten()
    t = t0
    h = (t1 - t0) / 1000  # Initial guess
    min_h = (t1 - t0) / 1e6
    max_h = (t1 - t0) / 10
    
    while t < t1:
        if time.time() > deadline - 5:
            break
        
        # Cap h to not overshoot t1
        h = min(h, t1 - t)
        
        # RK4 with step h
        y_h = _rk4_step(f, t, y, h)
        
        # RK4 with two steps of h/2
        y_half1 = _rk4_step(f, t, y, h/2)
        y_half2 = _rk4_step(f, t + h/2, y_half1, h/2)
        
        # Error estimate
        err = np.linalg.norm(y_half2 - y_h)
        scale = np.linalg.norm(y_half2) + 1e-12
        rel_err = err / scale
        
        # Accept step if error is small enough
        if rel_err < 1e-10:
            y = y_half2
            t += h
            # Try to increase step size
            if rel_err < 1e-12:
                h = min(h * 2, max_h)
        else:
            # Reduce step size
            h = max(h / 2, min_h)
    
    return y

def _rk4_step(f, t, y, h):
    k1 = np.asarray(f(t, y), dtype=float).flatten()
    k2 = np.asarray(f(t + h/2, y + h/2 * k1), dtype=float).flatten()
    k3 = np.asarray(f(t + h/2, y + h/2 * k2), dtype=float).flatten()
    k4 = np.asarray(f(t + h, y + h * k3), dtype=float).flatten()
    return y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
