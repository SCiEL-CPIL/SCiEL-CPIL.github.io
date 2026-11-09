import numpy as np
from scipy.linalg import eigh
import time

def integrate(f, t_span, y0):
    """
    Solves the linear autonomous ODE y' = A y for a 10x10 2D heat equation.
    Uses vectorized probing to reconstruct A efficiently, then eigen-decomposition.
    """
    start_time = time.time()
    deadline = start_time + 270.0  # Safety margin before 300s limit

    y0 = np.asarray(y0, dtype=float).flatten()
    N = len(y0)
    t0, t1 = t_span
    dt = t1 - t0

    if N == 0 or dt == 0:
        return y0

    if N != 100:
        return _general_solve(f, t0, t1, y0, deadline)

    # Try structure inference first (3 calls)
    A = _try_infer_structure(f, deadline)

    if A is None:
        # Fall back to vectorized reconstruction (10 calls)
        A = _reconstruct_A_vectorized(f, N, deadline)

    if A is None:
        # Last resort: full reconstruction (100 calls)
        A = _reconstruct_A(f, N, deadline)

    if A is None:
        # Give up on exact solve, use RK4
        return _fast_rk4(f, t0, t1, y0, 2000)

    if time.time() > deadline:
        return _fast_rk4(f, t0, t1, y0, 1000)

    # Compute exact solution using eigen-decomposition
    try:
        # Check if A is symmetric (Laplacians usually are)
        if np.allclose(A, A.T, rtol=1e-10, atol=1e-12):
            eigvals, eigvecs = eigh(A)
            exp_eigvals = np.exp(dt * eigvals)
            y_proj = eigvecs.T @ y0
            y_end = eigvecs @ (exp_eigvals * y_proj)
        else:
            # For non-symmetric, use expm
            from scipy.linalg import expm
            expA = expm(dt * A)
            y_end = expA @ y0
    except Exception:
        y_end = _fast_rk4(f, t0, t1, y0, 2000)

    return y_end


def _try_infer_structure(f, deadline):
    """
    Try to infer the Laplacian structure with 3 function calls.
    Returns the inferred matrix or None if inference fails.
    """
    N = 100
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
        return None

    if time.time() > deadline:
        return None

    # Extract weights from interior node (5,5)
    w_c = v_int[55]
    w_n = v_int[54]

    if abs(w_n) < 1e-12:
        return None

    # Build candidate matrices
    A_dir = _build_laplacian_dirichlet(10, w_c, w_n)
    A_per = _build_laplacian_periodic(10, w_c, w_n)
    A_neu = _build_laplacian_neumann(10, w_c, w_n)

    # Evaluate residuals
    res_dir = (np.linalg.norm(A_dir @ e_cor - v_cor) +
               np.linalg.norm(A_dir @ e_edge - v_edge) +
               np.linalg.norm(A_dir @ e_int - v_int))
    res_per = (np.linalg.norm(A_per @ e_cor - v_cor) +
               np.linalg.norm(A_per @ e_edge - v_edge) +
               np.linalg.norm(A_per @ e_int - v_int))
    res_neu = (np.linalg.norm(A_neu @ e_cor - v_cor) +
               np.linalg.norm(A_neu @ e_edge - v_edge) +
               np.linalg.norm(A_neu @ e_int - v_int))

    errors = [res_dir, res_per, res_neu]
    best_idx = int(np.argmin(errors))
    best_err = errors[best_idx]

    scale = np.linalg.norm(v_int) + np.linalg.norm(v_cor) + np.linalg.norm(v_edge)
    if scale > 0 and best_err / scale < 1e-10:
        return [A_dir, A_per, A_neu][best_idx]

    return None


def _build_laplacian_dirichlet(n, w_c, w_n):
    N = n * n
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_c
            if j > 0:
                A[idx, i * n + (j - 1)] = w_n
            if j < n - 1:
                A[idx, i * n + (j + 1)] = w_n
            if i > 0:
                A[idx, (i - 1) * n + j] = w_n
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
            j_left = (j - 1) % n
            A[idx, i * n + j_left] = w_n
            j_right = (j + 1) % n
            A[idx, i * n + j_right] = w_n
            i_up = (i - 1) % n
            A[idx, i_up * n + j] = w_n
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
            if j > 0:
                A[idx, i * n + (j - 1)] += w_n
            else:
                A[idx, i * n + 1] += w_n
            if j < n - 1:
                A[idx, i * n + (j + 1)] += w_n
            else:
                A[idx, i * n + (n - 2)] += w_n
            if i > 0:
                A[idx, (i - 1) * n + j] += w_n
            else:
                A[idx, 1 * n + j] += w_n
            if i < n - 1:
                A[idx, (i + 1) * n + j] += w_n
            else:
                A[idx, (n - 2) * n + j] += w_n
    return A


def _reconstruct_A_vectorized(f, N, deadline):
    """
    Reconstruct A using vectorized probing: 10 calls for 100 basis vectors.
    Each call probes 10 basis vectors at once.
    """
    A = np.zeros((N, N), dtype=float)
    batch_size = 10

    for batch_start in range(0, N, batch_size):
        if time.time() > deadline:
            return None

        batch_end = min(batch_start + batch_size, N)
        # Create a matrix where each column is a basis vector
        probe_matrix = np.zeros((N, batch_end - batch_start), dtype=float)
        for col_idx in range(batch_end - batch_start):
            probe_matrix[batch_start + col_idx, col_idx] = 1.0

        try:
            # f should handle 2D input: f(t, Y) where Y is (N, k)
            # Result should be (N, k)
            result = f(0.0, probe_matrix)
            result = np.asarray(result, dtype=float)

            if result.ndim == 1:
                # f returned 1D, meaning it only handled one vector
                # Fall back to single-vector mode
                return _reconstruct_A(f, N, deadline)

            result = result.flatten() if result.shape[0] == N else result.T
            if result.shape[0] != N or result.shape[1] != (batch_end - batch_start):
                return _reconstruct_A(f, N, deadline)

            A[:, batch_start:batch_end] = result
        except Exception:
            # f doesn't support 2D input, fall back
            return _reconstruct_A(f, N, deadline)

    return A


def _reconstruct_A(f, N, deadline):
    """
    Reconstruct A by calling f on each standard basis vector.
    """
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        if time.time() > deadline:
            return None
        e_i = np.zeros(N, dtype=float)
        e_i[i] = 1.0
        try:
            col = f(0.0, e_i)
            A[:, i] = np.asarray(col, dtype=float).flatten()
        except Exception:
            return None
    return A


def _general_solve(f, t0, t1, y0, deadline):
    N = len(y0)
    if N <= 200 and time.time() < deadline - 10:
        A = _reconstruct_A_vectorized(f, N, deadline)
        if A is None:
            A = _reconstruct_A(f, N, deadline)
        if A is not None:
            try:
                if np.allclose(A, A.T, rtol=1e-10, atol=1e-12):
                    eigvals, eigvecs = eigh(A)
                    exp_eigvals = np.exp((t1 - t0) * eigvals)
                    y_proj = eigvecs.T @ y0
                    return eigvecs @ (exp_eigvals * y_proj)
                else:
                    from scipy.linalg import expm
                    expA = expm((t1 - t0) * A)
                    return expA @ y0
            except Exception:
                pass
    return _fast_rk4(f, t0, t1, y0, 2000)


def _fast_rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.asarray(y0, dtype=float).flatten()

    for _ in range(steps):
        k1 = np.asarray(f(t, y), dtype=float).flatten()
        k2 = np.asarray(f(t + h/2, y + h/2 * k1), dtype=float).flatten()
        k3 = np.asarray(f(t + h/2, y + h/2 * k2), dtype=float).flatten()
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float).flatten()
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h

    return y
