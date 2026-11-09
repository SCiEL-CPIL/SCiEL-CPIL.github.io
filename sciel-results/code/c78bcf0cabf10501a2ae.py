import numpy as np
from scipy.linalg import expm, eigh
import time

def integrate(f, t_span, y0):
    """
    Solves the linear autonomous ODE y' = A y for a 10x10 2D heat equation.
    Reconstructs A via structural inference with minimal function calls,
    then applies the matrix exponential for a near-exact terminal state.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill

    y0 = np.asarray(y0, dtype=float).flatten()
    N = len(y0)
    t0, t1 = t_span
    dt = t1 - t0

    if N == 0 or dt == 0:
        return y0

    if N != 100:
        return _general_solve(f, t0, t1, y0, deadline)

    n = 10

    # --- Probe three representative nodes: corner, edge, interior ---
    idx_corner = 0          # (0,0)
    idx_edge = 5            # (0,5)
    idx_interior = 55       # (5,5)

    probes = {}
    for name, idx in (("corner", idx_corner), ("edge", idx_edge), ("interior", idx_interior)):
        e = np.zeros(N)
        e[idx] = 1.0
        try:
            probes[name] = np.asarray(f(t0, e), dtype=float).flatten()
        except Exception:
            return _fallback_rk4(f, t0, t1, y0)

    if time.time() > deadline:
        return _fallback_rk4(f, t0, t1, y0)

    col_corner = probes["corner"]
    col_edge = probes["edge"]
    col_interior = probes["interior"]

    # --- Infer weights from the probes ---
    w_diag = col_corner[idx_corner]

    # Corner (0,0): right neighbour (0,1) idx 1, down neighbour (1,0) idx 10
    w_right_corner = col_corner[1]
    w_down_corner = col_corner[10]
    w_corner = 0.5 * (w_right_corner + w_down_corner)

    # Edge (0,5): left (0,4) idx 4, right (0,6) idx 6, down (1,5) idx 15
    w_left_edge = col_edge[4]
    w_right_edge = col_edge[6]
    w_down_edge = col_edge[15]
    w_edge = (w_left_edge + w_right_edge + w_down_edge) / 3.0

    # Interior (5,5): up 45, down 65, left 54, right 56
    w_up = col_interior[45]
    w_down = col_interior[65]
    w_left = col_interior[54]
    w_right = col_interior[56]
    w_interior = 0.25 * (w_up + w_down + w_left + w_right)

    # --- Build parameterized candidate matrices ---
    # Each candidate is a function of (w_diag, w_corner, w_edge, w_interior)
    # so the actual inferred values are used, not hard-coded assumptions.
    candidates = [
        _build_dirichlet(n, w_diag, w_interior, w_interior, w_interior),
        _build_periodic(n, w_diag, w_interior),
        _build_neumann(n, w_diag, w_interior, w_corner, w_edge),
        # Anisotropic / generic: use inferred per-region weights directly
        _build_generic(n, w_diag, w_corner, w_edge, w_interior),
    ]

    # --- Verify against all three probes ---
    e_corner = np.zeros(N); e_corner[idx_corner] = 1.0
    e_edge = np.zeros(N); e_edge[idx_edge] = 1.0
    e_interior = np.zeros(N); e_interior[idx_interior] = 1.0

    probe_vecs = [e_corner, e_edge, e_interior]
    probe_cols = [col_corner, col_edge, col_interior]
    scale = sum(np.linalg.norm(c) for c in probe_cols)

    best_A = None
    best_res = float('inf')
    for A_cand in candidates:
        res = 0.0
        for ev, pc in zip(probe_vecs, probe_cols):
            res += np.linalg.norm(A_cand @ ev - pc)
        if res < best_res:
            best_res = res
            best_A = A_cand

    A = None
    if scale > 0 and best_res / scale < 1e-8:
        A = best_A
    else:
        # Structure inference failed -> full reconstruction (100 calls, still cheap)
        A = _full_reconstruct(f, t0, N, deadline)
        if A is None:
            return _fallback_rk4(f, t0, t1, y0)

    if time.time() > deadline:
        return _fallback_rk4(f, t0, t1, y0)

    # --- Apply the exact matrix exponential ---
    try:
        if np.allclose(A, A.T, rtol=1e-9, atol=1e-12):
            eigvals, eigvecs = eigh(A)
            exp_eigvals = np.exp(dt * eigvals)
            y_proj = eigvecs.T @ y0
            y_end = eigvecs @ (exp_eigvals * y_proj)
        else:
            y_end = expm(dt * A) @ y0
    except Exception:
        return _fallback_rk4(f, t0, t1, y0)

    return y_end


def _build_dirichlet(n, w_diag, w_int, w_corner, w_edge):
    """Dirichlet: no wrap, boundary simply has fewer neighbours (weights uniform)."""
    N = n * n
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_diag
            if j > 0:
                A[idx, i * n + (j - 1)] = w_int
            if j < n - 1:
                A[idx, i * n + (j + 1)] = w_int
            if i > 0:
                A[idx, (i - 1) * n + j] = w_int
            if i < n - 1:
                A[idx, (i + 1) * n + j] = w_int
    return A


def _build_periodic(n, w_diag, w_int):
    """Periodic: wrap-around neighbours, uniform weight."""
    N = n * n
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_diag
            A[idx, i * n + ((j - 1) % n)] = w_int
            A[idx, i * n + ((j + 1) % n)] = w_int
            A[idx, ((i - 1) % n) * n + j] = w_int
            A[idx, ((i + 1) % n) * n + j] = w_int
    return A


def _build_neumann(n, w_diag, w_int, w_corner, w_edge):
    """
    Neumann via ghost nodes u[-1]=u[1]:
    interior neighbours weight w_int.
    edge (single boundary axis) neighbour weight w_edge.
    corner (two boundary axes) neighbour weight w_corner.
    """
    N = n * n
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_diag
            # Right
            if j < n - 1:
                A[idx, i * n + (j + 1)] = w_int
            else:
                # boundary: ghost doubles the weight to the single interior neighbour
                A[idx, i * n + (n - 2)] = w_edge
            # Left
            if j > 0:
                A[idx, i * n + (j - 1)] = w_int
            else:
                A[idx, i * n + 1] = w_edge
            # Down
            if i < n - 1:
                A[idx, (i + 1) * n + j] = w_int
            else:
                A[idx, (n - 2) * n + j] = w_edge
            # Up
            if i > 0:
                A[idx, (i - 1) * n + j] = w_int
            else:
                A[idx, 1 * n + j] = w_edge
    return A


def _build_generic(n, w_diag, w_corner, w_edge, w_int):
    """
    Generic: assign each off-diagonal by the region of the *source* node.
    corner node -> w_corner, edge node -> w_edge, interior -> w_int.
    This is the most flexible: it matches the inferred per-region weights exactly.
    """
    N = n * n
    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            A[idx, idx] = w_diag
            # Determine region of source node
            boundary_axes = (0 if (i == 0 or i == n - 1) else 1) + \
                            (0 if (j == 0 or j == n - 1) else 1)
            if boundary_axes == 2:
                w = w_corner
            elif boundary_axes == 1:
                w = w_edge
            else:
                w = w_int
            # Neighbours (no wrap)
            if j > 0:
                A[idx, i * n + (j - 1)] = w
            if j < n - 1:
                A[idx, i * n + (j + 1)] = w
            if i > 0:
                A[idx, (i - 1) * n + j] = w
            if i < n - 1:
                A[idx, (i + 1) * n + j] = w
    return A


def _full_reconstruct(f, t0, N, deadline):
    """Reconstruct A by probing each standard basis vector (N calls)."""
    A = np.zeros((N, N))
    for i in range(N):
        if time.time() > deadline - 5:
            return None
        e_i = np.zeros(N)
        e_i[i] = 1.0
        try:
            A[:, i] = np.asarray(f(t0, e_i), dtype=float).flatten()
        except Exception:
            return None
    return A


def _fallback_rk4(f, t0, t1, y0, steps=2000):
    h = (t1 - t0) / steps
    t, y = t0, np.asarray(y0, dtype=float).flatten()
    for _ in range(steps):
        k1 = np.asarray(f(t, y), dtype=float).flatten()
        k2 = np.asarray(f(t + h / 2, y + h / 2 * k1), dtype=float).flatten()
        k3 = np.asarray(f(t + h / 2, y + h / 2 * k2), dtype=float).flatten()
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float).flatten()
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return y


def _general_solve(f, t0, t1, y0, deadline):
    """Fallback for non-100 sizes: full reconstruction then matrix exponential."""
    N = len(y0)
    A = _full_reconstruct(f, t0, N, deadline)
    if A is None:
        return _fallback_rk4(f, t0, t1, y0)
    try:
        if np.allclose(A, A.T, rtol=1e-9, atol=1e-12):
            eigvals, eigvecs = eigh(A)
            exp_eigvals = np.exp((t1 - t0) * eigvals)
            y_proj = eigvecs.T @ y0
            return eigvecs @ (exp_eigvals * y_proj)
        else:
            return expm((t1 - t0) * A) @ y0
    except Exception:
        return _fallback_rk4(f, t0, t1, y0)
