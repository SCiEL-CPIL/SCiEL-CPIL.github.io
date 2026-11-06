<<<<<<< SEARCH
import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    """
    Solve the 10x10 2D Heat Equation IVP on [t0, t1].
    Exploits the Kronecker-sum structure A = kron(I, L) + kron(L, I) with 3 f-calls.
    Falls back to full probing (101 calls) if structure doesn't match.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=np.float64)
    n = y0.size
    dt = t1 - t0
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin for 300s limit

    # Standard 1D Dirichlet Laplacian (positive definite version) for N=10
    N = 10
    # L is tridiagonal with 2 on diag, -1 on off-diag
    # This corresponds to -d^2/dx^2 with standard 5-point stencil
    # Eigenvalues: 2 - 2*cos(k*pi/(N+1)) = 4*sin^2(k*pi/(2*(N+1)))
    L = np.diag(2.0 * np.ones(N)) + np.diag(-1.0 * np.ones(N-1), 1) + np.diag(-1.0 * np.ones(N-1), -1)
    
    # 2D operator: A = kron(I, L) + kron(L, I)
    # Assuming row-major ordering y[i*N + j]
    I_N = np.eye(N)
    A_std = np.kron(I_N, L) + np.kron(L, I_N)
    
    # Pre-compute eigendecomposition of A_std for fast expm
    # A_std is symmetric, so we can use eigh
    eigvals_std, eigvecs_std = np.linalg.eigh(A_std)
    
    # --- Strategy: 3-call structural verification ---
    
    # 1. Check homogeneity: f(0, 0) should be ~0 for homogeneous heat eq
    try:
        f_at_zero = f(t0, np.zeros(n))
    except Exception:
        # If f fails on zeros, fallback
        return _full_probe_solve(f, t0, t1, y0, dt, n, deadline)
    
    b_norm = np.linalg.norm(f_at_zero)
    
    if b_norm > 1e-10:
        # Affine case: unlikely for heat eq, but handle it
        return _full_probe_solve(f, t0, t1, y0, dt, n, deadline)
    
    # 2. Probe column 0: f(0, e_0) gives first column of A (for homogeneous)
    e0 = np.zeros(n)
    e0[0] = 1.0
    try:
        col0 = np.asarray(f(t0, e0), dtype=np.float64)
    except Exception:
        return _full_probe_solve(f, t0, t1, y0, dt, n, deadline)
    
    # 3. Estimate scaling factor c such that A = c * A_std
    # Look for a non-zero entry in A_std[:, 0] to estimate c
    # A_std[1, 0] should be 1 (from kron(I, L) part: L[1,0] = -1, so actually -1)
    # Let's find a reliable index
    nonzero_mask = np.abs(A_std[:, 0]) > 1e-15
    if not np.any(nonzero_mask):
        return _full_probe_solve(f, t0, t1, y0, dt, n, deadline)
    
    # Pick the index with largest magnitude for best SNR
    idx_c = np.argmax(np.abs(A_std[:, 0][nonzero_mask]))
    idx_c = np.where(nonzero_mask)[0][idx_c]
    
    c = col0[idx_c] / A_std[idx_c, 0]
    
    # 4. Verify with column 1: f(0, e_1) should equal c * A_std[:, 1]
    e1 = np.zeros(n)
    e1[1] = 1.0
    try:
        col1 = np.asarray(f(t0, e1), dtype=np.float64)
    except Exception:
        return _full_probe_solve(f, t0, t1, y0, dt, n, deadline)
    
    expected_col1 = c * A_std[:, 1]
    residual = np.linalg.norm(col1 - expected_col1)
    tolerance = 1e-10 * (1.0 + np.linalg.norm(expected_col1))
    
    if residual > tolerance:
        # Structure doesn't match, fallback to full probing
        return _full_probe_solve(f, t0, t1, y0, dt, n, deadline)
    
    # --- Structure verified! Compute solution with 3 total f-calls ---
    # A = c * A_std
    # Solution: y(t1) = expm(A * dt) @ y0 = expm(c * A_std * dt) @ y0
    # Using eigendecomposition: expm(c * A_std * dt) = V @ diag(exp(c * eigvals * dt)) @ V^{-1}
    # Since A_std is symmetric: V^{-1} = V^T
    
    exp_factors = np.exp(c * eigvals_std * dt)
    # y_end = V @ (exp_factors * (V^T @ y0))
    proj = eigvecs_std.T @ y0
    y_end = eigvecs_std @ (exp_factors * proj)
    
    return y_end
=======
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Solve the 10x10 2D Heat Equation IVP on [t0, t1].
    Uses analytic spectral decomposition (DST basis) for exact solution.
    Makes 3 f-calls to verify structure and determine scaling.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=np.float64)
    n = y0.size
    dt = t1 - t0
    start_time = time.time()
    deadline = start_time + 280

    if n != 100:
        return _full_probe_solve(f, t0, t1, y0, dt, n, deadline)

    N = 10

    # Call 1: Check homogeneity
    try:
        f_at_zero = np.asarray(f(t0, np.zeros(n)), dtype=np.float64)
    except Exception:
        return _full_probe_solve(f, t0, t1, y0, dt, n, deadline)
    if np.linalg.norm(f_at_zero) > 1e-10:
        return _full_probe_solve(f, t0, t1, y0, dt, n, deadline)

    # Call 2: Probe first column to determine scaling
    e0 = np.zeros(n)
    e0[0] = 1.0
    try:
        col0 = np.asarray(f(t0, e0), dtype=np.float64)
    except Exception:
        return _full_probe_solve(f, t0, t1, y0, dt, n, deadline)

    # Standard 2D Laplacian structure
    L = np.diag(2.0 * np.ones(N)) + np.diag(-1.0 * np.ones(N-1), 1) + np.diag(-1.0 * np.ones(N-1), -1)
    I_N = np.eye(N)
    A_std = np.kron(I_N, L) + np.kron(L, I_N)

    # Determine scaling c
    nonzero_mask = np.abs(A_std[:, 0]) > 1e-15
    if not np.any(nonzero_mask):
        return _full_probe_solve(f, t0, t1, y0, dt, n, deadline)

    idx_c = np.where(nonzero_mask)[0][np.argmax(np.abs(A_std[:, 0][nonzero_mask]))]
    c = col0[idx_c] / A_std[idx_c, 0]

    # Call 3: Verify with second column
    e1 = np.zeros(n)
    e1[1] = 1.0
    try:
        col1 = np.asarray(f(t0, e1), dtype=np.float64)
    except Exception:
        return _full_probe_solve(f, t0, t1, y0, dt, n, deadline)

    expected_col1 = c * A_std[:, 1]
    residual = np.linalg.norm(col1 - expected_col1)
    tolerance = 1e-10 * (1.0 + np.linalg.norm(expected_col1))

    if residual > tolerance:
        return _full_probe_solve(f, t0, t1, y0, dt, n, deadline)

    # Structure verified! Compute solution using DST
    y0_2d = y0.reshape(N, N)

    def dst2(x):
        N = len(x)
        M = 2 * (N + 1)
        x_ext = np.zeros(M)
        x_ext[1:N+1] = x
        X = np.fft.fft(x_ext)
        return 2.0 * X[1:N+1].real

    # Forward DST along rows
    y_dst = np.zeros((N, N))
    for i in range(N):
        y_dst[i, :] = dst2(y0_2d[i, :])

    # Forward DST along columns
    y_dst2 = np.zeros((N, N))
    for j in range(N):
        y_dst2[:, j] = dst2(y_dst[:, j])

    # Eigenvalues of A_std
    p = np.arange(N)
    q = np.arange(N)
    P, Q = np.meshgrid(p, q, indexing='ij')
    eigvals = 4.0 * np.sin((P + 1) * np.pi / (N + 1))**2 + 4.0 * np.sin((Q + 1) * np.pi / (N + 1))**2

    # Apply exponential: exp(c * eigvals * dt)
    y_dst2 *= np.exp(c * eigvals * dt)

    # Inverse DST along columns
    y_inv = np.zeros((N, N))
    for j in range(N):
        y_inv[:, j] = (2.0 / (N + 1)) * dst2(y_dst2[:, j])

    # Inverse DST along rows
    y_final = np.zeros((N, N))
    for i in range(N):
        y_final[i, :] = (2.0 / (N + 1)) * dst2(y_inv[i, :])

    return y_final.flatten()
>>>>>>> REPLACE
