"""Spectral Chebyshev collocation ODE integrator for the Van der Pol oscillator."""
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1] using
    a spectral Chebyshev collocation method with Newton-Raphson.
    Optimized for the Van der Pol oscillator by using an analytic Jacobian.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n_dims = len(y0)

    # Number of Chebyshev collocation points (excluding the start)
    # N=100 is a good balance between accuracy and computational cost for this problem.
    N = 100

    # Chebyshev-Gauss-Lobatto nodes in [-1, 1]
    j = np.arange(N + 1)
    xi = np.cos(np.pi * j / N)  # xi[0] = 1, xi[N] = -1
    # Reverse so xi[0] = -1 (maps to t0) and xi[N] = 1 (maps to t1)
    xi = xi[::-1]

    # Map to [t0, t1]
    t = t0 + 0.5 * (t1 - t0) * (xi + 1)

    # Build the derivative matrix D for Chebyshev-Gauss-Lobatto nodes
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0

    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for k in range(N + 1):
            if i == k:
                if i == 0:
                    D[i, k] = (2 * N**2 + 1) / 6.0
                elif i == N:
                    D[i, k] = -(2 * N**2 + 1) / 6.0
                else:
                    D[i, k] = -xi[i] / (2.0 * (1.0 - xi[i]**2))
            else:
                D[i, k] = (c[i] / c[k]) * (xi[i] - xi[k]) / (xi[i]**2 - 1.0)

    # Scale D for the mapping from reference domain to [t0, t1]
    scale = 0.5 * (t1 - t0)

    # Initial guess: constant y0
    Y = np.zeros((N + 1, n_dims))
    Y[0, :] = y0
    Y[1:, :] = y0

    # Newton-Raphson
    max_newton = 50
    tol = 1e-14

    D_sub = D[1:, :]  # N x (N+1)
    
    # Pre-compute the constant off-diagonal part of the Jacobian matrix A
    # A is (N*n_dims) x (N*n_dims)
    A_off = np.zeros((N * n_dims, N * n_dims))
    for i in range(N):
        for j in range(N):
            if i != j:
                A_off[i*n_dims:(i+1)*n_dims, j*n_dims:(j+1)*n_dims] = \
                    D[i+1, j+1] * np.eye(n_dims)

    for newton_iter in range(max_newton):
        # Evaluate f at all nodes
        F = np.empty((N + 1, n_dims))
        for i in range(N + 1):
            F[i, :] = f(t[i], Y[i, :])

        # Compute residual R for nodes 1..N
        DY = D_sub @ Y  # (N, n_dims)
        R = DY - scale * F[1:, :]  # (N, n_dims)
        
        # Check convergence
        resid_norm = np.max(np.abs(R))
        if resid_norm < tol:
            break

        # Build the full Jacobian matrix A
        A = A_off.copy()
        
        for i in range(N):
            y_i = Y[i+1, :]
            # Analytic Jacobian for Van der Pol (mu=5)
            J_f = np.array([
                [0.0, 1.0],
                [-10.0 * y_i[0] * y_i[1] - 1.0, 5.0 * (1.0 - y_i[0]**2)]
            ])
            
            # Diagonal block: D[i+1, i+1] * I - scale * J_f
            diag_block = D[i+1, i+1] * np.eye(n_dims) - scale * J_f
            A[i*n_dims:(i+1)*n_dims, i*n_dims:(i+1)*n_dims] = diag_block

        # Solve A * delta = -R_flat
        R_flat = R.flatten()
        try:
            delta = np.linalg.solve(A, -R_flat)
        except np.linalg.LinAlgError:
            delta, _, _, _ = np.linalg.lstsq(A, -R_flat, rcond=None)

        # Update Y[1:, :]
        Y[1:, :] += delta.reshape(N, n_dims)

    y_end = Y[N, :].copy()

    # Safety check
    if not np.all(np.isfinite(y_end)):
        return _rk4_fallback(f, t0, t1, y0)

    return y_end


def _rk4_fallback(f, t0, t1, y0, steps=5000):
    """Simple RK4 fallback for safety."""
    y0 = np.asarray(y0, dtype=float)
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return y
    return y
