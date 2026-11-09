"""Spectral Chebyshev collocation ODE integrator for the Van der Pol oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1] using
    a spectral Chebyshev collocation method with Newton-Raphson.
    
    Parameters:
    -----------
    f : callable
        f(t, y) -> y' (array)
    t_span : tuple
        (t_start, t_end)
    y0 : array-like
        Initial condition y(t_start)
        
    Returns:
    --------
    y_end : ndarray
        State at t_span[1]
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n_dims = len(y0)
    
    # Number of Chebyshev nodes (collocation points)
    # We use N+1 nodes for N+1 unknowns (including the initial condition)
    # Let's use N = 100 collocation points (excluding the start), so 101 total nodes
    N = 100
    
    # Chebyshev nodes in [-1, 1]: x_j = cos(pi * j / N) for j = 0, ..., N
    # Map to [t0, t1]
    # We'll work in the reference domain [-1, 1] and map back
    
    # Chebyshev-Gauss-Lobatto nodes in [-1, 1]
    j = np.arange(N + 1)
    xi = np.cos(np.pi * j / N)  # in [-1, 1], xi[0] = 1, xi[N] = -1
    # Reverse so that xi[0] = -1 (corresponds to t0) and xi[N] = 1 (corresponds to t1)
    xi = xi[::-1]
    
    # Map to [t0, t1]
    t = t0 + 0.5 * (t1 - t0) * (xi + 1)
    
    # Derivative matrix D for Chebyshev collocation
    # D is (N+1) x (N+1)
    # For Chebyshev-Gauss-Lobatto nodes, the derivative matrix entries are:
    # D_ij = (c_i / c_j) * (x_i - x_j) / (x_i^2 - 1) for i != j
    # D_ii = -x_i / (2 * (1 - x_i^2)) for i != 0, N
    # D_00 = (2N^2 + 1) / 6
    # D_NN = -(2N^2 + 1) / 6
    # where c_i = 2 for i != 0, N and c_i = 1 for i = 0, N
    
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
    
    # The ODE in the reference domain:
    # dy/dt = f(t, y)
    # dy/dxi = (t1 - t0)/2 * f(t(xi), y(xi))
    # So: D @ Y = (t1 - t0)/2 * F(Y)
    # where Y is (N+1) x n_dims matrix of states at all nodes
    # and F(Y) is (N+1) x n_dims matrix of f(t, y) at all nodes
    
    # Unknowns: Y[1:, :] (N * n_dims unknowns), with Y[0, :] = y0 fixed
    # Residual: R = D @ Y - (t1 - t0)/2 * F(Y) = 0
    # We only enforce the residual at nodes 1, ..., N (N equations)
    
    # Initial guess: linear interpolation or constant
    Y = np.zeros((N + 1, n_dims))
    Y[0, :] = y0
    # Simple initial guess: constant
    Y[1:, :] = y0
    
    # Newton-Raphson iterations
    max_newton = 20
    tol = 1e-13
    
    # Precompute D_sub = D[1:, :] which is N x (N+1)
    D_sub = D[1:, :]  # N x (N+1)
    
    # We need to solve for Y[1:, :] given Y[0, :] = y0
    # Residual at node i (i = 1, ..., N):
    # R_i = sum_k D[i, k] * Y[k] - (t1-t0)/2 * f(t[i], Y[i]) = 0
    
    for newton_iter in range(max_newton):
        # Evaluate f at all nodes
        F = np.zeros((N + 1, n_dims))
        for i in range(N + 1):
            F[i, :] = f(t[i], Y[i, :])
        
        # Compute residual
        R = np.zeros((N, n_dims))
        for i in range(N):
            R[i, :] = D_sub[i, :] @ Y[:, :] - 0.5 * (t1 - t0) * F[i + 1, :]
        
        # Check convergence
        resid_norm = np.max(np.abs(R))
        if resid_norm < tol:
            break
        
        # Compute Jacobian of the residual with respect to Y[1:, :]
        # The residual R[i, :] depends on Y[k, :] for k = 0, ..., N
        # But Y[0, :] is fixed, so we only need dR/dY[j] for j = 1, ..., N
        # 
        # dR[i]/dY[j] = D[i+1, j+1] - 0.5*(t1-t0) * df/dy(t[i+1], Y[i+1]) * delta_{i+1, j+1}
        # 
        # For the diagonal blocks (i = j):
        # dR[i]/dY[i+1] = D[i+1, i+1] - 0.5*(t1-t0) * J_f
        # For off-diagonal (i != j):
        # dR[i]/dY[j+1] = D[i+1, j+1]
        
        # We need to form the block Jacobian of size (N * n_dims) x (N * n_dims)
        # This is expensive for large N and n_dims, but for n_dims = 2 and N = 100,
        # it's 200 x 200, which is fine.
        
        # First, compute the Jacobian of f at each node using finite differences
        J_f = np.zeros((N + 1, n_dims, n_dims))
        for i in range(N + 1):
            for k in range(n_dims):
                e = np.zeros(n_dims)
                e[k] = 1e-8 * max(1.0, abs(Y[i, k]))
                J_f[i, :, k] = (f(t[i], Y[i, :] + e) - f(t[i], Y[i, :] - e)) / (2 * e[k])
        
        # Build the block Jacobian
        # The unknown vector is Y[1:, :] flattened: (N * n_dims)
        # The residual vector is R flattened: (N * n_dims)
        # 
        # Jacobian matrix J_block has blocks J_block[i, j] = dR[i]/dY[j+1] for i, j = 0, ..., N-1
        # J_block[i, j] = D[i+1, j+1] * I - 0.5*(t1-t0) * J_f[i+1] * delta_{ij}
        
        J_block = np.zeros((N * n_dims, N * n_dims))
        for i in range(N):
            for j in range(N):
                if i == j:
                    J_block[i*n_dims:(i+1)*n_dims, j*n_dims:(j+1)*n_dims] = \
                        D[i+1, i+1] * np.eye(n_dims) - 0.5 * (t1 - t0) * J_f[i+1, :, :]
                else:
                    J_block[i*n_dims:(i+1)*n_dims, j*n_dims:(j+1)*n_dims] = \
                        D[i+1, j+1] * np.eye(n_dims)
        
        # Solve J_block * delta_Y = -R_flattened
        R_flat = R.flatten()
        try:
            delta_Y = np.linalg.solve(J_block, -R_flat)
        except np.linalg.LinAlgError:
            # Fallback to least squares
            delta_Y, _, _, _ = np.linalg.lstsq(J_block, -R_flat, rcond=None)
        
        # Update Y
        Y[1:, :] = Y[1:, :] + delta_Y.reshape(N, n_dims)
    
    # Return the state at the end (node N)
    y_end = Y[N, :]
    
    # Ensure finite values
    if not np.all(np.isfinite(y_end)):
        # Fallback: use RK4 as a safety net
        return _rk4_fallback(f, t0, t1, y0)
    
    return y_end


def _rk4_fallback(f, t0, t1, y0, steps=2000):
    """Simple RK4 fallback for safety."""
    y0 = np.asarray(y0, dtype=float)
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return y
    return y
# EVOLVE-BLOCK-END
