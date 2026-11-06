"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Solves the 10x10 2D Heat Equation.
    Strategy: 
    1. Assume the standard 5-point Laplacian structure for the 2D heat equation.
    2. Construct the matrix L analytically for a 10x10 grid.
    3. Probe f(t0, y0) once to determine the diffusion coefficient alpha.
    4. Verify that f(t0, y0) matches alpha * L @ y0.
    5. If verified, use the analytical eigendecomposition of the 2D grid Laplacian
       (via 1D sine transforms) to compute the exact solution y(t_end) = V diag(exp(lambda_i * dt)) V^T y0.
       This requires only 1 call to f (the probe) and achieves machine precision.
    6. If not verified, fallback to a robust implicit solver (Radau) with high accuracy.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Standard 10x10 grid
    grid_size = 10
    if n != grid_size * grid_size:
        # Fallback for unexpected sizes, though task specifies 10x10
        return _fallback_solve(f, t_span, y0, start_time)

    # Construct the 5-point Laplacian matrix A for the heat equation:
    # u_t = D * (u_xx + u_yy)
    # Discretized as: u_t = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]) * D / dx^2
    # We build the combinatorial Laplacian L (with -4 on diagonal, 1 on neighbors).
    # The actual ODE is y' = alpha * L @ y.
    
    # Instead of building the full dense matrix L (which is 100x100), we can use
    # the analytical eigendecomposition of the 2D grid Laplacian with Dirichlet BCs.
    # The eigenvalues of the 2D Laplacian on an N x N grid are:
    # lambda_{p,q} = -4 * (sin^2(p*pi/(2*(N+1))) + sin^2(q*pi/(2*(N+1))))
    # for p, q in 1..N
    # The eigenvectors are separable: v_{p,q}(i,j) = sin(p*i*pi/(N+1)) * sin(q*j*pi/(N+1))
    
    # First, probe f to determine alpha.
    # We need L @ y0 to compute the residual. Let's build L or compute L @ y0 directly.
    # Computing L @ y0 directly is O(N^2) and avoids storing the full matrix.
    
    Ly0 = np.zeros(n)
    idx = lambda i, j: i * grid_size + j
    
    # Compute L @ y0
    for i in range(grid_size):
        for j in range(grid_size):
            k = idx(i, j)
            val = -4.0 * y0[k]
            if i > 0:
                val += y0[idx(i-1, j)]
            if i < grid_size - 1:
                val += y0[idx(i+1, j)]
            if j > 0:
                val += y0[idx(i, j-1)]
            if j < grid_size - 1:
                val += y0[idx(i, j+1)]
            Ly0[k] = val
            
    f_probe = np.array(f(t0, y0))
    
    dot_Ly0_Ly0 = np.dot(Ly0, Ly0)
    
    if dot_Ly0_Ly0 < 1e-12:
        # y0 is in the null space of L (e.g., constant vector for Neumann, or zero).
        # For Dirichlet, only zero vector has eigenvalue 0.
        # If y0 is zero, solution is zero.
        if np.allclose(y0, 0):
            return y0
        # If not zero but Ly0 is zero, it's an eigenvector with eigenvalue 0?
        # For Dirichlet, this shouldn't happen unless y0 is zero.
        # If f_probe is also close to zero, then y(t) = y0 (steady state).
        if np.allclose(f_probe, 0, atol=1e-10):
            return y0
        # Otherwise, structure mismatch or different BCs.
        return _fallback_solve(f, t_span, y0, start_time)

    alpha = np.dot(f_probe, Ly0) / dot_Ly0_Ly0
    
    # Verify the fit: || f_probe - alpha * Ly0 || should be small
    residual = np.linalg.norm(f_probe - alpha * Ly0)
    norm_f = np.linalg.norm(f_probe)
    
    # If the residual is small relative to the norm, we assume this is the structure.
    if residual < 1e-8 * (norm_f + 1e-10):
        # We have identified the system as y' = A y with A = alpha * L.
        # Use the analytical eigendecomposition of the 2D grid Laplacian.
        
        N = grid_size
        dt = t1 - t0
        
        # Compute eigenvalues
        # lambda_{p,q} = -4 * (sin^2(p*pi/(2*(N+1))) + sin^2(q*pi/(2*(N+1))))
        p = np.arange(1, N+1)
        q = np.arange(1, N+1)
        
        # sin(p*pi/(N+1)) for p in 1..N
        s_p = np.sin(p * np.pi / (N + 1))
        s_q = np.sin(q * np.pi / (N + 1))
        
        # Eigenvalues of the combinatorial Laplacian L
        # lambda_{p,q} = -(s_p^2 * 2 + s_q^2 * 2) ? No.
        # The 1D Laplacian eigenvalues are -2*(1 - cos(k*pi/(N+1))) = -4*sin^2(k*pi/(2*(N+1)))
        # So 2D Laplacian eigenvalues are lambda_x + lambda_y = -4*sin^2(p*pi/(2*(N+1))) - 4*sin^2(q*pi/(2*(N+1)))
        
        lambda_x = -4.0 * s_p**2
        lambda_y = -4.0 * s_q**2
        
        # Outer sum to get 2D eigenvalues
        lambda_2d = lambda_x[:, None] + lambda_y[None, :]
        lambda_2d = lambda_2d.flatten()
        
        # The actual A = alpha * L, so eigenvalues of A are alpha * lambda_2d
        eig_A = alpha * lambda_2d
        
        # Now we need to project y0 onto the eigenvectors and back.
        # The eigenvectors are V_{p,q}(i,j) = sin(p*i*pi/(N+1)) * sin(q*j*pi/(N+1))
        # Note: i, j are 1-indexed in the formula, so i = 1..N, j = 1..N.
        # In 0-indexed: i = 0..N-1, so sin((i+1)*p*pi/(N+1))
        
        # We can use the Discrete Sine Transform (DST) to perform the projection efficiently.
        # The eigenvectors form an orthogonal basis (after normalization).
        # The projection coefficients c_{p,q} are given by the DST of y0.
        
        # Let's construct the eigenvector matrix V explicitly for simplicity (N=10 is small).
        # V is N*N x N*N, but we can use the separability.
        
        # 1D eigenvector matrix S (N x N) where S[i, p] = sin((i+1)*(p+1)*pi/(N+1))
        # Note: p is 0-indexed here, so p+1 is the mode number 1..N.
        i_idx = np.arange(1, N+1)
        p_idx = np.arange(1, N+1)
        
        S = np.sin(np.outer(i_idx, p_idx) * np.pi / (N + 1))
        
        # The 2D eigenvector matrix is the Kronecker product S ⊗ S (or similar, depending on ordering).
        # Our indexing is idx(i, j) = i * N + j.
        # So the 2D eigenvector for mode (p, q) is v_{p,q} = S[:, p-1] ⊗ S[:, q-1] ?
        # Let's check: v_{p,q}(i, j) = S[i, p-1] * S[j, q-1]
        # In the flattened vector, the entry at idx(i, j) is S[i, p-1] * S[j, q-1].
        # This is exactly the Kronecker product S ⊗ S if we order the basis correctly.
        
        # Let's form V = S ⊗ S. The columns of V are the eigenvectors.
        # The order of columns in S ⊗ S corresponds to (i, j) where i is the slow index and j is the fast index.
        # Our idx(i, j) = i * N + j, which matches this ordering.
        
        V = np.kron(S, S)
        
        # Normalize the eigenvectors. The columns of V are orthogonal but not normalized.
        # The norm of each column is the same due to symmetry.
        # norm^2 = sum_{i,j} sin^2((i+1)(p+1)pi/(N+1)) * sin^2((j+1)(q+1)pi/(N+1))
        # = (sum_i sin^2(...)) * (sum_j sin^2(...))
        # For a fixed mode p, sum_{i=1}^N sin^2(i*p*pi/(N+1)) = (N+1)/2
        # So norm^2 = ((N+1)/2) * ((N+1)/2) = (N+1)^2 / 4
        # norm = (N+1) / 2
        
        norm_V = (N + 1) / 2.0
        V_norm = V / norm_V
        
        # Now V_norm is an orthogonal matrix (V_norm^T @ V_norm = I).
        # c = V_norm^T @ y0
        # y_end = V_norm @ (exp(eig_A * dt) * c)
        
        c = V_norm.T @ y0
        scaled_c = c * np.exp(eig_A * dt)
        y_end = V_norm @ scaled_c
        
        # Sanity check: is the result finite?
        if np.all(np.isfinite(y_end)):
            return y_end
        else:
            return _fallback_solve(f, t_span, y0, start_time)
            
    # Fallback: The structure didn't match perfectly or expm failed.
    # Use a robust implicit solver.
    return _fallback_solve(f, t_span, y0, start_time)

def _fallback_solve(f, t_span, y0, start_time):
    """
    Robust fallback using SciPy's Radau method (implicit, stiff-accurate).
    """
    from scipy.integrate import solve_ivp
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Set a tight tolerance to maximize accuracy score.
    # rtol=1e-10, atol=1e-12
    # Radau is good for stiff problems like heat equation.
    
    # We need to be careful with the number of calls.
    # Radau might use many Jacobian evaluations if not provided.
    # But we don't have the Jacobian analytically here (or we do, but the probe failed).
    # Let's just let it estimate.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=(t1-t0)/10 # Limit step size to ensure accuracy
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: High-order RK4 with many steps
    # This is slow and less accurate for stiff systems, but better than failing.
    steps = 1000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    
    for _ in range(steps):
        if time.time() > start_time + 300:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
# EVOLVE-BLOCK-END
