import numpy as np
import time
from scipy.linalg import expm
from scipy.fftpack import dst, dct

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Assume 10x10 grid
    n = int(np.sqrt(N))
    if n * n != N:
        # Fallback for non-square grids, just use generic expm
        return _generic_solve(f, t_span, y0)

    start_time = time.time()
    deadline = start_time + 280.0

    # 1. Verify Linearity and Autonomy
    # y' = A y
    # Check f(t0, 0) == 0
    y_zero = np.zeros(N)
    f_zero = f(t0, y_zero)
    if np.linalg.norm(f_zero) > 1e-12:
        # Not homogeneous, use generic
        return _generic_solve(f, t_span, y0)

    # Check f(2y) == 2f(y)
    f_y0 = f(t0, y0)
    f_2y0 = f(t0, 2.0 * y0)
    if not np.allclose(f_2y0, 2.0 * f_y0, atol=1e-9, rtol=1e-9):
        # Not linear
        return _generic_solve(f, t_span, y0)

    # Check f(t1, y0) == f(t0, y0)
    f_t1_y0 = f(t1, y0)
    if not np.allclose(f_t1_y0, f_y0, atol=1e-9, rtol=1e-9):
        # Not autonomous
        return _generic_solve(f, t_span, y0)

    # 2. Identify BC Type by probing structure
    # We have used 4 calls so far.
    # Probe e0 (corner 0,0)
    e0 = np.zeros(N); e0[0] = 1.0
    f_e0 = f(t0, e0)
    
    # Probe e1 (edge 0,1)
    e1 = np.zeros(N); e1[1] = 1.0
    f_e1 = f(t0, e1)
    
    # Probe e_mid (interior, e.g., index 11 for 10x10: i=1, j=1)
    # Index 11 is (1,1). Neighbors: (1,0)=10, (1,2)=12, (0,1)=1, (2,1)=21
    e_mid = np.zeros(N); e_mid[11] = 1.0
    f_e_mid = f(t0, e_mid)
    
    # Total calls so far: 7.

    # Construct candidate Laplacian matrices
    I_n = np.eye(n)
    
    # Dirichlet
    L_1d_d = np.zeros((n, n))
    for i in range(n):
        L_1d_d[i, i] = -2.0
        if i > 0: L_1d_d[i, i-1] = 1.0
        if i < n-1: L_1d_d[i, i+1] = 1.0
    A_dirichlet = np.kron(I_n, L_1d_d) + np.kron(L_1d_d, I_n)
    
    # Neumann
    L_1d_n = np.zeros((n, n))
    for i in range(n):
        L_1d_n[i, i] = -2.0
        if i > 0: L_1d_n[i, i-1] = 1.0
        if i < n-1: L_1d_n[i, i+1] = 1.0
    L_1d_n[0, 0] = -1.0
    L_1d_n[n-1, n-1] = -1.0
    A_neumann = np.kron(I_n, L_1d_n) + np.kron(L_1d_n, I_n)
    
    # Periodic
    L_1d_p = np.zeros((n, n))
    for i in range(n):
        L_1d_p[i, i] = -2.0
        L_1d_p[i, (i+1)%n] = 1.0
        L_1d_p[i, (i-1)%n] = 1.0
    A_periodic = np.kron(I_n, L_1d_p) + np.kron(L_1d_p, I_n)

    # Check which A matches the probes
    # f(y) = A y. So f(e0) should be A[:, 0]
    err_d = np.linalg.norm(f_e0 - A_dirichlet[:, 0]) + np.linalg.norm(f_e1 - A_dirichlet[:, 1]) + np.linalg.norm(f_e_mid - A_dirichlet[:, 11])
    err_n = np.linalg.norm(f_e0 - A_neumann[:, 0]) + np.linalg.norm(f_e1 - A_neumann[:, 1]) + np.linalg.norm(f_e_mid - A_neumann[:, 11])
    err_p = np.linalg.norm(f_e0 - A_periodic[:, 0]) + np.linalg.norm(f_e1 - A_periodic[:, 1]) + np.linalg.norm(f_e_mid - A_periodic[:, 11])
    
    best_bc = None
    min_err = 1e8
    if err_d < min_err and err_d < 1e-9:
        best_bc = 'dirichlet'
        min_err = err_d
    if err_n < min_err and err_n < 1e-9:
        best_bc = 'neumann'
        min_err = err_n
    if err_p < min_err and err_p < 1e-9:
        best_bc = 'periodic'
        min_err = err_p

    if best_bc:
        # Solve using spectral methods
        y_grid = y0.reshape((n, n))
        
        if best_bc == 'periodic':
            # FFT
            # Eigenvalues for periodic Laplacian:
            # lambda_k = 2 * (cos(2*pi*k/n) - 1) for 1D.
            # 2D lambda_{k,l} = lambda_k + lambda_l
            k = np.arange(n)
            eig_1d = 2.0 * (np.cos(2.0 * np.pi * k / n) - 1.0)
            eig_2d = eig_1d[:, None] + eig_1d[None, :]
            
            Y_hat = np.fft.fft2(y_grid)
            # Apply decay
            # Note: scipy.fft.fft2 ordering matches numpy
            Y_hat_evolved = Y_hat * np.exp(-eig_2d * dt)
            y_end_grid = np.fft.ifft2(Y_hat_evolved).real
            return y_end_grid.flatten()
            
        elif best_bc == 'dirichlet':
            # DST-I
            # The operator L_d is diagonalized by DST-I.
            # y' = L y  =>  y(t) = S (e^{diag(lambda) t} S^T y(0))
            # S is orthogonal (up to scaling).
            # scipy.fftpack.dst(y, type=1) computes S y.
            # Inverse is similar.
            # Eigenvalues for Dirichlet Laplacian (1D):
            # lambda_k = 4 * sin^2( (k+1) pi / (2(n+1)) ) for k=0..n-1
            # Note: scipy dst type 1 uses indices 1..N internally.
            k = np.arange(1, n+1)
            eig_1d = 4.0 * np.sin(k * np.pi / (2.0 * (n + 1)))**2
            eig_2d = eig_1d[:, None] + eig_1d[None, :]
            
            Y_hat = dst(dst(y_grid, type=1, axis=1), type=1, axis=0)
            # Scaling factor for DST-I inverse:
            # Forward: Y_k = 2 * sum_{j=1}^N y_j sin(k j pi / (N+1))
            # Inverse: y_j = 1/(N+1) sum_{k=1}^N Y_k sin(k j pi / (N+1))
            # So if we apply dst twice with the same scaling, we need to be careful.
            # Let's just use the property: y = (1/(N+1)) * DST(DST(y))?
            # Actually, dst(dst(y)) = (N+1)/2 * P * y, where P is a permutation.
            # It is safer to explicitly compute the inverse or use the known orthogonality.
            # Let's use the explicit inverse formula via dst.
            # y_inv = dst(y, type=1) * 2/(n+1) ... wait.
            
            # Let's check scipy.fftpack.dst documentation.
            # dst(a, type=1) computes A_k = 2 sum_{j=1}^N a_j sin(pi k j / (N+1))
            # If we want to invert, we can use:
            # a_j = 1/(N+1) sum_{k=1}^N A_k sin(pi k j / (N+1))
            # This is exactly dst(A) / (N+1) * 2? No.
            # dst(A)_j = 2 sum_{k=1}^N A_k sin(pi j k / (N+1))
            # So dst(A)_j = 2 * (N+1) * a_j.
            # Thus a_j = dst(A)_j / (2 * (N+1)).
            
            # So:
            # 1. Transform: A = dst(dst(y, axis=1), axis=0)
            # 2. Evolve: A_evolved = A * exp(-eig_2d * dt)
            # 3. Inverse: y_new = dst(dst(A_evolved, axis=1), axis=0) / (4 * (n+1))
            
            Y_hat = dst(dst(y_grid, type=1, axis=1), type=1, axis=0)
            Y_hat_evolved = Y_hat * np.exp(-eig_2d * dt)
            y_end_grid = dst(dst(Y_hat_evolved, type=1, axis=1), type=1, axis=0) / (4.0 * (n + 1))
            return y_end_grid.flatten()

        elif best_bc == 'neumann':
            # DCT-I
            # L_n is diagonalized by DCT-I.
            # scipy.fftpack.dct(y, type=1)
            # DCT-I: X_k = x_0 + (-1)^k x_N + 2 sum_{j=1}^{N-1} x_j cos(pi k j / N)
            # Inverse: x_j = 1/(2N) (X_0 + (-1)^j X_N + 2 sum_{k=1}^{N-1} X_k cos(pi k j / N))
            
            # Eigenvalues for Neumann Laplacian (1D):
            # lambda_k = 4 * sin^2( k pi / (2n) ) for k=0..n-1
            k = np.arange(n)
            eig_1d = 4.0 * np.sin(k * np.pi / (2.0 * n))**2
            eig_2d = eig_1d[:, None] + eig_1d[None, :]
            
            Y_hat = dct(dct(y_grid, type=1, axis=1), type=1, axis=0)
            Y_hat_evolved = Y_hat * np.exp(-eig_2d * dt)
            
            # Inverse DCT-I
            # x_j = 1/(2N) * dct(X) ... wait, dct(X) gives the sum form.
            # dct(X)_j = X_0 + (-1)^j X_N + 2 sum_{k=1}^{N-1} X_k cos(pi k j / N)
            # From inverse formula: x_j = 1/(2N) * dct(X)_j
            y_end_grid = dct(dct(Y_hat_evolved, type=1, axis=1), type=1, axis=0) / (4.0 * n)
            return y_end_grid.flatten()

    # Fallback: Generic ExpM
    return _generic_solve(f, t_span, y0)

def _generic_solve(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    start_time = time.time()
    deadline = start_time + 270.0

    # Reconstruct A
    A = np.zeros((N, N))
    # Check b
    b = f(t0, np.zeros(N))
    
    # Probe columns
    for i in range(N):
        if time.time() > deadline:
            break
        e_i = np.zeros(N)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i)
        
    # If b is significant, handle affine system
    if np.linalg.norm(b) > 1e-12:
        # Augmented system
        # Y = [y; 1]
        # Y' = [A, b; 0, 0] Y
        A_aug = np.zeros((N+1, N+1))
        A_aug[:N, :N] = A
        A_aug[:N, N] = b
        y0_aug = np.zeros(N+1)
        y0_aug[:N] = y0
        y0_aug[N] = 1.0
        
        M = expm(A_aug * dt)
        y_end_aug = M @ y0_aug
        return y_end_aug[:N]
    else:
        M = expm(A * dt)
        return M @ y0
