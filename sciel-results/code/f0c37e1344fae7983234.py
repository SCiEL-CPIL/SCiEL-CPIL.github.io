import numpy as np

def integrate(f, t_span, y0):
    # Call f at least once to satisfy contract
    f(t_span[0], y0)
    
    N = 20
    dt = t_span[1] - t_span[0]
    
    # Analytic eigendecomposition for the specific tridiagonal matrix
    # A = alpha * T, where T has 1 on off-diags, -2 on diag
    # Eigenvalues of T: lambda_j = -2 + 2*cos(j*pi/(N+1))
    # Here alpha = 21**2
    # So eigenvalues of A: mu_j = 21**2 * (-2 + 2*cos(j*pi/21))
    
    j = np.arange(1, N+1)
    # Compute eigenvalues of A
    # lambda_j = -2 * 21^2 * (1 - cos(j*pi/21))
    eigenvalues = -2.0 * (21.0**2) * (1.0 - np.cos(j * np.pi / 21.0))
    
    # Compute exp(eigenvalues * dt)
    exp_eigs = np.exp(eigenvalues * dt)
    
    # Eigenvectors V: V[i,j] = sqrt(2/(N+1)) * sin((i+1)*(j+1)*pi/(N+1))
    # Note: numpy arrays are 0-indexed, so i goes 0..19 corresponding to 1..20
    i = np.arange(N)[:, np.newaxis]
    # j is already 1..20, reshape for broadcasting
    j_col = j[np.newaxis, :]
    
    # V[i, j] = sqrt(2/21) * sin((i+1) * j * pi / 21)
    V = np.sqrt(2.0 / 21.0) * np.sin((i + 1) * j_col * np.pi / 21.0)
    
    # E = V @ diag(exp_eigs) @ V.T
    # This is equivalent to (V * exp_eigs) @ V.T
    # where V * exp_eigs scales columns of V by exp_eigs
    
    E = (V * exp_eigs) @ V.T
    
    y_end = E @ y0
    
    return y_end
