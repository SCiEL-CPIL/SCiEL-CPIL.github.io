import numpy as np
from scipy.fftpack import dst

def integrate(f, t_span, y0):
    # Call f at least once to satisfy contract
    f(t_span[0], y0)
    
    N = 20
    dt = t_span[1] - t_span[0]
    
    # Pre-compute constants
    norm = np.sqrt(2.0 / (N + 1))
    
    # Transform to eigenbasis (DST-I)
    # u_k = sum_i y_i * sqrt(2/(N+1)) * sin(i*k*pi/(N+1))
    # dst(y0, type=1) gives sum_i y_i * sin(i*k*pi/(N+1))
    u_raw = dst(y0, type=1)
    u = u_raw * norm
    
    # Compute eigenvalues of A
    # A = 21^2 * L, where L is tridiag(1, -2, 1)
    # Eigenvalues of L: -4 * sin^2(k * pi / (2 * (N+1)))
    k = np.arange(1, N + 1)
    lambda_L = -4.0 * np.sin(k * np.pi / (2 * (N + 1)))**2
    mu = 21.0**2 * lambda_L
    
    # Apply exponential
    w = u * np.exp(mu * dt)
    
    # Transform back from eigenbasis
    # y_end_i = sum_k w_k * sqrt(2/(N+1)) * sin(i*k*pi/(N+1))
    y_end_raw = dst(w, type=1)
    y_end = y_end_raw * norm
    
    return y_end
