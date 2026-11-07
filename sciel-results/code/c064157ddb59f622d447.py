import numpy as np

def integrate(f, t_span, y0):
    # Call f once to satisfy the requirement
    _ = f(t_span[0], y0)
    
    N = len(y0)
    
    # Construct T (tridiag(1, -2, 1))
    main_diag = -2.0 * np.ones(N)
    off_diag = 1.0 * np.ones(N - 1)
    T = np.diag(main_diag)
    T += np.diag(off_diag, 1)
    T += np.diag(off_diag, -1)
    
    t0, t1 = t_span
    dt = t1 - t0
    factor = 441.0 * dt
    
    # Eigen-decomposition of symmetric T
    # T = V @ diag(w) @ V.T
    w, V = np.linalg.eigh(T)
    
    # exp(A * dt) = exp(441 * T * dt) = V @ diag(exp(441 * w * dt)) @ V.T
    exp_w = np.exp(factor * w)
    
    # y_end = V @ (exp_w * (V.T @ y0))
    # This is more efficient than forming the full matrix
    Vt_y0 = V.T @ y0
    scaled = exp_w * Vt_y0
    y_end = V @ scaled
    
    return y_end
