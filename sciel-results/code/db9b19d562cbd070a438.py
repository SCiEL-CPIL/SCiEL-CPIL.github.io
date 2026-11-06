import numpy as np

def construct(training_x, training_subgrid_tendency):
    x = np.asarray(training_x, dtype=float).flatten()
    t = np.asarray(training_subgrid_tendency, dtype=float).flatten()
    
    mask = np.isfinite(x) & np.isfinite(t)
    x = x[mask]
    t = t[mask]
    
    if len(x) == 0:
        return np.zeros(6)
    
    # 1. Determine range for mapping to [-1, 1]
    # Using min/max is standard for Chebyshev, but can be sensitive to outliers.
    # Let's use a slightly robust range: 1st and 99th percentiles to avoid extreme leverage points
    # influencing the scaling too much, but keep it close to data range.
    # However, if we clip, we might extrapolate more. 
    # Let's stick to min/max for now as it's the standard domain for Chebyshev approximation of the data support.
    x_min = np.min(x)
    x_max = np.max(x)
    
    if x_max - x_min < 1e-10:
        # Degenerate case
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.mean(t)])

    # 2. Map x to u in [-1, 1]
    alpha = 2.0 / (x_max - x_min)
    beta = - (x_max + x_min) / (x_max - x_min)
    u = alpha * x + beta
    
    # 3. Construct Chebyshev Design Matrix
    n_samples = len(u)
    C = np.zeros((n_samples, 6))
    C[:, 0] = 1.0
    C[:, 1] = u
    for k in range(2, 6):
        C[:, k] = 2.0 * u * C[:, k-1] - C[:, k-2]
        
    # 4. Solve for Chebyshev coefficients a
    # Use SVD for stability
    try:
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        # Add small regularization to avoid division by zero or ill-conditioning
        # The singular values of Chebyshev matrix are well-behaved
        tol = np.finfo(float).eps * max(C.shape) * s[0]
        rank = np.sum(s > tol)
        
        # Ridge regression with tiny lambda for numerical safety
        lambda_reg = 1e-10 * (s[0] ** 2) if s[0] > 0 else 1e-10
        a = Vt.T @ (s / (s**2 + lambda_reg) * (U.T @ t))
    except np.linalg.LinAlgError:
        a, _, _, _ = np.linalg.lstsq(C, t, rcond=None)
        
    if not np.all(np.isfinite(a)):
        return np.zeros(6)
        
    # 5. Convert Chebyshev coefficients a to Monomial coefficients in u (m_u)
    # T_0 = 1
    # T_1 = u
    # T_2 = 2u^2 - 1
    # T_3 = 4u^3 - 3u
    # T_4 = 8u^4 - 8u^2 + 1
    # T_5 = 16u^5 - 20u^3 + 5u
    #
    # f(u) = a_0 T_0 + a_1 T_1 + a_2 T_2 + a_3 T_3 + a_4 T_4 + a_5 T_5
    # f(u) = (a_0 - a_2 + a_4) * 1
    #      + (a_1 - 3*a_3 + 5*a_5) * u
    #      + (2*a_2 - 8*a_4) * u^2
    #      + (4*a_3 - 20*a_5) * u^3
    #      + (8*a_4) * u^4
    #      + (16*a_5) * u^5
    
    m_u = np.zeros(6)
    m_u[0] = a[0] - a[2] + a[4]
    m_u[1] = a[1] - 3*a[3] + 5*a[5]
    m_u[2] = 2*a[2] - 8*a[4]
    m_u[3] = 4*a[3] - 20*a[5]
    m_u[4] = 8*a[4]
    m_u[5] = 16*a[5]
    
    # 6. Convert Monomial coefficients in u to Monomial coefficients in x
    # u = alpha * x + beta
    # P(x) = sum_{j=0}^5 m_u[j] * (alpha * x + beta)^j
    # (alpha * x + beta)^j = sum_{k=0}^j binom(j, k) * alpha^k * beta^(j-k) * x^k
    # c[k] = sum_{j=k}^5 m_u[j] * binom(j, k) * alpha^k * beta^(j-k)
    
    c = np.zeros(6)
    for k in range(6):
        sum_val = 0.0
        for j in range(k, 6):
            # Calculate binom(j, k)
            binom_val = 1.0
            for m in range(k):
                binom_val *= (j - m)
                binom_val /= (m + 1)
            
            term = m_u[j] * binom_val * (alpha ** k) * (beta ** (j - k))
            sum_val += term
        c[k] = sum_val
        
    if not np.all(np.isfinite(c)):
        return np.zeros(6)
        
    return c
