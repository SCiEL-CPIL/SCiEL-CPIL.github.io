<<<<<<< SEARCH
def _bound_grad(params):
    """Compute bound and its gradient analytically."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H
    
    # Vectorized moment computation
    M0 = np.array([[_I[i+j] for j in range(7)] for i in range(7)])
    M2 = np.array([[_I[1+i+j] for j in range(7)] for i in range(7)])
    
    m0 = p @ M0 @ p
    m2 = p @ M2 @ p
    
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    q = p * signs
    
    n0 = 2.0 * np.pi * (q @ M0 @ q)
    n2 = 2.0 * np.pi * (q @ M2 @ q)
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient: d(bound)/da_c = (vF * d(vf)/da_c + vf * d(vF)/da_c) / (2*bound)
    # d(vf)/da_c = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    
    # da12/da_c for c = a0, a4, a8:
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp_k/da_c = H[c,k] + da12[c] * H[3,k]
    grad = np.zeros(3)
    for c in range(3):
        dp = _H[c, :] + da12[c] * _H[3, :]
        
        # dm0/da_c = 2 * dp^T M0 p
        # dm2/da_c = 2 * dp^T M2 p
        dm0 = 2.0 * dp @ M0 @ p
        dm2 = 2.0 * dp @ M2 @ p
        
        dq = dp * signs
        
        dn0 = 4.0 * np.pi * (dq @ M0 @ q)
        dn2 = 4.0 * np.pi * (dq @ M2 @ q)
        
        dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm + vf * dvn) / (2.0 * bound)
    
    return bound, grad
=======
def _bound_grad(params):
    """Compute bound and its gradient analytically."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H
    
    # Use precomputed moment matrices
    m0 = p @ _M0 @ p
    m2 = p @ _M2 @ p
    
    q = p * _SIGNS
    
    n0 = 2.0 * np.pi * (q @ _M0 @ q)
    n2 = 2.0 * np.pi * (q @ _M2 @ q)
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Gradient: d(bound)/da_c = (vF * d(vf)/da_c + vf * d(vF)/da_c) / (2*bound)
    # d(vf)/da_c = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    
    # da12/da_c for c = a0, a4, a8:
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # dp_k/da_c = H[c,k] + da12[c] * H[3,k]
    grad = np.zeros(3)
    for c in range(3):
        dp = _H[c, :] + da12[c] * _H[3, :]
        
        # dm0/da_c = 2 * dp^T M0 p
        # dm2/da_c = 2 * dp^T M2 p
        dm0 = 2.0 * dp @ _M0 @ p
        dm2 = 2.0 * dp @ _M2 @ p
        
        dq = dp * _SIGNS
        
        dn0 = 4.0 * np.pi * (dq @ _M0 @ q)
        dn2 = 4.0 * np.pi * (dq @ _M2 @ q)
        
        dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)
        
        grad[c] = (vF * dvm + vf * dvn) / (2.0 * bound)
    
    return bound, grad
>>>>>>> REPLACE

Now let me enhance the `construct` function with better optimization strategy - reparameterization to equalize scales and more diverse starting points:

