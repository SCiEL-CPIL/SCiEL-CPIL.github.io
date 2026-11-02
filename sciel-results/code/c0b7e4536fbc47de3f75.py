<<<<<<< SEARCH
def _check_rank1_residual(R, tol=1e-9):
    """
    Check if tensor R (shape a, b, c) is rank 1.
    Returns (success, u, v, w) where R ~= outer(u, outer(v, w))
    """
    a, b, c = R.shape
    # Find a non-zero pivot
    pivot_found = False
    a0, b0, c0 = -1, -1, -1
    for i in range(a):
        for j in range(b):
            for k in range(c):
                if abs(R[i, j, k]) > tol:
                    a0, b0, c0 = i, j, k
                    pivot_found = True
                    break
            if pivot_found: break
        if pivot_found: break
    
    if not pivot_found:
        return True, np.zeros(a), np.zeros(b), np.zeros(c)
    
    val = R[a0, b0, c0]
    
    # Assume w_c0 is the scaling factor
    # w_c = R[a0, b0, c] / val
    w = R[a0, b0, :] / val
    # v_b = R[a0, b, c0] / val
    v = R[a0, :, c0] / val
    # u_a = R[a, b0, c0] / val
    u = R[:, b0, c0] / val
    
    # Verify
    # R_recon = np.einsum('a,b,c->abc', u, v, w)
    # Check max error
    err = 0.0
    # Sample check for speed, or full check if small
    for i in range(a):
        for j in range(b):
            for k in range(c):
                recon = u[i] * v[j] * w[k]
                if abs(recon - R[i, j, k]) > tol:
                    return False, None, None, None
    return True, u, v, w
=======
def _rank1_residual_loss(R, tol=1e-9):
    """
    Compute a smooth loss measuring how far R is from being rank 1.
    Returns (loss, u, v, w) where u,v,w are the best rank-1 approximation.
    Uses iterative refinement.
    """
    a, b, c = R.shape
    
    # Find a non-zero pivot for initialization
    pivot_found = False
    a0, b0, c0 = 0, 0, 0
    max_val = 0.0
    for i in range(a):
        for j in range(b):
            for k in range(c):
                if abs(R[i, j, k]) > max_val:
                    max_val = abs(R[i, j, k])
                    a0, b0, c0 = i, j, k
                    pivot_found = True
    
    if not pivot_found or max_val < tol:
        return 0.0, np.zeros(a), np.zeros(b), np.zeros(c)
    
    val = R[a0, b0, c0]
    if abs(val) < tol:
        # Try different pivot
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    if abs(R[i, j, k]) > tol:
                        a0, b0, c0 = i, j, k
                        val = R[i, j, k]
                        break
                if abs(val) > tol:
                    break
            if abs(val) > tol:
                break
        if abs(val) < tol:
            return 0.0, np.zeros(a), np.zeros(b), np.zeros(c)
    
    # Initialize u, v, w from pivot
    w = R[a0, b0, :].copy() / val
    v = R[a0, :, c0].copy() / val
    u = R[:, b0, c0].copy() / val
    
    # Iteratively refine using alternating least squares
    for iteration in range(50):
        # Update u: for each i, u[i] = sum_{j,k} R[i,j,k] * v[j] * w[k] / (sum_{j,k} v[j]^2 * w[k]^2)
        denom_u = np.sum(v**2) * np.sum(w**2)
        if denom_u > tol:
            u_new = np.zeros(a)
            for i in range(a):
                numerator = np.sum(R[i, :, :] * np.outer(v, w))
                u_new[i] = numerator / denom_u
            u = u_new
        
        # Update v: for each j, v[j] = sum_{i,k} R[i,j,k] * u[i] * w[k] / (sum_{i,k} u[i]^2 * w[k]^2)
        denom_v = np.sum(u**2) * np.sum(w**2)
        if denom_v > tol:
            v_new = np.zeros(b)
            for j in range(b):
                numerator = np.sum(R[:, j, :] * np.outer(u, w))
                v_new[j] = numerator / denom_v
            v = v_new
        
        # Update w: for each k, w[k] = sum_{i,j} R[i,j,k] * u[i] * v[j] / (sum_{i,j} u[i]^2 * v[j]^2)
        denom_w = np.sum(u**2) * np.sum(v**2)
        if denom_w > tol:
            w_new = np.zeros(c)
            for k in range(c):
                numerator = np.sum(R[:, :, k] * np.outer(u, v))
                w_new[k] = numerator / denom_w
            w = w_new
    
    # Compute loss: ||R - outer(u, outer(v, w))||_F^2
    R_recon = np.einsum('a,b,c->abc', u, v, w)
    loss = np.sum((R - R_recon)**2)
    
    return loss, u, v, w
>>>>>>> REPLACE

