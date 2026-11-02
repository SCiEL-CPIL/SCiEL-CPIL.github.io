# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook(n, m, p):
    """Construct the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify exact decomposition of the matrix multiplication tensor."""
    # The tensor T is defined by T[i, j, k, i', j', k'] = delta_{i, i'} delta_{j, j'} delta_{k, k'}
    # Wait, the convention is W[k*n+i, r].
    # Let's check the indices in the original verify:
    # expected = 1 if (ip == i and jp == j and kp == k)
    # computed = sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
    # This corresponds to the tensor contraction:
    # T_{i,j,k, ip,jp,kp} = sum_r U_{i,j,r} V_{jp,k,r} W_{kp,ip,r}
    # Note the indices in V and W in the prompt's verify function:
    # V index: jp * p + k  (row index of V)
    # W index: kp * n + ip (row index of W)
    # U index: i * m + j   (row index of U)
    
    # Let's construct the tensor from U, V, W and compare to identity.
    # T[i, j, k, ip, jp, kp] = sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
    
    # We can check this by iterating or using einsum.
    # U: (n*m, rank) -> reshape to (n, m, rank)
    # V: (m*p, rank) -> reshape to (m, p, rank) ? 
    #    Index in V is jp * p + k. So row = jp*p + k.
    #    If we reshape V to (m, p, rank), then V[jp, k, r] corresponds to index jp*p+k.
    #    So V_reshaped[jp, k, r] = V[jp*p+k, r].
    # W: (p*n, rank) -> reshape to (p, n, rank) ?
    #    Index in W is kp * n + ip. So row = kp*n + ip.
    #    If we reshape W to (p, n, rank), then W[kp, ip, r] corresponds to index kp*n+ip.
    #    So W_reshaped[kp, ip, r] = W[kp*n+ip, r].
    # U: (n*m, rank) -> reshape to (n, m, rank)
    #    Index in U is i * m + j.
    #    If we reshape U to (n, m, rank), then U[i, j, r] corresponds to index i*m+j.
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute T[i, j, k, ip, jp, kp] = sum_r U_r[i, j, r] * V_r[jp, k, r] * W_r[kp, ip, r]
    # Using einsum:
    # U_r: i, j, r
    # V_r: jp, k, r  -> need to align indices for einsum
    # W_r: kp, ip, r
    
    # T = np.einsum('ijr,jkr,kpr->ijkipp', U_r, V_r, W_r) # This syntax is tricky with repeated indices in output
    # Better:
    # T_{i,j,k,ip,jp,kp} = sum_r U_{i,j,r} V_{jp,k,r} W_{kp,ip,r}
    
    # Let's just iterate to be safe and fast enough for small dims.
    for i in range(n):
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (ip == i and jp == j and kp == k) else 0.0
                            computed = 0.0
                            for r in range(rank):
                                computed += U[i * m + j, r] * V[jp * p + k, r] * W[kp * n + ip, r]
                            if abs(computed - expected) > 1e-10:
                                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    
    # Convert to lists if required, but numpy arrays are usually accepted as "matrices".
    # The prompt says "factor matrices of shapes...". Numpy arrays are matrices.
    # If lists are strictly required, we can convert.
    # Given "integer/half-integer entries", numpy int/float is fine.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
