def _residuals_fast(U, V, W, n, m, p, rank):
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Res[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is a contraction over r
    # Result shape: (n, m, p)
    # We can do:
    # T1[i, j, k] = sum_r U[i,j,r] V[j,k,r] -> (n, m, p)
    # T1 = np.einsum('ijr,jkr->ijk', U_r, V_r)
    # Res = T1 * W_r[k,i,r] ... wait, W depends on k,i,r.
    # So we need sum_r T1[i,j,k] * W[k,i,r]? No, T1 already summed over r.
    # We need sum_r (U[i,j,r] * V[j,k,r] * W[k,i,r])
    # This is not separable into two contractions easily because W depends on r.
    # We must sum over r for each (i,j,k).
    
    # Loop over r is still needed or use a 4D array
    # But n,m,p are small.
    # Let's stick to the loop but make it tight.
    
    res = np.ones((n, m, p))
    for r in range(rank):
        u_r = U_r[:, :, r]
        v_r = V_r[:, :, r]
        w_r = W_r[:, :, r]
        # u_r is (n, m), v_r is (m, p), w_r is (p, n)
        # We need res[i,j,k] -= u_r[i,j] * v_r[j,k] * w_r[k,i]
        # This is a 3-way product.
        # We can compute this as:
        # temp[i,j,k] = u_r[i,j] * v_r[j,k]
        # temp = np.einsum('ij,jk->ijk', u_r, v_r)
        # res -= temp * w_r.T? No, w_r[k,i] is w_r.T[i,k]?
        # w_r is (p, n). w_r[k,i] is element at row k, col i.
        # So w_r.T is (n, p). w_r.T[i,k] = w_r[k,i].
        # So res[i,j,k] -= temp[i,j,k] * w_r.T[i,k]
        # This is element-wise multiplication.
        temp = np.einsum('ij,jk->ijk', u_r, v_r)
        res -= temp * w_r.T[:, np.newaxis, :] # w_r.T is (n,p), need (n,1,p) to broadcast with (n,m,p)
    return res
