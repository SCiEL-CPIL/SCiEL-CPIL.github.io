def _als_iteration_fast(U, V, W, n, m, p, rank):
    # U update: for each (i,j), solve (V[j,:] ⊙ W_{:,i})^T @ U[i*m+j,:] = 1_p
    # V update: for each (j,k), solve (U_{:,j} ⊙ W_{k,:})^T @ V[j*p+k,:] = 1_n  
    # W update: for each (i,k), solve (U_{i,:} ⊙ V_{:,k})^T @ W[k*n+i,:] = 1_m
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    # U update: for each i,j: A[j,a] = V[j,k,a]*W[k,i,a] summed over k -> (m x rank)
    # Actually: sum_k V[j,k,a] * W[k,i,a] for each a -> this is (V[j] * Wr[:,i]).sum(axis=0)
    # So Gram[i,j] = (Vr[j] * Wr[:,i]).sum(axis=0)  shape (rank,)
    # Then we need Gram[i,j] @ U[i*m+j,:] = 1
    # This is a rank-1 system! U[i*m+j,:] = 1 / (Gram[i,j] @ Gram[i,j]) * Gram[i,j] if Gram is nonzero
    # Wait no, the system is: for each k: sum_a U[i*m+j,a] * V[j,k,a] * W[k,i,a] = 1
    # This is: (V[j] * Wr[:,i]) @ U[i*m+j,:] = ones(p)
    # V[j] is (p, rank), Wr[:,i] is (p, rank), element-wise product is (p, rank)
    # So A = (Vr[j] * Wr[:,i]) shape (p, rank), solve A @ U[i*m+j,:] = ones(p)
    
    for i in range(n):
        for j in range(m):
            A = Vr[j] * Wr[:, i]  # (p, rank)
            try:
                Ur[i, j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
            except:
                pass
    U = Ur.reshape(n*m, rank)
    
    # V update: for each j,k: A = (Ur[:,j] * Wr[k,:]) shape (n, rank), solve A @ V[j*p+k,:] = ones(n)
    Vr = V.reshape(m, p, rank)
    for j in range(m):
        for k in range(p):
            A = Ur[:, j] * Wr[k, :]  # (n, rank)
            try:
                Vr[j, k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
            except:
                pass
    V = Vr.reshape(m*p, rank)
    
    # W update: for each i,k: A = (Ur[i,:] * Vr[:,k]) shape (m, rank), solve A @ W[k*n+i,:] = ones(m)
    Wr = W.reshape(p, n, rank)
    for i in range(n):
        for k in range(p):
            A = Ur[i, :] * Vr[:, k]  # (m, rank)
            try:
                Wr[k, i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
            except:
                pass
    W = Wr.reshape(p*n, rank)
    
    return U, V, W
