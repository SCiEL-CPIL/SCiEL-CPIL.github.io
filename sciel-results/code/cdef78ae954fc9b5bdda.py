def find_333_decomp(rank):
    n, m, p = 3, 3, 3
    T = np.ones((n, m, p))
    
    for seed in range(10):
        np.random.seed(seed)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        for iter in range(100):
            # Update U
            # R[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            # Fix V, W. Solve for U.
            # This is a linear system.
            # For each i, j:
            # sum_c U[i*m+j, c] * (V[j*p+k, c] W[k*n+i, c] summed over k?)
            # No, the term is U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
            # We want sum_c U[i*m+j, c] M_c[i, j, k] = T[i, j, k]
            # where M_c[i, j, k] = V[j*p+k, c] W[k*n+i, c].
            # This is not a simple matrix multiply.
            
            # Alternative: Use gradient descent.
            # Compute R
            R = np.zeros((n, m, p))
            for c in range(rank):
                uc = U[:, c]
                vc = V[:, c]
                wc = W[:, c]
                # Reshape to (n, m), (m, p), (p, n)
                uc_r = uc.reshape(n, m)
                vc_r = vc.reshape(m, p)
                wc_r = wc.reshape(p, n)
                # Term is uc[i, j] * vc[j, k] * wc[k, i]
                # Sum over c.
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            R[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
                            
            # Gradient
            # dL/dU[i, j, c] = sum_k (R[i, j, k] - T[i, j, k]) * V[j*p+k, c] * W[k*n+i, c]
            # This is slow in Python.
            
            # Better: Vectorize.
            # R = sum_c U_c * V_c * W_c (element-wise product of reshaped tensors?)
            # No, the indices are coupled.
            
            pass
