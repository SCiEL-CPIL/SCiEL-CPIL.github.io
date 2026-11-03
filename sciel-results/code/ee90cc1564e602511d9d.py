import numpy as np

def _verify(U, V, W):
    n, m, p = 3, 3, 7
    r = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We check if sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == T[i*m+j, j*p+k, k*n+i]
    
    # Instead of building the full tensor, we can check the reconstruction of C
    # But the task asks for tensor decomposition.
    # The condition is equivalent to:
    # For all i, j, k:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1 if (i,j,k) is valid else 0?
    # No, the tensor is sparse.
    # T_flat[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    
    # Let's check the bilinear form property.
    # C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Reconstruction: C_rec[i,k] = sum_c W[k*n+i, c] (sum_{i',j'} U[i'*m+j', c] A[i',j']) (sum_{j',k'} V[j'*p+k', c] B[j',k'])
    
    # We can test this with random matrices A and B.
    np.random.seed(0)
    for _ in range(10):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        C_rec = np.zeros((n, p))
        for c in range(r):
            u = Ua[:, c].reshape(n, m)
            v = Va[:, c].reshape(m, p)
            w = Wa[:, c].reshape(p, n) # w[k,i]
            
            # L_A = sum u[i,j] A[i,j]
            l_a = np.sum(u * A)
            # L_B = sum v[j,k] B[j,k]
            l_b = np.sum(v * B)
            
            # Contribution to C[i,k] is w[k,i] * l_a * l_b
            # w is p x n. w[k,i].
            # We need to add l_a * l_b * w.T to C_rec?
            # C_rec[i,k] += l_a * l_b * w[k,i]
            # w.T[i,k] = w[k,i].
            C_rec += l_a * l_b * w.T
        
        if not np.allclose(C_rec, C_true, atol=1e-7):
            return False
    return True
