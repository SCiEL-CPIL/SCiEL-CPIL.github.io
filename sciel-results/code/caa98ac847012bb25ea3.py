# ... (previous imports)

def _verify_full(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Full verification using random inputs."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    # Reshape
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    
    # Test with a few random matrices
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C'
        C_comp = np.zeros((n, p))
        for r in range(rank):
            # L_r(A) is a scalar? No, U is (n, m, rank).
            # The term is W[k, i, r] * (sum_j U[i, j, r] A[i, j]) * (sum_j V[j, k, r] B[j, k])
            # Wait, the standard form is:
            # C[i, k] = sum_r W[k, i, r] * M_r
            # M_r = (sum_{i,j} U[i, j, r] A[i, j]) * (sum_{j,k} V[j, k, r] B[j, k])
            # This is NOT correct.
            # The bilinear form is:
            # C[i, k] = sum_r W[k, i, r] * (sum_j U[i, j, r] A[i, j]) * (sum_j V[j, k, r] B[j, k])
            # This assumes U and V depend on i and k?
            # No, U is a matrix.
            # Let's use the tensor definition:
            # C[i, k] = sum_j A[i, j] B[j, k]
            # = sum_j sum_r U[i, j, r] V[j, k, r] W[k, i, r] A[i, j] B[j, k] ??
            # No.
            # The decomposition is:
            # sum_r (sum_{i,j} U[i, j, r] A[i, j]) (sum_{j,k} V[j, k, r] B[j, k]) W[k, i, r]
            # This does not match indices.
            
            # Correct Bilinear Complexity Definition:
            # We have R bilinear multiplications M_r.
            # M_r = L_r(A) R_r(B).
            # C[i, k] = sum_r W[k, i, r] M_r.
            # L_r(A) = sum_{i,j} U[i, j, r] A[i, j].
            # R_r(B) = sum_{j,k} V[j, k, r] B[j, k].
            # This is not separable in i, k.
            
            # Actually, for matrix multiplication, the standard decomposition is:
            # C[i, k] = sum_j A[i, j] B[j, k].
            # We want to write this as sum_r W_r(i, k) L_r(A) R_r(B).
            # L_r(A) is linear in A. R_r(B) is linear in B.
            # L_r(A) = sum_{i,j} U[i, j, r] A[i, j].
            # R_r(B) = sum_{j,k} V[j, k, r] B[j, k].
            # Then C[i, k] = sum_r W[i, k, r] L_r(A) R_r(B).
            # This must equal sum_j A[i, j] B[j, k].
            
            # Let's check if the current U, V, W satisfy this.
            # The current code uses:
            # T[i, k] = sum_{j, r} U[i, j, r] V[j, k, r] W[k, i, r].
            # This is NOT the bilinear form above.
            # The bilinear form is:
            # C[i, k] = sum_r W[k, i, r] (sum_{i', j'} U[i', j', r] A[i', j']) (sum_{j'', k''} V[j'', k'', r] B[j'', k''])
            # This is a scalar M_r.
            # So C[i, k] is the same for all i, k? No.
            # W[k, i, r] depends on i, k.
            
            # So the algorithm is:
            # 1. Compute R scalars M_r = L_r(A) R_r(B).
            # 2. Compute C[i, k] = sum_r W[k, i, r] M_r.
            
            # Let's implement this.
            M = np.zeros(rank)
            for r in range(rank):
                L_r = np.dot(Ur[:, :, r].flatten(), A.flatten())
                R_r = np.dot(Vr[:, :, r].flatten(), B.flatten())
                M[r] = L_r * R_r
            
            # C[i, k] = sum_r Wr[k, i, r] M[r]
            # Wr is (p, n, rank).
            # C_comp[i, k] = sum_r Wr[k, i, r] M[r]
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] = np.dot(Wr[k, i, :], M)
            
        if not np.allclose(C_expected, C_comp, atol=atol):
            return False
    
    return True

