import numpy as np

def verify():
    n, m, p = 2, 3, 9
    result = construct()
    U, V, W = result['U'], result['V'], result['W']
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check: for random A (2x3) and B (3x9), C = A@B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C[i,k] = sum_c U[i*m+j... 
    # Actually: C[i,k] = sum_c (sum_j U[i*m+j, c] * ... )
    
    # The tensor: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # Rank-1: U[i,j] * V[j,k] * W[k,i]
    # C[i,k] = sum_c U[i,j_c] V[j_c,k] W[k,i] ... no
    
    # C[i,k] = sum_c (U_{i*,c} dot A_i) * (V_{*k,c} dot B_k) * W[k,i,c]
    # More precisely: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
    
    for trial in range(10):
        A = np.random.randn(2, 3)
        B = np.random.randn(3, 9)
        C_true = A @ B
        
        C_comp = np.zeros((2, 9))
        for c in range(len(U[0])):
            a_part = sum(U[i*3+j, c] * A[i, j] for i in range(2) for j in range(3))
            # Wait, U has shape (6, rank), indexed U[i*m+j, c]
            # The "a_part" for term c is: sum over all (i,j) of U[i*m+j,c] * A[i,j]
            # But that's not right either. Let me think...
            
            # C[i,k] = sum_c W[k*2+i, c] * (sum_j U[i*3+j, c] * A[i,j]) * (sum_j V[j*9+k, c] * B[j,k])
            
            # This doesn't decompose cleanly per i,k. Let me just compute the full thing.
        
        # Better: C[i,k] = sum_c W[k*2+i, c] * (U_row_i @ A_i) * (V_col_k @ B_col_k)
        # where U_row_i = [U[i*3+0,c], U[i*3+1,c], U[i*3+2,c]] for fixed c
        # and V_col_k = [V[0*9+k,c], V[1*9+k,c], V[2*9+k,c]] for fixed c
        
        for i in range(2):
            for k in range(9):
                val = 0
                for c in range(len(U[0])):
                    u_val = sum(U[i*3+j, c] * A[i, j] for j in range(3))
                    v_val = sum(V[j*9+k, c] * B[j, k] for j in range(3))
                    val += W[k*2+i, c] * u_val * v_val
                C_comp[i, k] = val
        
        if not np.allclose(C_true, C_comp, atol=1e-10):
            print(f"Trial {trial}: MISMATCH")
            print("True:", C_true)
            print("Comp:", C_comp)
            return False
    print("All trials passed!")
    return True
