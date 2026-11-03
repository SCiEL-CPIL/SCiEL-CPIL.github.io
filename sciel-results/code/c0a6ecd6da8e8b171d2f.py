"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Define blocks
    # i: 0..2
    # j: 0..4
    # k: 0..4
    
    # Split i into [0,1] and [2]
    # Split j into [0,1], [2,3], [4]
    # Split k into [0,1], [2,3], [4]
    
    i_splits = [[0, 1], [2]]
    j_splits = [[0, 1], [2, 3], [4]]
    k_splits = [[0, 1], [2, 3], [4]]
    
    U_list = []
    V_list = []
    W_list = []
    
    # Strassen vectors for 2x2x2
    # U, V, W are 4x7 matrices
    # Indexing: A: 00, 01, 10, 11 -> 0, 1, 2, 3
    #           B: 00, 01, 10, 11 -> 0, 1, 2, 3
    #           C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # We will generate these and verify
    def get_strassen_factors():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def get_schoolbook_factors(n_block, m_block, p_block):
        rank = n_block * m_block * p_block
        U = np.zeros((n_block * m_block, rank))
        V = np.zeros((m_block * p_block, rank))
        W = np.zeros((p_block * n_block, rank))
        
        col = 0
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    U[i * m_block + j, col] = 1
                    V[j * p_block + k, col] = 1
                    W[k * n_block + i, col] = 1
                    col += 1
        return U, V, W

    strassen_U, strassen_V, strassen_W = get_strassen_factors()
    
    # Verify Strassen
    # Check if sum_r U_r V_r W_r == Identity Tensor
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r] ? No.
    # Tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # For 2x2x2:
    # U is 4x7, V is 4x7, W is 4x7.
    # i,j in 0..3 (A indices), j,k in 0..3 (B indices), k,i in 0..3 (C indices)
    # Wait, the indices in U are (i, j) flattened.
    # U[im+j, r]
    # V[jp+k, r]
    # W[kn+i, r]
    
    # Let's verify the product
    # C = A @ B
    # A is 2x2, B is 2x2.
    # Flatten A to 4, B to 4.
    # We want to check if the decomposition works.
    
    # Construct full U, V, W
    for i_block in i_splits:
        for j_block in j_splits:
            for k_block in k_splits:
                n_b = len(i_block)
                m_b = len(j_block)
                p_b = len(k_block)
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    u_loc, v_loc, w_loc = strassen_U, strassen_V, strassen_W
                else:
                    u_loc, v_loc, w_loc = get_schoolbook_factors(n_b, m_b, p_b)
                
                rank_block = u_loc.shape[1]
                
                # Expand to global size
                U_global = np.zeros((n * m, rank_block))
                V_global = np.zeros((m * p, rank_block))
                W_global = np.zeros((p * n, rank_block))
                
                for r in range(rank_block):
                    # Map local indices to global
                    for i_local in range(n_b):
                        i_global = i_block[i_local]
                        for j_local in range(m_b):
                            j_global = j_block[j_local]
                            U_global[i_global * m + j_global, r] = u_loc[i_local * m_b + j_local, r]
                            
                    for j_local in range(m_b):
                        j_global = j_block[j_local]
                        for k_local in range(p_b):
                            k_global = k_block[k_local]
                            V_global[j_global * p + k_global, r] = v_loc[j_local * p_b + k_local, r]
                            
                    for k_local in range(p_b):
                        k_global = k_block[k_local]
                        for i_local in range(n_b):
                            i_global = i_block[i_local]
                            W_global[k_global * n + i_global, r] = w_loc[k_local * n_b + i_local, r]
                            
                U_list.append(U_global)
                V_list.append(V_global)
                W_list.append(W_global)
                
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    # Verify the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can check a few random entries or all entries.
    # Since n*m*p = 75, it's small.
    is_valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(U.shape[1]):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Expected value:
                # The tensor for matrix multiplication is 1 if the indices correspond to a valid multiplication?
                # No, the tensor T is defined such that C = A B is computed by contracting A, B with T.
                # T[i, j, k] is 1 if i, j, k form a valid path?
                # Actually, the tensor T for matrix multiplication has T[i, j, k] = 1 for all i, j, k?
                # No.
                # The decomposition represents the bilinear map.
                # The "tensor" is the set of coefficients.
                # For schoolbook, T[i, j, k] is 1 for the specific (i, j, k) triple?
                # In the schoolbook decomposition, each rank-1 term corresponds to a specific (i, j, k).
                # U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
                # So for a specific r, the term is A[i,j] B[j,k] C[k,i]?
                # The product is A[i,j] B[j,k] * 1.
                # This contributes to C[i,k].
                # The evaluator checks if the decomposition is exact.
                # This means that for any A, B, the result C computed via the decomposition matches A @ B.
                # C[i,k] = sum_j A[i,j] B[j,k].
                # In the decomposition:
                # C[i,k] = sum_r (sum_{i',j'} U[i',j',r] A[i',j']) (sum_{j',k'} V[j',k',r] B[j',k']) W[k',i',r] ?
                # No, the structure is:
                # M_r = (u_r^T A) (v_r^T B)
                # C = sum_r M_r w_r^T ?
                # W is (p*n, rank).
                # The output is C[i,k].
                # W[k*n+i, r] is the coefficient for C[i,k] from M_r.
                # So C[i,k] = sum_r M_r * W[k*n+i, r].
                
                # So we need to check if for all i, j, k, the "virtual" tensor element matches 1?
                # Actually, the tensor T is defined by T[i, j, k] = 1?
                # No, T is the multiplication tensor.
                # T[i, j, k] is 1 if the indices are compatible?
                # For matrix multiplication, the tensor is sparse.
                # T[i, j, k] = 1 for all i, j, k?
                # If T[i, j, k] = 1, then C[i, k] = sum_j A[i, j] B[j, k].
                # This is exactly what we want.
                # So we should check if the reconstructed tensor T' has T'[i, j, k] = 1 for all i, j, k.
                
                if abs(val - 1.0) > 1e-9:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break
            
    if not is_valid:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
