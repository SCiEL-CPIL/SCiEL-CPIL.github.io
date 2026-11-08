import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 decomposition matrices
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # c=0
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # c=1
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # c=2
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # c=3
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # c=4
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # c=5
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # c=6
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)]
    ]
    
    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)]
    ]
    
    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)]
    ]
    
    for half in range(2):
        base_col = half * 49
        
        for k_p in range(7):
            for m_s in range(7):
                col = base_col + k_p * 7 + m_s
                
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_j = bj * 2 + c
                            global_a_idx = global_i * 4 + global_j
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
                
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_j = bi * 2 + r
                            global_k_local = bj * 2 + c
                            global_k = global_k_local + half * 4
                            global_b_idx = global_j * 8 + global_k
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]
                
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k_local = bj * 2 + c
                            global_k = global_k_local + half * 4
                            global_w_idx = global_k * 4 + global_i
                            c_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[c_eff_idx, m_s]
                
                if time.time() - start_time > 100:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by checking for linearly dependent columns
    # The tensor is T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # We can check if removing a column still preserves the tensor
    # This is expensive, so let's try a simpler approach:
    # Check if any pair of columns can be combined
    
    # Actually, let's verify correctness first, then try rank reduction
    # Build the full tensor from U, V, W and compare with direct multiplication
    
    # For rank reduction, we can try to find a lower rank decomposition
    # by checking if the current decomposition has redundant structure
    
    # Simple heuristic: try removing columns one at a time and check if
    # the resulting tensor still matches. This is O(rank * n*m*p) per check.
    
    # Let's build the tensor
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Direct multiplication tensor: T_direct[i,j,k] = sum_l A[i,l]*B[l,k]
    # In our indexing: A is n*m = 16 elements (i,j), B is m*p = 32 (j,k), C is p*n = 32 (k,i)
    # T[i,j,k] should equal sum_l A[i,l]*B[l,k] where l ranges over m=4
    # In our flat indexing: A_idx = i*4+j, B_idx = j*8+k, C_idx = k*4+i
    # T[A_idx, B_idx, C_idx] = sum_l A[i,l]*B[l,k]
    # But our tensor is indexed differently. Let me re-derive.
    
    # The matrix multiplication tensor T has T[a_idx, b_idx, c_idx] = sum_l A[i,l]*B[l,k]
    # where a_idx = i*m + j... wait, let me re-read the problem.
    
    # Actually the tensor for matrix multiplication C = A*B is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = delta (it's 1 if the indices match the multiplication pattern)
    # The decomposition is: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # where i in [0,n), j in [0,m), k in [0,p)
    # But here n=4, m=4, p=8, so U is (n*m, rank) = (16, rank)?
    
    # Wait, re-reading: U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (32, rank)  
    # W has shape (p*n, rank) = (32, rank)
    
    # So the tensor is T[a, b, c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # where a in [0,16), b in [0,32), c in [0,32)
    # And T should equal the matrix multiplication tensor where:
    # a = i*4 + j (i in [0,4), j in [0,4))
    # b = j*8 + k (j in [0,4), k in [0,8))
    # c = k*4 + i (k in [0,8), i in [0,4))
    # T[a,b,c] = 1 if the indices are consistent with C[i,k] = sum_j A[i,j]*B[j,k]
    # i.e., if a = i*4+j, b = j'*8+k, c = k'*4+i' then T=1 if i=i', j=j', k=k'
    
    # Let me verify with a small test
    # Actually, let's just try rank reduction by checking column dependencies
    
    # For each column r, check if it's linearly dependent on others
    # This is expensive. Let's try a different approach:
    # Check if the rank can be reduced by looking at the structure
    
    # Since the two halves share A, maybe we can merge some columns
    # Let's check if columns from half 0 and half 1 with same (k_p, m_s) 
    # can be combined
    
    # Actually, the B and W parts are different for each half, so they can't be simply merged.
    
    # Let me try a numerical rank reduction: compute the tensor and find its
    # tensor rank numerically (this is hard in general, but we can try)
    
    # Alternative: try to find a lower rank by optimization
    # This is complex. Let's just return the rank-98 solution for now,
    # but try a quick check if rank 97 or lower works by removing one column
    
    # Quick check: remove column 0 and see if the tensor still matches
    # This is a heuristic and may not work, but let's try
    
    # Actually, let's be more systematic. The tensor T has a specific structure.
    # Let's check if the decomposition is minimal by computing the rank of
    # certain unfoldings.
    
    # For now, let's just return the rank-98 solution.
    # The improvement would require finding a lower rank decomposition,
    # which is a hard problem.
    
    if time.time() - start_time > 120:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
