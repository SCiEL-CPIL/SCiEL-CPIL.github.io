"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Use the known optimal rank-18 decomposition for 4x4x4 matrix multiplication
    # and extend it to handle the 6-column dimension.
    # 
    # For 4x4 * 4x4, rank is 18. We can use this as a building block.
    # For 4x4 * 4x6, we can view it as 6 separate 4x4 * 4x1 multiplications,
    # but a better approach is to use the rank-18 decomposition structure.
    #
    # The rank-18 decomposition for 4x4x4 uses specific linear combinations.
    # Let me implement the standard rank-18 algorithm.
    
    # The rank-18 decomposition for 4x4x4 matrix multiplication:
    # This is a known result. Let me construct it explicitly.
    
    # For now, let's use a simple but correct approach:
    # Direct multiplication has rank 4*4*4 = 64 for 4x4x4, but we know 18 is optimal.
    # For 4x4x6, a simple upper bound is 4*4*6 = 96, but we can do better.
    
    # Let me use a block approach: partition the 6 columns into groups.
    # Actually, let me just implement a correct decomposition, even if not optimal.
    # The key is to get it working first.
    
    # Simple approach: for each output element C[i][k], it's sum_j A[i][j]*B[j][k]
    # This gives rank = n*m*p = 4*4*6 = 96
    # But we can do better with Strassen-like methods.
    
    # Let me use the Strassen approach correctly this time.
    # Partition A (4x4) into 2x2 blocks of 2x2
    # Partition B (4x6) into 2x3 blocks of 2x2
    # C (4x6) will have 2x3 blocks of 2x2
    
    # For each pair of blocks that contribute to a C block, use Strassen.
    # C_ik = A_i0 * B_0k + A_i1 * B_1k for i in {0,1}, k in {0,1,2}
    
    # Each block product (2x2)*(2x2) uses 7 Strassen multiplications.
    # For each C block, we have 2 block products, each using 7 mults = 14 mults.
    # Total C blocks: 2*3 = 6, so total rank = 6 * 14 = 84
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for i in range(2):  # A row block index (0 or 1)
        for k in range(3):  # C col block index (0, 1, or 2)
            # C_ik = A_i0 * B_0k + A_i1 * B_1k
            
            for j in range(2):  # Sum over j: A_ij * B_jk
                # Block A_ij: rows [2*i, 2*i+1], cols [2*j, 2*j+1]
                # Block B_jk: rows [2*j, 2*j+1], cols [2*k, 2*k+1]
                
                # A_ij elements:
                a_idx = 2*i * m + 2*j
                b_idx = 2*i * m + 2*j + 1
                c_idx = (2*i+1) * m + 2*j
                d_idx = (2*i+1) * m + 2*j + 1
                
                # B_jk elements:
                e_idx = 2*j * p + 2*k
                f_idx = 2*j * p + 2*k + 1
                g_idx = (2*j+1) * p + 2*k
                h_idx = (2*j+1) * p + 2*k + 1
                
                # C_ik block: rows [2*i, 2*i+1], cols [2*k, 2*k+1]
                # W index: W[col*n + row, r]
                c11_idx = (2*k) * n + 2*i
                c12_idx = (2*k+1) * n + 2*i
                c21_idx = (2*k) * n + 2*i + 1
                c22_idx = (2*k+1) * n + 2*i + 1
                
                # Strassen 7 multiplications
                # M1 = (a+d)(e+h)
                U[a_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M2 = (c+d)e
                U[c_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                W[c21_idx][col] = 1
                W[c22_idx][col] = -1
                col += 1
                
                # M3 = a(f-h)
                U[a_idx][col] = 1
                V[f_idx][col] = 1
                V[h_idx][col] = -1
                W[c12_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M4 = d(g-e)
                U[d_idx][col] = 1
                V[g_idx][col] = 1
                V[e_idx][col] = -1
                W[c11_idx][col] = 1
                W[c21_idx][col] = 1
                col += 1
                
                # M5 = (a+b)h
                U[a_idx][col] = 1
                U[b_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = -1
                W[c12_idx][col] = 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[c_idx][col] = 1
                U[a_idx][col] = -1
                V[e_idx][col] = 1
                V[f_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[b_idx][col] = 1
                U[d_idx][col] = -1
                V[g_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
