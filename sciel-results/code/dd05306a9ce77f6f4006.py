"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 decomposition using Strassen for 2x2x2 blocks and schoolbook for remainders
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # local_i, local_j, local_k are 0 or 1
    # global_i_offset, global_j_offset, global_k_offset
    def add_strassen_block(g_i, g_j, g_k, start_col):
        # g_i, g_j, g_k are the base indices for i, j, k
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        u00 = g_i * m + g_j
        u01 = g_i * m + (g_j + 1)
        u10 = (g_i + 1) * m + g_j
        u11 = (g_i + 1) * m + (g_j + 1)
        
        v00 = g_j * p + g_k
        v01 = g_j * p + (g_k + 1)
        v10 = (g_j + 1) * p + g_k
        v11 = (g_j + 1) * p + (g_k + 1)
        
        w00 = g_k * n + g_i
        w01 = g_k * n + (g_i + 1)
        w10 = (g_k + 1) * n + g_i
        w11 = (g_k + 1) * n + (g_i + 1)
        
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        U[u00][start_col] += 1
        U[u11][start_col] += 1
        V[v00][start_col] += 1
        V[v11][start_col] += 1
        W[w00][start_col] += 1
        W[w11][start_col] += 1
        start_col += 1
        
        # M2: (A10+A11)B00 -> C10 - C11
        U[u10][start_col] += 1
        U[u11][start_col] += 1
        V[v00][start_col] += 1
        W[w10][start_col] += 1
        W[w11][start_col] -= 1
        start_col += 1
        
        # M3: A00(B01-B11) -> C01 + C11
        U[u00][start_col] += 1
        V[v01][start_col] += 1
        V[v11][start_col] -= 1
        W[w01][start_col] += 1
        W[w11][start_col] += 1
        start_col += 1
        
        # M4: A11(B10-B00) -> C00 + C10
        U[u11][start_col] += 1
        V[v10][start_col] += 1
        V[v00][start_col] -= 1
        W[w00][start_col] += 1
        W[w10][start_col] += 1
        start_col += 1
        
        # M5: (A00+A01)B11 -> -C00 + C01
        U[u00][start_col] += 1
        U[u01][start_col] += 1
        V[v11][start_col] += 1
        W[w00][start_col] -= 1
        W[w01][start_col] += 1
        start_col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[u10][start_col] += 1
        U[u00][start_col] -= 1
        V[v00][start_col] += 1
        V[v01][start_col] += 1
        W[w11][start_col] += 1
        start_col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U[u01][start_col] += 1
        U[u11][start_col] -= 1
        V[v10][start_col] += 1
        V[v11][start_col] += 1
        W[w00][start_col] += 1
        start_col += 1
        
        return start_col

    # Part 1: M_{2,2,7} for j in {0,1}, k in {0..6}
    # Block 1: k=0,1
    col = add_strassen_block(0, 0, 0, col)
    # Block 2: k=2,3
    col = add_strassen_block(0, 0, 2, col)
    # Block 3: k=4,5
    col = add_strassen_block(0, 0, 4, col)
    
    # Block 4: k=6 (M_{2,2,1})
    # Schoolbook for 2x2x1
    # A is 2x2 (i,j), B is 2x1 (j,k=6), C is 2x1 (i,k=6)
    # C_i6 = A_i0 B_06 + A_i1 B_16
    
    # Term 1: A_00 B_06 -> C_06
    U[0][col] += 1
    V[6][col] += 1
    W[12][col] += 1
    col += 1
    
    # Term 2: A_01 B_16 -> C_06
    U[1][col] += 1
    V[13][col] += 1
    W[12][col] += 1
    col += 1
    
    # Term 3: A_10 B_06 -> C_16
    U[3][col] += 1
    V[6][col] += 1
    W[13][col] += 1
    col += 1
    
    # Term 4: A_11 B_16 -> C_16
    U[4][col] += 1
    V[13][col] += 1
    W[13][col] += 1
    col += 1
    
    # Part 2: M_{2,1,7} for j=2, k in {0..6}
    # C_i,k = A_i2 B_2k
    # Rank 7: For each k, (e_0+e_1) u (e_2) v (e_k) w
    for k in range(7):
        U[0][col] += 1
        U[1][col] += 1
        V[2][col] += 1 # j=2, k=0..6. V index 2*7+k? No.
        # V index is j*p + k. j=2.
        # So V[2*7 + k] = V[14+k]
        V[14 + k][col] += 1
        W[k * n][col] += 1 # W index k*n + i. i=0,1.
        # Wait, the term is (e_0 + e_1) u e_2 v e_k w
        # This means U[0]=1, U[1]=1. V[2,k]=1. W[0,k]=1, W[1,k]=1?
        # No, W[k*n + i] corresponds to C[i,k].
        # The tensor term is u_i v_j w_k.
        # Sum over r u_{i,r} v_{j,r} w_{k,r}.
        # For fixed k, we need C[0,k] += A[0,2]B[2,k] and C[1,k] += A[1,2]B[2,k].
        # If we set U[0,r]=1, U[1,r]=1, V[14+k,r]=1, W[k,r]=1, W[k+1,r]=1?
        # Then C[0,k] += 1*1*1 = 1. C[1,k] += 1*1*1 = 1.
        # But we need C[0,k] += A[0,2]B[2,k].
        # The factors U,V,W are constant matrices.
        # The contraction is sum_r U[i,r] V[j,r] W[k,r].
        # If U[0,r]=1, U[1,r]=1, V[14+k,r]=1, W[k,r]=1, W[k+1,r]=1 is not possible for one r to serve both i=0 and i=1 with different W indices?
        # W index is k*n + i.
        # For i=0, W index is k*2+0 = 2k.
        # For i=1, W index is k*2+1 = 2k+1.
        # So we need W[2k, r]=1 and W[2k+1, r]=1.
        # And U[0, r]=1, U[1, r]=1.
        # And V[14+k, r]=1.
        # Then C[0,k] += U[0,r]V[14+k,r]W[2k,r] = 1*1*1 = 1.
        # C[1,k] += U[1,r]V[14+k,r]W[2k+1,r] = 1*1*1 = 1.
        # This works!
        
        # Correction in code above:
        # W[k*n][col] was W[2k]. We also need W[2k+1].
        # Let's fix the loop below.
        
    # Redo Part 2 correctly
    col -= 7 # backtrack
    for k in range(7):
        U[0][col] += 1
        U[1][col] += 1
        V[14 + k][col] += 1
        W[2 * k][col] += 1
        W[2 * k + 1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
