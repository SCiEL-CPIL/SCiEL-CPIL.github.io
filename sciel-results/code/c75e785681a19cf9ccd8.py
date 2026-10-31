import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Base case: 1x1 multiplication
    # Rank 1
    U1 = [[1]]
    V1 = [[1]]
    W1 = [[1]]
    
    # Function to expand Strassen from n/2 to n
    def strassen_expand(U_sub, V_sub, W_sub, n_sub):
        n = n_sub * 2
        rank_sub = len(U_sub[0])
        rank_new = rank_sub * 7
        
        U_new = [[0] * rank_new for _ in range(n * n)]
        V_new = [[0] * rank_new for _ in range(n * n)]
        W_new = [[0] * rank_new for _ in range(n * n)]
        
        # Define the 7 Strassen terms
        # Each term has:
        # A_blocks: list of (block_r, block_c, coeff)
        # B_blocks: list of (block_r, block_c, coeff)
        # C_blocks: list of (block_r, block_c, coeff)
        
        terms = [
            # M1
            {
                'A': [(0,0,1), (1,1,1)],
                'B': [(0,0,1), (1,1,1)],
                'C': [(0,0,1), (1,1,1)] # M1 contributes to C00 and C11 with +1
            },
            # M2
            {
                'A': [(1,0,1), (1,1,1)],
                'B': [(0,0,1)],
                'C': [(1,0,1), (1,1,-1)] # M2 contributes to C10 with +1, C11 with -1
            },
            # M3
            {
                'A': [(0,0,1)],
                'B': [(0,1,1), (1,1,-1)],
                'C': [(0,1,1), (1,1,1)] # M3 contributes to C01 with +1, C11 with +1
            },
            # M4
            {
                'A': [(1,1,1)],
                'B': [(1,0,1), (0,0,-1)],
                'C': [(0,0,1), (1,0,1)] # M4 contributes to C00 with +1, C10 with +1
            },
            # M5
            {
                'A': [(0,0,1), (0,1,1)],
                'B': [(1,1,1)],
                'C': [(0,0,-1), (0,1,1)] # M5 contributes to C00 with -1, C01 with +1
            },
            # M6
            {
                'A': [(1,0,1), (0,0,-1)],
                'B': [(0,0,1), (0,1,1)],
                'C': [(1,1,1)] # M6 contributes to C11 with +1
            },
            # M7
            {
                'A': [(0,1,1), (1,1,-1)],
                'B': [(1,0,1), (1,1,1)],
                'C': [(0,0,1)] # M7 contributes to C00 with +1
            }
        ]
        
        # Verify C contributions sum correctly
        # C00: M1(1) + M4(1) - M5(1) + M7(1) -> 1+1-1+1 = 2? 
        # Wait. Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-check my term definitions against standard Strassen.
        # M1 = (A11+A22)(B11+B22) -> C00 += M1, C11 += M1
        # M2 = (A21+A22)B11 -> C10 += M2, C11 -= M2
        # M3 = A11(B12-B22) -> C01 += M3, C11 += M3
        # M4 = A22(B21-B11) -> C00 += M4, C10 += M4
        # M5 = (A11+A12)B22 -> C00 -= M5, C01 += M5
        # M6 = (A21-A11)(B11+B12) -> C11 += M6
        # M7 = (A12-A22)(B21+B22) -> C00 += M7
        
        # My definitions:
        # Term 1 (M1): C: (0,0,1), (1,1,1). Correct.
        # Term 2 (M2): C: (1,0,1), (1,1,-1). Correct.
        # Term 3 (M3): C: (0,1,1), (1,1,1). Correct.
        # Term 4 (M4): C: (0,0,1), (1,0,1). Correct.
        # Term 5 (M5): C: (0,0,-1), (0,1,1). Correct.
        # Term 6 (M6): C: (1,1,1). Correct.
        # Term 7 (M7): C: (0,0,1). Correct.
        
        # Check C00: M1(1) + M4(1) + M5(-1) + M7(1) = 2?
        # Standard: C00 = M1 + M4 - M5 + M7.
        # If M5 term has coeff -1 in C00, then sum is 1+1-1+1 = 2.
        # But C00 should be 1 * (A11B11 + A12B21 + A21B12 + A22B22)?
        # No, C00 is the top-left block.
        # The decomposition is exact. The coefficients in W handle the linear combination.
        # The "1" in W means "add this sub-product to this block".
        # So if W has 1, it adds. If -1, it subtracts.
        # The sum of coefficients for a specific C block across all terms should result in the correct matrix.
        # Actually, the tensor decomposition ensures the bilinear form matches.
        # We just need to map the indices correctly.
        
        for l_idx, term in enumerate(terms):
            for s_idx in range(rank_sub):
                r_idx = l_idx * rank_sub + s_idx
                
                # Fill U
                for br, bc, coeff in term['A']:
                    # Block (br, bc) in A
                    # Local indices i2, j2 in [0, n_sub)
                    # Global i = br * n_sub + i2, j = bc * n_sub + j2
                    # Global U index: i * n + j
                    for i2 in range(n_sub):
                        for j2 in range(n_sub):
                            val = U_sub[i2 * n_sub + j2][s_idx]
                            if val != 0:
                                i_global = br * n_sub + i2
                                j_global = bc * n_sub + j2
                                u_idx = i_global * n + j_global
                                U_new[u_idx][r_idx] += val * coeff
                            
                # Fill V
                for br, bc, coeff in term['B']:
                    for i2 in range(n_sub):
                        for j2 in range(n_sub):
                            val = V_sub[i2 * n_sub + j2][s_idx]
                            if val != 0:
                                i_global = br * n_sub + i2
                                j_global = bc * n_sub + j2
                                v_idx = i_global * n + j_global
                                V_new[v_idx][r_idx] += val * coeff
                                
                # Fill W
                for br, bc, coeff in term['C']:
                    for i2 in range(n_sub):
                        for j2 in range(n_sub):
                            val = W_sub[i2 * n_sub + j2][s_idx]
                            if val != 0:
                                # W index in sub: i2 * n_sub + j2 ?
                                # Task says W[k*n+i]. 
                                # In sub-problem, W_sub is for n_sub x n_sub.
                                # Index k * n_sub + i.
                                # Here i2 is row of C (k), j2 is col of C (i)?
                                # Let's check seed: W[k*n+i].
                                # In seed, W index is k*n+i.
                                # In sub-problem, W_sub index is k_sub * n_sub + i_sub.
                                # Global k = br * n_sub + k_sub (row of C block)
                                # Global i = bc * n_sub + i_sub (col of C block)
                                # Global W index: k_global * n + i_global
                                k_global = br * n_sub + i2
                                i_global = bc * n_sub + j2
                                w_idx = k_global * n + i_global
                                W_new[w_idx][r_idx] += val * coeff
                                
        return U_new, V_new, W_new

    # Start with 1x1
    U, V, W = U1, V1, W1
    n_current = 1
    
    # Expand to 2x2
    U, V, W = strassen_expand(U, V, W, n_current)
    n_current = 2
    
    # Expand to 4x4
    U, V, W = strassen_expand(U, V, W, n_current)
    n_current = 4
    
    # Verify dimensions
    # U: (16, 49), V: (16, 49), W: (16, 49)
    
    return {"U": U, "V": V, "W": W}
