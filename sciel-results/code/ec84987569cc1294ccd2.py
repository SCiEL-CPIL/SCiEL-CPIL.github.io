"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # We decompose M_{4,5,5} into two M_{2,5,5} blocks.
    # M_{2,5,5} is decomposed using a 3+2 split of the middle dimension.
    # A (2x5) = [A_3 (2x3) | A_2 (2x2)]
    # B (5x5) = [[B_33 (3x3), B_32 (3x2)], [B_23 (2x3), B_22 (2x2)]]
    # C = A_3 B_33 + A_3 B_32 + A_2 B_23 + A_2 B_22
    
    # We use known low-rank decompositions for the sub-blocks.
    # M_{2,2,2}: Rank 7 (Strassen)
    # M_{2,3,3}: Rank 11 (Known)
    # M_{2,3,2}: Rank 11 (Using schoolbook 12, but we can try 11 if known, else 12)
    # M_{2,2,3}: Rank 11 (Using schoolbook 12)
    
    # For simplicity and guaranteed correctness, we will use:
    # M_{2,2,2} with Rank 7
    # M_{2,3,3} with Rank 11
    # M_{2,3,2} with Rank 12 (Schoolbook)
    # M_{2,2,3} with Rank 12 (Schoolbook)
    # Total Rank for M_{2,5,5} = 7 + 11 + 12 + 12 = 42
    # Total Rank for M_{4,5,5} = 2 * 42 = 84
    
    n, m, p = DIMENSIONS
    rank_total = 84
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Helper to add a decomposition
    def add_decomp(U, V, W, offset, U_sub, V_sub, W_sub, i_off, j_off, k_off, i_dim, j_dim, k_dim):
        # i_off, j_off, k_off are offsets in the full matrices
        # i_dim, j_dim, k_dim are dimensions of the sub-problem
        for i in range(i_dim):
            for j in range(j_dim):
                for k in range(k_dim):
                    # This is not how it works. U, V, W are already factored.
                    pass

    # We need to construct U, V, W for each sub-problem and then embed them.
    
    # 1. M_{2,2,2} Strassen
    # Indices: i in 0..1, j in 0..1, k in 0..1
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    def get_m222():
        U_sub = [[0]*7 for _ in range(4)]
        V_sub = [[0]*7 for _ in range(4)]
        W_sub = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        U_sub[0][0] = 1; U_sub[3][0] = 1
        V_sub[0][0] = 1; V_sub[3][0] = 1
        W_sub[0][0] = 1; W_sub[3][0] = 1
        # Wait, W index is k*n+i. For 2x2, W[k*2+i].
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's define the 7 multipliers explicitly in U, V, W.
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # m=2, p=2, n=2.
        
        # M1: (A11+A22) * (B11+B22)
        # A11 is i=0,j=0. A22 is i=1,j=1.
        # B11 is j=0,k=0. B22 is j=1,k=1.
        U_sub[0][0] = 1; U_sub[3][0] = 1
        V_sub[0][0] = 1; V_sub[3][0] = 1
        # C11 gets +1, C22 gets +1
        W_sub[0][0] = 1 # k=0,i=0 -> C11
        W_sub[3][0] = 1 # k=1,i=1 -> C22
        
        # M2: (A21+A22) * B11
        # A21 i=1,j=0. A22 i=1,j=1.
        # B11 j=0,k=0.
        U_sub[2][1] = 1; U_sub[3][1] = 1
        V_sub[0][1] = 1
        # C21 gets +1
        W_sub[2][1] = 1 # k=0,i=1 -> C21
        
        # M3: A11 * (B12-B22)
        # A11 i=0,j=0.
        # B12 j=0,k=1. B22 j=1,k=1.
        U_sub[0][2] = 1
        V_sub[1][2] = 1; V_sub[3][2] = -1
        # C12 gets +1
        W_sub[1][2] = 1 # k=1,i=0 -> C12
        
        # M4: A22 * (B21-B11)
        # A22 i=1,j=1.
        # B21 j=1,k=0. B11 j=0,k=0.
        U_sub[3][3] = 1
        V_sub[2][3] = 1; V_sub[0][3] = -1
        # C11 gets +1, C21 gets +1
        W_sub[0][3] = 1 # k=0,i=0 -> C11
        W_sub[2][3] = 1 # k=0,i=1 -> C21
        
        # M5: (A11+A12) * B22
        # A11 i=0,j=0. A12 i=0,j=1.
        # B22 j=1,k=1.
        U_sub[0][4] = 1; U_sub[1][4] = 1
        V_sub[3][4] = 1
        # C11 gets -1, C12 gets +1
        W_sub[0][4] = -1 # k=0,i=0 -> C11
        W_sub[1][4] = 1 # k=1,i=0 -> C12
        
        # M6: (A21-A11) * (B11+B12)
        # A21 i=1,j=0. A11 i=0,j=0.
        # B11 j=0,k=0. B12 j=0,k=1.
        U_sub[2][5] = 1; U_sub[0][5] = -1
        V_sub[0][5] = 1; V_sub[1][5] = 1
        # C22 gets +1
        W_sub[3][5] = 1 # k=1,i=1 -> C22
        
        # M7: (A12-A22) * (B21+B22)
        # A12 i=0,j=1. A22 i=1,j=1.
        # B21 j=1,k=0. B22 j=1,k=1.
        U_sub[1][6] = 1; U_sub[3][6] = -1
        V_sub[2][6] = 1; V_sub[3][6] = 1
        # C11 gets +1, C22 gets +1
        W_sub[0][6] = 1 # k=0,i=0 -> C11
        W_sub[3][6] = 1 # k=1,i=1 -> C22
        
        return U_sub, V_sub, W_sub

    # 2. M_{2,3,3} Rank 11
    # This is more complex. I will use a known decomposition or fallback to schoolbook if not available.
    # Schoolbook rank is 18.
    # If I use schoolbook for M_{2,3,3}, total rank increases.
    # Let's use schoolbook for M_{2,3,3}, M_{2,3,2}, M_{2,2,3} to be safe and quick to implement.
    # Rank M_{2,3,3} = 18
    # Rank M_{2,3,2} = 12
    # Rank M_{2,2,3} = 12
    # Rank M_{2,2,2} = 7
    # Total M_{2,5,5} = 18 + 12 + 12 + 7 = 49
    # Total M_{4,5,5} = 98
    # This is barely better than 100.
    
    # I MUST use the rank 11 for M_{2,3,3}.
    # Reference: "The rank of the 2x3x3 matrix multiplication tensor is 11".
    # I will implement a generic function to generate schoolbook and then try to hardcode the 11-rank one if I can find it.
    # Since I don't have the 11-rank coefficients memorized, I will use a slightly higher bound that is easy to construct.
    
    # Alternative: Use M_{2,4,4} rank 13.
    # M_{2,5,5} via 4+1 split.
    # M_{2,4,4} rank 13.
    # M_{2,1,4} rank 8.
    # M_{2,4,1} rank 8.
    # M_{2,1,1} rank 2.
    # Total 13+8+8+2 = 31.
    # Total M_{4,5,5} = 62.
    
    # This is a significant improvement (100 -> 62).
    # I need the rank 13 decomposition for M_{2,4,4}.
    # M_{2,4,4} can be decomposed using two M_{2,2,2} (rank 7) and some cross terms?
    # A (2x4) = [A1 A2] (2x2 each).
    # B (4x4) = [[B11 B12], [B21 B22]] (2x2 each).
    # C = A1 B11 + A1 B12 + A2 B21 + A2 B22.
    # A1 B11: M_{2,2,2} rank 7.
    # A2 B22: M_{2,2,2} rank 7.
    # A1 B12: M_{2,2,2} rank 7.
    # A2 B21: M_{2,2,2} rank 7.
    # Total 28.
    # But we can combine?
    # No, they are independent in the output?
    # C11 (from A1B11) and C12 (from A1B12) are different parts of C.
    # So we just sum the ranks.
    # Rank 28 for M_{2,4,4}.
    # Then M_{2,5,5} rank 28 + 8 + 8 + 2 = 46.
    # Total 92.
    
    # The 13-rank decomposition for M_{2,4,4} is non-trivial.
    # However, 92 is still an improvement.
    
    # Let's go with the 4+1 split and schoolbook for the 2x2 blocks.
    # Rank M_{2,4,4} = 4 * 7 = 28.
    # Rank M_{2,1,4} = 8.
    # Rank M_{2,4,1} = 8.
    # Rank M_{2,1,1} = 2.
    # Total 46.
    # Total 92.

    # Let's implement this.
    
    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def get_m222_strassen():
        return get_m222()

    # Build M_{2,4,4} using 4 M_{2,2,2}
    def get_m244():
        U_blocks = []
        V_blocks = []
        W_blocks = []
        # A (2x4) = [A1 A2]
        # B (4x4) = [[B11 B12], [B21 B22]]
        # C1 = A1 B11 + A2 B21
        # C2 = A1 B12 + A2 B22
        
        # We need to map the sub-problems to the full indices.
        # Sub-problem 1: A1 (i=0..1, j=0..1), B11 (j=0..1, k=0..1), C11 (k=0..1, i=0..1)
        # Sub-problem 2: A1 (i=0..1, j=0..1), B12 (j=0..1, k=2..3), C12 (k=2..3, i=0..1)
        # Sub-problem 3: A2 (i=0..1, j=2..3), B21 (j=2..3, k=0..1), C21 (k=0..1, i=0..1)
        # Sub-problem 4: A2 (i=0..1, j=2..3), B22 (j=2..3, k=2..3), C22 (k=2..3, i=0..1)
        
        U_full = [[0] * 28 for _ in range(8)] # 2*4 = 8
        V_full = [[0] * 28 for _ in range(16)] # 4*4 = 16
        W_full = [[0] * 28 for _ in range(8)] # 4*2 = 8
        
        offset = 0
        for (j_off, k_off, c_k_off) in [(0, 0, 0), (0, 2, 2), (2, 0, 0), (2, 2, 2)]:
            U_sub, V_sub, W_sub = get_m222_strassen()
            for i in range(2):
                for j in range(2):
                    U_full[i * 4 + (j_off + j)][offset + j] = U_sub[i * 2 + j][j] # Wait, U_sub index is i*2+j
                    # Correction: U_sub[i*2+j] is the row.
                    # U_full row is i*4 + j_off + j.
                    # U_sub col is 0..6.
                    # We add to U_full col offset + c.
                    for c in range(7):
                        U_full[i * 4 + j_off + j][offset + c] += U_sub[i * 2 + j][c]
            
            for j in range(2):
                for k in range(2):
                    for c in range(7):
                        V_full[(j_off + j) * 4 + (k_off + k)][offset + c] += V_sub[j * 2 + k][c]
            
            for k in range(2):
                for i in range(2):
                    for c in range(7):
                        W_full[(c_k_off + k) * 2 + i][offset + c] += W_sub[k * 2 + i][c]
            
            offset += 7
            
        return U_full, V_full, W_full

    # Build M_{2,5,5} using M_{2,4,4} and schoolbook for the rest.
    def get_m255():
        U_244, V_244, W_244 = get_m244()
        
        # M_{2,1,4} (A is 2x1, B is 1x4, C is 2x4)
        # Rank 8.
        U_214, V_214, W_214 = get_schoolbook(2, 1, 4)
        
        # M_{2,4,1} (A is 2x4, B is 4x1, C is 2x1)
        # Rank 8.
        U_241, V_241, W_241 = get_schoolbook(2, 4, 1)
        
        # M_{2,1,1} (A is 2x1, B is 1x1, C is 2x1)
        # Rank 2.
        U_211, V_211, W_211 = get_schoolbook(2, 1, 1)
        
        rank_total = 28 + 8 + 8 + 2
        U = [[0] * rank_total for _ in range(10)] # 2*5
        V = [[0] * rank_total for _ in range(25)] # 5*5
        W = [[0] * rank_total for _ in range(10)] # 5*2
        
        # Embed M_{2,4,4} (i=0..1, j=0..3, k=0..3)
        offset = 0
        for i in range(2):
            for j in range(4):
                for c in range(28):
                    U[i * 5 + j][offset + c] += U_244[i * 4 + j][c]
        for j in range(4):
            for k in range(4):
                for c in range(28):
                    V[j * 5 + k][offset + c] += V_244[j * 4 + k][c]
        for k in range(4):
            for i in range(2):
                for c in range(28):
                    W[k * 2 + i][offset + c] += W_244[k * 2 + i][c]
        offset += 28
        
        # Embed M_{2,1,4} (i=0..1, j=4, k=0..3)
        # A row i*5+4. B row 4*5+k. C row k*2+i.
        for i in range(2):
            for c in range(8):
                U[i * 5 + 4][offset + c] += U_214[i * 1 + 0][c] # U_214 is 2x1
        for k in range(4):
            for c in range(8):
                V[4 * 5 + k][offset + c] += V_214[0 * 4 + k][c] # V_214 is 1x4
        for k in range(4):
            for i in range(2):
                for c in range(8):
                    W[k * 2 + i][offset + c] += W_214[k * 2 + i][c]
        offset += 8
        
        # Embed M_{2,4,1} (i=0..1, j=0..3, k=4)
        for i in range(2):
            for j in range(4):
                for c in range(8):
                    U[i * 5 + j][offset + c] += U_241[i * 4 + j][c]
        for j in range(4):
            for c in range(8):
                V[j * 5 + 4][offset + c] += V_241[j * 1 + 0][c] # V_241 is 4x1
        for i in range(2):
            for c in range(8):
                W[4 * 2 + i][offset + c] += W_241[0 * 2 + i][c] # W_241 is 1x2
        offset += 8
        
        # Embed M_{2,1,1} (i=0..1, j=4, k=4)
        for i in range(2):
            for c in range(2):
                U[i * 5 + 4][offset + c] += U_211[i * 1 + 0][c]
        for c in range(2):
            V[4 * 5 + 4][offset + c] += V_211[0 * 1 + 0][c]
        for i in range(2):
            for c in range(2):
                W[4 * 2 + i][offset + c] += W_211[0 * 2 + i][c]
        
        return U, V, W

    # Build M_{4,5,5} by duplicating M_{2,5,5}
    U_255, V_255, W_255 = get_m255()
    rank_255 = 46
    rank_total = 2 * rank_255
    
    U = [[0] * rank_total for _ in range(20)]
    V = [[0] * rank_total for _ in range(25)]
    W = [[0] * rank_total for _ in range(20)]
    
    # First 2 rows of A
    for i in range(2):
        for j in range(5):
            for c in range(rank_255):
                U[i * 5 + j][c] += U_255[i * 5 + j][c]
    for j in range(5):
        for k in range(5):
            for c in range(rank_255):
                V[j * 5 + k][c] += V_255[j * 5 + k][c]
    for k in range(5):
        for i in range(2):
            for c in range(rank_255):
                W[k * 4 + i][c] += W_255[k * 2 + i][c]
                
    # Next 2 rows of A (offset by 2 in i)
    for i in range(2):
        for j in range(5):
            for c in range(rank_255):
                U[(i + 2) * 5 + j][rank_255 + c] += U_255[i * 5 + j][c]
    for j in range(5):
        for k in range(5):
            for c in range(rank_255):
                V[j * 5 + k][rank_255 + c] += V_255[j * 5 + k][c]
    for k in range(5):
        for i in range(2):
            for c in range(rank_255):
                W[k * 4 + (i + 2)][rank_255 + c] += W_255[k * 2 + i][c]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
