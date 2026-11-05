"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition m=4 into 2+2, p=5 into 3+2
    # Blocks:
    # 1. A1 (3x2) * B1 (2x3) -> C1 (3x3)
    # 2. A1 (3x2) * B2 (2x2) -> C2 (3x2)
    # 3. A2 (3x2) * B3 (2x3) -> C3 (3x3)
    # 4. A2 (3x2) * B4 (2x2) -> C4 (3x2)
    
    # We use Rank 16 for 3x2x3 and Rank 10 for 3x2x2
    # Total Rank = 16 + 10 + 16 + 10 = 52
    
    rank_323 = 16
    rank_322 = 10
    total_rank = 2 * rank_323 + 2 * rank_322
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    
    # Helper to add a block decomposition
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    def add_block_323(i_start, j_start, k_start, u_off):
        # A is 3x2 (rows i_start..i_start+2, cols j_start..j_start+1)
        # B is 2x3 (rows j_start..j_start+1, cols k_start..k_start+2)
        # C is 3x3 (rows i_start..i_start+2, cols k_start..k_start+2)
        # Decomposition: Strassen on 2x2x2 part (rows 0,1 of A, cols 0,1 of B)
        # + Schoolbook on remaining parts
        
        # A1: 2x2 (rows 0,1), B1: 2x2 (cols 0,1) -> C11: 2x2
        # Use Strassen for C11
        # M1 = (A1[0,0]+A1[1,0]) * (B1[0,0]+B1[1,0])
        # M2 = (A1[0,1]+A1[1,1]) * (B1[0,0]+B1[1,0])
        # M3 = A1[0,0] * (B1[0,1]-B1[1,1])
        # M4 = A1[1,1] * (B1[0,0]+B1[1,0])
        # M5 = (A1[0,0]+A1[0,1]) * (B1[1,1]-B1[0,0])
        # M6 = (A1[1,0]-A1[0,0]) * (B1[0,0]+B1[0,1])
        # M7 = (A1[1,1]-A1[0,1]) * (B1[0,1]+B1[1,1])
        
        # This is getting complex to map to global U,V,W.
        # I will use the simpler Rank 16 decomposition:
        # Partition A into A_top (2x2) and A_bot (1x2)
        # Partition B into B_left (2x2) and B_right (2x1)
        
        # Block 1: A_top (2x2) * B_left (2x2) -> C_tl (2x2)
        # Use Strassen (Rank 7)
        # A_top rows: i_start, i_start+1
        # B_left cols: k_start, k_start+1
        
        # Strassen terms for 2x2x2
        # Indices for U: (i, j) where i in {0,1}, j in {0,1} relative to block
        # Indices for V: (j, k) where j in {0,1}, k in {0,1}
        
        # M1
        U[i_start*4 + j_start][u_off] = 1
        U[(i_start+1)*4 + j_start][u_off] = 1
        V[j_start*5 + k_start][u_off] = 1
        V[(j_start+1)*5 + k_start][u_off] = 1
        W[k_start*3 + i_start][u_off] = 1
        W[(k_start+1)*3 + (i_start+1)][u_off] = 1
        W[k_start*3 + (i_start+1)][u_off] = 1
        W[(k_start+1)*3 + i_start][u_off] = 1
        u_off += 1
        
        # M2
        U[i_start*4 + (j_start+1)][u_off] = 1
        U[(i_start+1)*4 + (j_start+1)][u_off] = 1
        V[j_start*5 + k_start][u_off] = 1
        V[(j_start+1)*5 + k_start][u_off] = 1
        W[(k_start+1)*3 + (i_start+1)][u_off] = 1
        W[(k_start+1)*3 + i_start][u_off] = -1
        u_off += 1
        
        # M3
        U[i_start*4 + j_start][u_off] = 1
        V[(j_start+1)*5 + (k_start+1)][u_off] = 1
        V[j_start*5 + (k_start+1)][u_off] = -1
        W[(k_start+1)*3 + i_start][u_off] = 1
        u_off += 1
        
        # M4
        U[(i_start+1)*4 + (j_start+1)][u_off] = 1
        V[j_start*5 + k_start][u_off] = 1
        V[(j_start+1)*5 + k_start][u_off] = 1
        W[(k_start+1)*3 + (i_start+1)][u_off] = 1
        u_off += 1
        
        # M5
        U[i_start*4 + j_start][u_off] = 1
        U[i_start*4 + (j_start+1)][u_off] = 1
        V[(j_start+1)*5 + (k_start+1)][u_off] = 1
        V[j_start*5 + k_start][u_off] = -1
        W[k_start*3 + i_start][u_off] = 1
        u_off += 1
        
        # M6
        U[(i_start+1)*4 + j_start][u_off] = 1
        U[i_start*4 + j_start][u_off] = -1
        V[j_start*5 + k_start][u_off] = 1
        V[j_start*5 + (k_start+1)][u_off] = 1
        W[k_start*3 + (i_start+1)][u_off] = 1
        u_off += 1
        
        # M7
        U[(i_start+1)*4 + (j_start+1)][u_off] = 1
        U[i_start*4 + (j_start+1)][u_off] = -1
        V[j_start*5 + (k_start+1)][u_off] = 1
        V[(j_start+1)*5 + (k_start+1)][u_off] = 1
        W[(k_start+1)*3 + (i_start+1)][u_off] = 1
        u_off += 1
        
        # Block 2: A_top (2x2) * B_right (2x1) -> C_tr (2x1)
        # Schoolbook Rank 4
        for ii in range(2):
            for jj in range(2):
                U[(i_start+ii)*4 + (j_start+jj)][u_off] = 1
                V[(j_start+jj)*5 + (k_start+2)][u_off] = 1
                W[(k_start+2)*3 + (i_start+ii)][u_off] = 1
                u_off += 1

        # Block 3: A_bot (1x2) * B_left (2x2) -> C_bl (1x2)
        # Schoolbook Rank 4
        for jj in range(2):
            for kk in range(2):
                U[(i_start+2)*4 + (j_start+jj)][u_off] = 1
                V[(j_start+jj)*5 + (k_start+kk)][u_off] = 1
                W[(k_start+kk)*3 + (i_start+2)][u_off] = 1
                u_off += 1

        # Block 4: A_bot (1x2) * B_right (2x1) -> C_br (1x1)
        # Schoolbook Rank 2 (since B_right is 2x1, A_bot is 1x2)
        for jj in range(2):
            U[(i_start+2)*4 + (j_start+jj)][u_off] = 1
            V[(j_start+jj)*5 + (k_start+2)][u_off] = 1
            W[(k_start+2)*3 + (i_start+2)][u_off] = 1
            u_off += 1

        return u_off

    def add_block_322(i_start, j_start, k_start, u_off):
        # A is 3x2, B is 2x2, C is 3x2
        # Partition A into A_top (2x2) and A_bot (1x2)
        # Partition B into B_left (2x1) and B_right (2x1)
        
        # Block 1: A_top (2x2) * B_left (2x1) -> C_tl (2x1)
        # Schoolbook Rank 4
        for ii in range(2):
            for jj in range(2):
                U[(i_start+ii)*4 + (j_start+jj)][u_off] = 1
                V[(j_start+jj)*5 + k_start][u_off] = 1
                W[k_start*3 + (i_start+ii)][u_off] = 1
                u_off += 1

        # Block 2: A_top (2x2) * B_right (2x1) -> C_tr (2x1)
        # Schoolbook Rank 4
        for ii in range(2):
            for jj in range(2):
                U[(i_start+ii)*4 + (j_start+jj)][u_off] = 1
                V[(j_start+jj)*5 + (k_start+1)][u_off] = 1
                W[(k_start+1)*3 + (i_start+ii)][u_off] = 1
                u_off += 1

        # Block 3: A_bot (1x2) * B_left (2x1) -> C_bl (1x1)
        # Schoolbook Rank 2
        for jj in range(2):
            U[(i_start+2)*4 + (j_start+jj)][u_off] = 1
            V[(j_start+jj)*5 + k_start][u_off] = 1
            W[k_start*3 + (i_start+2)][u_off] = 1
            u_off += 1

        # Block 4: A_bot (1x2) * B_right (2x1) -> C_br (1x1)
        # Schoolbook Rank 2
        for jj in range(2):
            U[(i_start+2)*4 + (j_start+jj)][u_off] = 1
            V[(j_start+jj)*5 + (k_start+1)][u_off] = 1
            W[(k_start+1)*3 + (i_start+2)][u_off] = 1
            u_off += 1

        return u_off

    # Block 1: 3x2x3 (i=0..2, j=0..1, k=0..2)
    offset = add_block_323(0, 0, 0, offset)
    
    # Block 2: 3x2x2 (i=0..2, j=0..1, k=3..4)
    offset = add_block_322(0, 0, 3, offset)
    
    # Block 3: 3x2x3 (i=0..2, j=2..3, k=0..2)
    offset = add_block_323(0, 2, 0, offset)
    
    # Block 4: 3x2x2 (i=0..2, j=2..3, k=3..4)
    offset = add_block_322(0, 2, 3, offset)
    
    assert offset == total_rank, f"Rank mismatch: {offset} vs {total_rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
