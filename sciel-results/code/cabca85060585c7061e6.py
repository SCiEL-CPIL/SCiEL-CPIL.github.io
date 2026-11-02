"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose A (n x m) and B (m x p) into blocks.
    # m=5 is split into 2 and 3.
    # p=6 is split into 2, 2, 2 for the first part and 3, 3 for the second.
    
    # A is n x m. Split columns into 2 and 3.
    # A1 is n x 2, A2 is n x 3.
    # B is m x p. Split rows into 2 and 3.
    # B1 is 2 x p, B2 is 3 x p.
    # C = A1 * B1 + A2 * B2.
    
    # A1 * B1: n x 2 * 2 x p.
    # p=6 is split into three 2s.
    # B1 is 2 x 6. Split into B1a, B1b, B1c each 2 x 2.
    # A1 * B1a, A1 * B1b, A1 * B1c.
    # Each is n x 2 * 2 x 2. Since n=2, this is 2x2 * 2x2.
    # Use Strassen for 2x2 * 2x2 (rank 7).
    
    # A2 * B2: n x 3 * 3 x p.
    # p=6 is split into two 3s.
    # B2 is 3 x 6. Split into B2a, B2b each 3 x 3.
    # A2 * B2a, A2 * B2b.
    # Each is n x 3 * 3 x 3. Since n=2, this is 2x3 * 3x3.
    # Use schoolbook for 2x3 * 3x3 (rank 18).
    
    rank_strassen = 7
    rank_schoolbook_233 = 2 * 3 * 3 # 18
    
    total_rank = 3 * rank_strassen + 2 * rank_schoolbook_233
    # 3 * 7 + 2 * 18 = 21 + 36 = 57.
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2 Strassen decomposition
    def add_strassen_22(u_offset, v_offset, w_offset, start_col):
        # u_offset: (i, j) index in A1 (2x2)
        # v_offset: (j, k) index in B1x (2x2)
        # w_offset: (k, i) index in C (2x2)
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M4
        # C21 = M2 + M5
        # C22 = M1 - M3 + M6 + M7
        
        # A indices: (0,0), (0,1), (1,0), (1,1)
        # B indices: (0,0), (0,1), (1,0), (1,1)
        # C indices: (0,0), (0,1), (1,0), (1,1)
        
        # M1
        c = start_col
        U[u_offset[0]*2 + u_offset[1]][c] += 1 # A11
        U[1*2 + 1][c] += 1 # A22
        V[v_offset[0]*2 + v_offset[1]][c] += 1 # B11
        V[1*2 + 1][c] += 1 # B22
        W[w_offset[0]*2 + w_offset[1]][c] += 1 # C11
        W[0*2 + 1][c] += 1 # C12
        W[1*2 + 0][c] += 1 # C21
        W[1*2 + 1][c] += 1 # C22
        # Wait, W is p*n. Index is k*n + i.
        # For 2x2 block, i is row of C, k is col of C.
        # C11 is i=0, k=0. Index 0*2 + 0 = 0.
        # C12 is i=0, k=1. Index 1*2 + 0 = 2.
        # C21 is i=1, k=0. Index 0*2 + 1 = 1.
        # C22 is i=1, k=1. Index 1*2 + 1 = 3.
        
        # Correct W indices for 2x2 block starting at w_offset (k_start, i_start)
        # w_offset is (k, i) in the larger matrix?
        # Let's define w_offset as (k0, i0).
        # C11 -> k=k0, i=i0. Index k0*n + i0.
        # C12 -> k=k0+1, i=i0. Index (k0+1)*n + i0.
        # C21 -> k=k0, i=i0+1. Index k0*n + (i0+1).
        # C22 -> k=k0+1, i=i0+1. Index (k0+1)*n + (i0+1).
        
        # My previous W assignment was wrong.
        # Let's rewrite add_strassen_22 carefully.
        
        # A is 2x2. Indices (i, j). i in 0..1, j in 0..1.
        # B is 2x2. Indices (j, k). j in 0..1, k in 0..1.
        # C is 2x2. Indices (i, k). i in 0..1, k in 0..1.
        
        # M1 = (A11+A22)(B11+B22)
        U[u_offset[0]*2 + 0][c] += 1 # A11
        U[u_offset[0]*2 + 1][c] += 1 # A22 (Wait, A22 is i=1, j=1. Index 1*2+1=3. A11 is 0*2+0=0. A21 is 1*2+0=2. A12 is 0*2+1=1.)
        # Let's use explicit indices.
        # A11: i=0, j=0.
        # A22: i=1, j=1.
        # B11: j=0, k=0.
        # B22: j=1, k=1.
        
        # I will implement this more simply by generating the terms.
        
        return start_col + 7 # Placeholder
    
    # This is getting complicated to implement inline.
    # I will use a simpler approach: just use schoolbook for everything first to ensure correctness, then optimize.
    # But schoolbook is 60.
    # I will implement the 57 rank decomposition by explicitly listing the terms.
    
    # Let's restart the implementation with a clear structure.
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 * B1 (3 blocks of 2x2 * 2x2)
    # A1 is rows 0-1, cols 0-1 of A.
    # B1 is rows 0-1, cols 0-5 of B.
    # Split B1 into B1a (cols 0-1), B1b (cols 2-3), B1c (cols 4-5).
    
    for block_idx in range(3):
        # B1x is 2x2.
        # B1a: B rows 0-1, cols 0-1.
        # B1b: B rows 0-1, cols 2-3.
        # B1c: B rows 0-1, cols 4-5.
        k_start = block_idx * 2
        
        # Strassen terms for 2x2 * 2x2
        # M1 = (A11+A22)(B11+B22)
        c = col
        U[0][c] += 1 # A11
        U[3][c] += 1 # A22
        V[0*2 + k_start][c] += 1 # B11
        V[1*2 + k_start + 1][c] += 1 # B22
        W[k_start*2 + 0][c] += 1 # C11 (k=0, i=0)
        W[(k_start+1)*2 + 0][c] += 1 # C12 (k=1, i=0)
        W[k_start*2 + 1][c] += 1 # C21 (k=0, i=1)
        W[(k_start+1)*2 + 1][c] += 1 # C22 (k=1, i=1)
        col += 1
        
        # M2 = (A21+A22)B11
        c = col
        U[2][c] += 1 # A21
        U[3][c] += 1 # A22
        V[0*2 + k_start][c] += 1 # B11
        W[k_start*2 + 1][c] += 1 # C21
        W[(k_start+1)*2 + 1][c] += 1 # C22
        col += 1
        
        # M3 = A11(B21-B22)
        c = col
        U[0][c] += 1 # A11
        V[1*2 + k_start][c] += 1 # B21
        V[1*2 + k_start + 1][c] -= 1 # B22
        W[k_start*2 + 0][c] += 1 # C11
        W[(k_start+1)*2 + 0][c] += 1 # C12
        col += 1
        
        # M4 = A22(B11-B21)
        c = col
        U[3][c] += 1 # A22
        V[0*2 + k_start][c] += 1 # B11
        V[1*2 + k_start][c] -= 1 # B21
        W[k_start*2 + 0][c] -= 1 # C11
        W[k_start*2 + 1][c] += 1 # C21
        col += 1
        
        # M5 = (A11+A12)B22
        c = col
        U[0][c] += 1 # A11
        U[1][c] += 1 # A12
        V[1*2 + k_start + 1][c] += 1 # B22
        W[k_start*2 + 0][c] += 1 # C11
        W[(k_start+1)*2 + 1][c] += 1 # C22
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        c = col
        U[2][c] += 1 # A21
        U[0][c] -= 1 # A11
        V[0*2 + k_start][c] += 1 # B11
        V[0*2 + k_start + 1][c] += 1 # B12
        W[(k_start+1)*2 + 1][c] += 1 # C22
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        c = col
        U[1][c] += 1 # A12
        U[3][c] -= 1 # A22
        V[1*2 + k_start][c] += 1 # B21
        V[1*2 + k_start + 1][c] += 1 # B22
        W[k_start*2 + 0][c] += 1 # C11
        W[(k_start+1)*2 + 0][c] += 1 # C12
        W[(k_start+1)*2 + 1][c] += 1 # C22
        col += 1
        
    # Part 2: A2 * B2 (2 blocks of 2x3 * 3x3)
    # A2 is rows 0-1, cols 2-4 of A.
    # B2 is rows 2-4, cols 0-5 of B.
    # Split B2 into B2a (cols 0-2), B2b (cols 3-5).
    
    for block_idx in range(2):
        k_start = block_idx * 3
        
        # Schoolbook for 2x3 * 3x3
        # A2 is 2x3. Indices i in 0..1, j in 0..2.
        # B2x is 3x3. Indices j in 0..2, k in 0..2.
        # C is 2x3. Indices i in 0..1, k in 0..2.
        
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    c = col
                    U[i*5 + (2+j)][c] += 1 # A2[i,j] -> A[i, 2+j]
                    V[(2+j)*6 + (k_start+k)][c] += 1 # B2[j,k] -> B[2+j, k_start+k]
                    W[(k_start+k)*2 + i][c] += 1 # C[i,k] -> C[k_start+k, i]
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
