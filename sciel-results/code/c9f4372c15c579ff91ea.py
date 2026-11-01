import time

DIMENSIONS = [2, 2, 14]

def construct():
    """
    Rank-49 decomposition for 2x2 * 2x14 matrix multiplication.
    Uses 7 independent Strassen blocks.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # Indices: A00=0, A01=1, A10=2, A11=3
    U_forms = [
        [1, 0, 0, 1],   # M1 = A00 + A11
        [0, 0, 1, 1],   # M2 = A10 + A11
        [1, 0, 0, 0],   # M3 = A00
        [0, 0, 0, 1],   # M4 = A11
        [1, 1, 0, 0],   # M5 = A00 + A01
        [-1, 0, 1, 0],  # M6 = A10 - A00
        [0, 1, 0, -1]   # M7 = A01 - A11
    ]
    
    for r in range(num_blocks):
        # B block indices
        # B is 2x14. Row 0 is 0..13, Row 1 is 14..27? 
        # Wait, V shape is (m*p, rank) = (2*14, rank) = (28, rank).
        # Indexing for V:
        # The problem says "output factor follows the official trace convention W[k*n+i,r]".
        # Usually input matrices are flattened row-major or column-major.
        # Standard convention in these tasks:
        # U: (n*m, rank). A is n x m. Index i*n + j? Or j*m + i?
        # Let's check the seed code's assumption.
        # Seed code:
        # U[i][c] where i in 0..3.
        # V[idx_e][c] where idx_e = 2*r.
        # If B is 2x14, and flattened row-major:
        # Row 0: B[0,0]...B[0,13] -> indices 0..13
        # Row 1: B[1,0]...B[1,13] -> indices 14..27
        # So B[j, k] is at index j*14 + k.
        # In seed code:
        # idx_e = 2*r. This corresponds to B[0, 2r].
        # idx_f = 2*r + 1. This corresponds to B[0, 2r+1].
        # idx_g = 14 + 2*r. This corresponds to B[1, 2r].
        # idx_h = 14 + 2*r + 1. This corresponds to B[1, 2r+1].
        # This matches row-major flattening of B (2x14).
        
        # W shape is (p*n, rank) = (14*2, rank) = (28, rank).
        # Convention: W[k*n + i, r].
        # C is 2x14. C[i, k].
        # Index in W: k*n + i = 2*k + i.
        # Seed code:
        # w_idx_c00 = 4*r. 
        # C[0, 2r] -> k=2r, i=0. Index = 2*(2r) + 0 = 4r. Matches.
        # w_idx_c10 = 4*r + 1.
        # C[1, 2r] -> k=2r, i=1. Index = 2*(2r) + 1 = 4r+1. Matches.
        # w_idx_c01 = 4*r + 2.
        # C[0, 2r+1] -> k=2r+1, i=0. Index = 2*(2r+1) + 0 = 4r+2. Matches.
        # w_idx_c11 = 4*r + 3.
        # C[1, 2r+1] -> k=2r+1, i=1. Index = 2*(2r+1) + 1 = 4r+3. Matches.
        
        idx_e = 2 * r       # B[0, 2r]
        idx_f = 2 * r + 1   # B[0, 2r+1]
        idx_g = 14 + 2 * r       # B[1, 2r]
        idx_h = 14 + 2 * r + 1   # B[1, 2r+1]
        
        w_idx_c00 = 4 * r       # C[0, 2r]
        w_idx_c10 = 4 * r + 1   # C[1, 2r]
        w_idx_c01 = 4 * r + 2   # C[0, 2r+1]
        w_idx_c11 = 4 * r + 3   # C[1, 2r+1]
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][c] = float(U_forms[s][i])
                
            # Fill V
            if s == 0:
                V[idx_e][c] = 1.0
                V[idx_h][c] = 1.0
            elif s == 1:
                V[idx_e][c] = 1.0
            elif s == 2:
                V[idx_f][c] = 1.0
                V[idx_h][c] = -1.0
            elif s == 3:
                V[idx_e][c] = -1.0
                V[idx_g][c] = 1.0
            elif s == 4:
                V[idx_h][c] = 1.0
            elif s == 5:
                V[idx_e][c] = 1.0
                V[idx_f][c] = 1.0
            elif s == 6:
                V[idx_g][c] = 1.0
                V[idx_h][c] = 1.0
                
            # Fill W
            if s == 0:
                W[w_idx_c00][c] = 1.0
                W[w_idx_c11][c] = 1.0
            elif s == 1:
                W[w_idx_c10][c] = 1.0
                W[w_idx_c11][c] = -1.0
            elif s == 2:
                W[w_idx_c01][c] = 1.0
                W[w_idx_c11][c] = 1.0
            elif s == 3:
                W[w_idx_c00][c] = 1.0
                W[w_idx_c10][c] = 1.0
            elif s == 4:
                W[w_idx_c00][c] = -1.0
                W[w_idx_c01][c] = 1.0
            elif s == 5:
                W[w_idx_c11][c] = 1.0
            elif s == 6:
                W[w_idx_c00][c] = 1.0
                
    return {"U": U, "V": V, "W": W}
