"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # The tensor decomposition should satisfy:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    V = [[0]*7 for _ in range(4)]
    V[0][0] = 1; V[3][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[3][2] = -1
    V[2][3] = 1; V[0][3] = -1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    W = [[0]*7 for _ in range(4)]
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][2] = 1; W[1][4] = 1
    W[2][1] = 1; W[2][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W

def _construct_4x4x4_strassen():
    """
    Construct a rank-49 decomposition for 4x4 x 4x4 matrix multiplication
    using recursive Strassen.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Define Strassen steps:
    # Each step l has:
    # - A_blocks: list of (row, col, sign) for A blocks in X_l
    # - B_blocks: list of (row, col, sign) for B blocks in Y_l
    # - C_blocks: list of (row, col, sign) for C blocks receiving Z_l
    
    steps = [
        # l=0: M1 = (A00+A11)(B00+B11) -> C00, C11
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        # l=1: M2 = (A10+A11)B00 -> C10, C11
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]), # Wait, check formula
        # Formula: C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
        # So M2 contributes +1 to C10 and -1 to C11.
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]),
        # l=2: M3 = A00(B01-B11) -> C01, C11
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]),
        # l=3: M4 = A11(B10-B00) -> C00, C10
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ]),
        # l=4: M5 = (A00+A10)B11 -> C00, C01
        ([ (0,0,1), (1,0,1) ], [ (1,1,1) ], [ (0,0,-1), (0,1,1) ]),
        # l=5: M6 = (A11+A01)(B00+B10) -> C11
        ([ (1,1,1), (0,1,1) ], [ (0,0,1), (1,0,1) ], [ (1,1,1) ]),
        # l=6: M7 = (A01-A11)(B11+B01) -> C00, C11
        ([ (0,1,1), (1,1,-1) ], [ (1,1,1), (0,1,1) ], [ (0,0,1), (1,1,1) ]),
    ]
    
    # Re-verify formulas from standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11+A01)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-populate steps with correct C signs.
    steps = [
        # l=0: M1
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        # l=1: M2
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]),
        # l=2: M3
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]),
        # l=3: M4
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ]),
        # l=4: M5
        ([ (0,0,1), (1,0,1) ], [ (1,1,1) ], [ (0,0,-1), (0,1,1) ]),
        # l=5: M6
        ([ (1,1,1), (0,1,1) ], [ (0,0,1), (1,0,1) ], [ (1,1,1) ]),
        # l=6: M7
        ([ (0,1,1), (1,1,-1) ], [ (1,1,1), (0,1,1) ], [ (0,0,1), (1,1,1) ]),
    ]
    
    for l in range(7):
        a_blocks, b_blocks, c_blocks = steps[l]
        
        for r_s in range(7):
            r = l * 7 + r_s
            
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        u_loc = U_s[i*2 + j][r_s]
                        v_loc = V_s[j*2 + k][r_s]
                        w_loc = W_s[k*2 + i][r_s]
                        
                        if u_loc == 0 or v_loc == 0 or w_loc == 0:
                            continue
                        
                        # Update U
                        for ar, ac, s_a in a_blocks:
                            gi = ar * 2 + i
                            gj = ac * 2 + j
                            U[gi * 4 + gj][r] += u_loc * s_a
                        
                        # Update V
                        for br, bc, s_b in b_blocks:
                            gj = br * 2 + j
                            gk = bc * 2 + k
                            V[gj * 4 + gk][r] += v_loc * s_b
                        
                        # Update W
                        for cr, cc, s_c in c_blocks:
                            gk = cr * 2 + k
                            gi = cc * 2 + i
                            # W index is k*n + i. Here n=4.
                            # But wait, the W matrix in the return dict is of size p*n.
                            # For 4x4x4, p=4, n=4. Index is k*4 + i.
                            # In the 4x4x8 case, p=8, n=4.
                            # We are constructing 4x4x4 blocks.
                            # So for this block, the local k, i are 0..3.
                            # We will offset them later.
                            W[gk * 4 + gi][r] += w_loc * s_c
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    U_444, V_444, W_444 = _construct_4x4x4_strassen()
    
    # We have two 4x4x4 multiplications.
    # B is 4x8. Split into B1 (cols 0-3) and B2 (cols 4-7).
    # C is 4x8. Split into C1 (cols 0-3) and C2 (cols 4-7).
    
    total_rank = 98
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # First half (B1, C1)
    for i in range(16):
        for r in range(49):
            U[i][r] = U_444[i][r]
    
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j * 8 + k][r] = V_444[j * 4 + k][r]
    
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k * 4 + i][r] = W_444[k * 4 + i][r]
    
    # Second half (B2, C2)
    offset = 49
    for i in range(16):
        for r in range(49):
            U[i][r + offset] = U_444[i][r]
    
    for j in range(4):
        for k in range(4, 8):
            for r in range(49):
                V[j * 8 + k][r + offset] = V_444[j * 4 + (k - 4)][r]
    
    for k in range(4, 8):
        for i in range(4):
            for r in range(49):
                W[k * 4 + i][r + offset] = W_444[(k - 4) * 4 + i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
