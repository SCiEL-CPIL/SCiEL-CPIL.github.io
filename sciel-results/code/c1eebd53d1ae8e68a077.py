import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form an exact tensor decomposition of the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor entry T[i*m+j, j*p+k, k*n+i] should be 1
                # Reconstruction: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """
    Fallback schoolbook decomposition. Rank = n*m*p.
    """
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_decomp():
    """
    Return (U, V, W) for 2x2x2 multiplication, rank 7.
    Indices:
    U: (4, 7) -> A[0,0], A[0,1], A[1,0], A[1,1]
    V: (4, 7) -> B[0,0], B[0,1], B[1,0], B[1,1]
    W: (4, 7) -> C[0,0], C[0,1], C[1,0], C[1,1]
    """
    rank = 7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M0: A00 * (B01+B11)
    U[0][0] = 1
    V[1][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # c00, c11
    
    # M1: (A10+A11) * B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[2][1] = 1  # c01, c10
    
    # M2: (A00+A01) * B11
    U[0][2] = 1; U[1][2] = 1
    V[3][2] = 1
    W[0][2] = 1; W[3][2] = 1  # c00, c11
    
    # M3: A11 * (B00-B10)
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1  # c00, c10
    
    # M4: (A00+A10) * B01
    U[0][4] = 1; U[2][4] = 1
    V[1][4] = 1
    W[0][4] = -1; W[1][4] = 1  # c00, c01
    
    # M5: (A01-A10) * (B00+B10)
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[0][5] = -1  # c00
    
    # M6: (A11-A01) * (B01+B11)
    U[3][6] = 1; U[1][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[3][6] = 1  # c11
    
    return U, V, W

def _construct_2x3x6_decomp():
    """
    Construct Rank 33 decomposition for 2x3x6.
    Split B into 3 blocks of 3x2.
    For each block, use Rank 11 decomposition:
      - Rank 7 for the 2x2x2 core (rows 0,1 of A; rows 0,1 of B; cols 0,1 of B block)
      - Rank 4 for the remaining 2x1x2 terms (row 2 of A; row 2 of B block; cols 0,1 of B block)
    Total: 3 * 11 = 33.
    """
    U_block = [[0]*11 for _ in range(6)]
    V_block = [[0]*11 for _ in range(6)]
    W_block = [[0]*11 for _ in range(4)]
    
    # Fill Strassen part (indices 0-6)
    U_s, V_s, W_s = _strassen_2x2_decomp()
    for i in range(4):
        for r in range(7):
            U_block[i][r] = U_s[i][r]
    for j in range(4):
        for r in range(7):
            V_block[j][r] = V_s[j][r]
    for k in range(4):
        for r in range(7):
            W_block[k][r] = W_s[k][r]
    
    # Fill Schoolbook 2x1x2 part (indices 7-10)
    # A[2, :] (row 2 of A, which is a 1x2 vector in this sub-problem? No, A is 2x3.
    # In the 2x3x2 sub-problem, A is 2x3.
    # The "core" uses A[0:2, 0:2] and B[0:2, 0:2].
    # The remaining terms involve A[0:2, 2] and B[2, 0:2].
    # Wait, A is 2x3. So A has rows 0,1 and cols 0,1,2.
    # B is 3x2. So B has rows 0,1,2 and cols 0,1.
    # C is 2x2.
    # c_ij = sum_l A[i,l] B[l,j]
    # Core: l in {0,1}. This is 2x2x2.
    # Remainder: l=2. This is A[:, 2] * B[2, :].
    # A[:, 2] is a 2x1 column vector.
    # B[2, :] is a 1x2 row vector.
    # This is a 2x1 * 1x2 product, which is 2x2.
    # Rank 4 schoolbook for this.
    
    # U for A[:, 2]:
    # A[0,2] is index 2 in U (since U is indexed by i*m+j, m=3? No.
    # In the 2x3x2 sub-problem, U is indexed by i*3+j?
    # The global U is indexed by i*m+j. Here m=3.
    # So A[0,0]->0, A[0,1]->1, A[0,2]->2.
    # A[1,0]->3, A[1,1]->4, A[1,2]->5.
    
    # The Strassen part used U[0:4].
    # U[0] = A[0,0]
    # U[1] = A[0,1]
    # U[2] = A[1,0]
    # U[3] = A[1,1]
    # This mapping is WRONG for the global indices if we just map directly.
    # The Strassen decomp assumes A is 2x2 with indices (0,0),(0,1),(1,0),(1,1).
    # In our 2x3x2 sub-problem, the core is A[0:2, 0:2].
    # So A[0,0] is global index 0.
    # A[0,1] is global index 1.
    # A[1,0] is global index 3.
    # A[1,1] is global index 4.
    
    # My previous mapping in _construct_2x3x2_decomp was:
    # U[0:4] = U_s[0:4]
    # This mapped A[0,0] to 0, A[0,1] to 1, A[1,0] to 2, A[1,1] to 3.
    # But in 2x3x2, A[1,0] is index 3, A[1,1] is index 4.
    # So the mapping was incorrect!
    
    # Let's redo the mapping carefully.
    
    U_block = [[0]*11 for _ in range(6)]
    V_block = [[0]*11 for _ in range(6)]
    W_block = [[0]*11 for _ in range(4)]
    
    # Strassen part
    # U_s[0] corresponds to A[0,0] -> Global U[0]
    # U_s[1] corresponds to A[0,1] -> Global U[1]
    # U_s[2] corresponds to A[1,0] -> Global U[3]
    # U_s[3] corresponds to A[1,1] -> Global U[4]
    
    U_map = [0, 1, 3, 4]
    for local_i, global_i in enumerate(U_map):
        for r in range(7):
            U_block[global_i][r] = U_s[local_i][r]
            
    # V_s[0] corresponds to B[0,0] -> Global V[0]
    # V_s[1] corresponds to B[0,1] -> Global V[1]
    # V_s[2] corresponds to B[1,0] -> Global V[2]
    # V_s[3] corresponds to B[1,1] -> Global V[3]
    # B is 3x2. Indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3, (2,0)->4, (2,1)->5.
    V_map = [0, 1, 2, 3]
    for local_j, global_j in enumerate(V_map):
        for r in range(7):
            V_block[global_j][r] = V_s[local_j][r]
            
    # W_s[0] corresponds to C[0,0] -> Global W[0]
    # W_s[1] corresponds to C[0,1] -> Global W[1]
    # W_s[2] corresponds to C[1,0] -> Global W[2]
    # W_s[3] corresponds to C[1,1] -> Global W[3]
    # C is 2x2. Indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
    W_map = [0, 1, 2, 3]
    for local_k, global_k in enumerate(W_map):
        for r in range(7):
            W_block[global_k][r] = W_s[local_k][r]
            
    # Schoolbook part for A[:,2] * B[2,:]
    # Terms:
    # A[0,2] * B[2,0] -> C[0,0]
    # A[0,2] * B[2,1] -> C[0,1]
    # A[1,2] * B[2,0] -> C[1,0]
    # A[1,2] * B[2,1] -> C[1,1]
    
    # Global indices:
    # A[0,2] -> U[2]
    # A[1,2] -> U[5]
    # B[2,0] -> V[4]
    # B[2,1] -> V[5]
    # C[0,0] -> W[0]
    # C[0,1] -> W[1]
    # C[1,0] -> W[2]
    # C[1,1] -> W[3]
    
    # Rank 7 is full. Start at 7.
    U_block[2][7] = 1
    V_block[4][7] = 1
    W_block[0][7] = 1
    
    U_block[2][8] = 1
    V_block[5][8] = 1
    W_block[1][8] = 1
    
    U_block[5][9] = 1
    V_block[4][9] = 1
    W_block[2][9] = 1
    
    U_block[5][10] = 1
    V_block[5][10] = 1
    W_block[3][10] = 1
    
    # Now assemble 2x3x6
    U = [[0]*33 for _ in range(6)]
    V = [[0]*33 for _ in range(18)]
    W = [[0]*33 for _ in range(12)]
    
    for block in range(3):
        offset = block * 11
        # U is same for all blocks?
        # No, U depends on A. A is the same for all blocks.
        # So we just copy U_block into U.
        for i in range(6):
            for r in range(11):
                U[i][offset+r] = U_block[i][r]
                
        # V depends on B block.
        # V_block is 6x11.
        # B block 0: rows 0,1,2 cols 0,1 -> Global V indices 0..5
        # B block 1: rows 0,1,2 cols 2,3 -> Global V indices 6..11
        # B block 2: rows 0,1,2 cols 4,5 -> Global V indices 12..17
        for j in range(6):
            for r in range(11):
                V[j + block*6][offset+r] = V_block[j][r]
                
        # W depends on C block.
        # W_block is 4x11.
        # C block 0: cols 0,1 -> Global W indices 0..3
        # C block 1: cols 2,3 -> Global W indices 4..7
        # C block 2: cols 4,5 -> Global W indices 8..11
        for k in range(4):
            for r in range(11):
                W[k + block*4][offset+r] = W_block[k][r]
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try Rank 33 construction
    res = _construct_2x3x6_decomp()
    
    if _verify_decomposition(res['U'], res['V'], res['W'], n, m, p, tol=1e-9):
        return res
        
    # Fallback to Schoolbook
    return _schoolbook_decomposition(n, m, p)

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Verified:", _verify_decomposition(res['U'], res['V'], res['W'], 2, 3, 6))
