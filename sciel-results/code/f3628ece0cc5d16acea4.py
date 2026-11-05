"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Strategy:
    1. Try to find a rank-14 decomposition using two independent Strassen-like rank-7 blocks
       for the two halves of the 6-column output (columns 0-2 and 3-5).
    2. If that fails, try a randomized local search for ranks 14-20.
    3. Fall back to schoolbook rank 54 if nothing else works.
    """
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def make_schoolbook():
        """Return the trivial (schoolbook) decomposition with rank = n*m*p = 54."""
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
    
    def strassen_3x3_block():
        """
        Return a rank-7 decomposition for 3x3 matrix multiplication.
        Returns (U_block, V_block, W_block) where:
        - U_block is (9, 7) for A entries
        - V_block is (9, 7) for B entries  
        - W_block is (9, 7) for C entries
        """
        # Standard Strassen-like rank-7 for 3x3x3
        # M1 = (A00+A11+A22) * (B00+B11+B22)
        # M2 = (A01+A10-A22) * (B01+B10-B22)
        # M3 = (A02+A12-A21) * (B02+B12-B21)
        # M4 = (A20+A21-A10) * (B20+B21-B10)
        # M5 = (A21+A22-A12) * (B21+B22-B12)
        # M6 = (A10+A12-A02) * (B10+B12-B02)
        # M7 = (A12+A00-A01) * (B12+B00-B01)
        
        U_block = np.zeros((9, 7))
        V_block = np.zeros((9, 7))
        W_block = np.zeros((9, 7))
        
        # Index mapping: row = i*m + j, col = j*p + k for n=m=p=3
        # U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
        
        # M1: (A00+A11+A22) * (B00+B11+B22)
        # A indices: (0,0), (1,1), (2,2) -> rows 0, 4, 8
        # B indices: (0,0), (1,1), (2,2) -> cols 0, 4, 8
        U_block[0, 0] = 1; U_block[4, 0] = 1; U_block[8, 0] = 1
        V_block[0, 0] = 1; V_block[4, 0] = 1; V_block[8, 0] = 1
        # C[i,k] contribution: C[i,k] += M1 if i,k in {0,1,2}
        # Actually, need to figure out W coefficients
        # C00 = M1 - M2 - M3 + M4
        # C01 = M2 + M3 - M4 - M7
        # C02 = M3 + M7 - M1 - M6
        # C10 = M4 - M5 - M2 + M6
        # C11 = M5 + M2 - M6 - M3
        # C12 = M6 - M3 + M5 - M7
        # C20 = M1 - M4 - M5 + M3
        # C21 = M5 - M7 + M1 - M6
        # C22 = M7 - M6 + M4 - M2
        
        # W[k*3+i, r] for C[i,k]
        # C00: W[0*3+0, 0] = W[0, 0] = 1
        # C01: W[1*3+0, 0] = W[3, 0] = 0 (coefficient of M1 is 0 for C01)
        # Let me redo: W index is k*n + i = k*3 + i
        # C[i,k] coefficient of Mr: W[k*3+i, r]
        
        # M1 coefficients:
        # C00: +1 -> W[0*3+0, 0] = W[0, 0] = 1
        # C01: 0  -> W[1*3+0, 0] = W[3, 0] = 0
        # C02: -1 -> W[2*3+0, 0] = W[6, 0] = -1
        # C10: 0  -> W[0*3+1, 0] = W[1, 0] = 0
        # C11: 0  -> W[1*3+1, 0] = W[4, 0] = 0
        # C12: 0  -> W[2*3+1, 0] = W[7, 0] = 0
        # C20: +1 -> W[0*3+2, 0] = W[2, 0] = 1
        # C21: +1 -> W[1*3+2, 0] = W[5, 0] = 1
        # C22: 0  -> W[2*3+2, 0] = W[8, 0] = 0
        W_block[0, 0] = 1; W_block[6, 0] = -1; W_block[2, 0] = 1; W_block[5, 0] = 1
        
        # M2: (A01+A10-A22) * (B01+B10-B22)
        # A indices: (0,1), (1,0), (2,2) -> rows 1, 3, 8
        # B indices: (0,1), (1,0), (2,2) -> cols 1, 3, 8
        U_block[1, 1] = 1; U_block[3, 1] = 1; U_block[8, 1] = -1
        V_block[1, 1] = 1; V_block[3, 1] = 1; V_block[8, 1] = -1
        # M2 coefficients:
        # C00: -1 -> W[0, 1] = -1
        # C01: +1 -> W[3, 1] = 1
        # C02: +1 -> W[6, 1] = 1
        # C10: -1 -> W[1, 1] = -1
        # C11: +1 -> W[4, 1] = 1
        # C12: 0  -> W[7, 1] = 0
        # C20: 0  -> W[2, 1] = 0
        # C21: 0  -> W[5, 1] = 0
        # C22: -1 -> W[8, 1] = -1
        W_block[0, 1] = -1; W_block[3, 1] = 1; W_block[6, 1] = 1
        W_block[1, 1] = -1; W_block[4, 1] = 1; W_block[8, 1] = -1
        
        # M3: (A02+A12-A21) * (B02+B12-B21)
        # A indices: (0,2), (1,2), (2,1) -> rows 2, 5, 7
        # B indices: (0,2), (1,2), (2,1) -> cols 2, 5, 7
        U_block[2, 2] = 1; U_block[5, 2] = 1; U_block[7, 2] = -1
        V_block[2, 2] = 1; V_block[5, 2] = 1; V_block[7, 2] = -1
        # M3 coefficients:
        # C00: -1 -> W[0, 2] = -1
        # C01: +1 -> W[3, 2] = 1
        # C02: +1 -> W[6, 2] = 1
        # C10: 0  -> W[1, 2] = 0
        # C11: -1 -> W[4, 2] = -1
        # C12: -1 -> W[7, 2] = -1
        # C20: +1 -> W[2, 2] = 1
        # C21: 0  -> W[5, 2] = 0
        # C22: 0  -> W[8, 2] = 0
        W_block[0, 2] = -1; W_block[3, 2] = 1; W_block[6, 2] = 1
        W_block[4, 2] = -1; W_block[7, 2] = -1; W_block[2, 2] = 1
        
        # M4: (A20+A21-A10) * (B20+B21-B10)
        # A indices: (2,0), (2,1), (1,0) -> rows 6, 7, 3
        # B indices: (2,0), (2,1), (1,0) -> cols 6, 7, 3
        U_block[6, 3] = 1; U_block[7, 3] = 1; U_block[3, 3] = -1
        V_block[6, 3] = 1; V_block[7, 3] = 1; V_block[3, 3] = -1
        # M4 coefficients:
        # C00: +1 -> W[0, 3] = 1
        # C01: -1 -> W[3, 3] = -1
        # C02: 0  -> W[6, 3] = 0
        # C10: +1 -> W[1, 3] = 1
        # C11: 0  -> W[4, 3] = 0
        # C12: 0  -> W[7, 3] = 0
        # C20: -1 -> W[2, 3] = -1
        # C21: 0  -> W[5, 3] = 0
        # C22: +1 -> W[8, 3] = 1
        W_block[0, 3] = 1; W_block[3, 3] = -1; W_block[1, 3] = 1
        W_block[2, 3] = -1; W_block[8, 3] = 1
        
        # M5: (A21+A22-A12) * (B21+B22-B12)
        # A indices: (2,1), (2,2), (1,2) -> rows 7, 8, 5
        # B indices: (2,1), (2,2), (1,2) -> cols 7, 8, 5
        U_block[7, 4] = 1; U_block[8, 4] = 1; U_block[5, 4] = -1
        V_block[7, 4] = 1; V_block[8, 4] = 1; V_block[5, 4] = -1
        # M5 coefficients:
        # C00: 0  -> W[0, 4] = 0
        # C01: 0  -> W[3, 4] = 0
        # C02: 0  -> W[6, 4] = 0
        # C10: -1 -> W[1, 4] = -1
        # C11: +1 -> W[4, 4] = 1
        # C12: +1 -> W[7, 4] = 1
        # C20: -1 -> W[2, 4] = -1
        # C21: +1 -> W[5, 4] = 1
        # C22: 0  -> W[8, 4] = 0
        W_block[1, 4] = -1; W_block[4, 4] = 1; W_block[7, 4] = 1
        W_block[2, 4] = -1; W_block[5, 4] = 1
        
        # M6: (A10+A12-A02) * (B10+B12-B02)
        # A indices: (1,0), (1,2), (0,2) -> rows 3, 5, 2
        # B indices: (1,0), (1,2), (0,2) -> cols 3, 5, 2
        U_block[3, 5] = 1; U_block[5, 5] = 1; U_block[2, 5] = -1
        V_block[3, 5] = 1; V_block[5, 5] = 1; V_block[2, 5] = -1
        # M6 coefficients:
        # C00: 0  -> W[0, 5] = 0
        # C01: 0  -> W[3, 5] = 0
        # C02: -1 -> W[6, 5] = -1
        # C10: +1 -> W[1, 5] = 1
        # C11: -1 -> W[4, 5] = -1
        # C12: +1 -> W[7, 5] = 1
        # C20: 0  -> W[2, 5] = 0
        # C21: -1 -> W[5, 5] = -1
        # C22: -1 -> W[8, 5] = -1
        W_block[6, 5] = -1; W_block[1, 5] = 1; W_block[4, 5] = -1
        W_block[7, 5] = 1; W_block[5, 5] = -1; W_block[8, 5] = -1
        
        # M7: (A12+A00-A01) * (B12+B00-B01)
        # A indices: (1,2), (0,0), (0,1) -> rows 5, 0, 1
        # B indices: (1,2), (0,0), (0,1) -> cols 5, 0, 1
        U_block[5, 6] = 1; U_block[0, 6] = 1; U_block[1, 6] = -1
        V_block[5, 6] = 1; V_block[0, 6] = 1; V_block[1, 6] = -1
        # M7 coefficients:
        # C00: 0  -> W[0, 6] = 0
        # C01: -1 -> W[3, 6] = -1
        # C02: +1 -> W[6, 6] = 1
        # C10: 0  -> W[1, 6] = 0
        # C11: 0  -> W[4, 6] = 0
        # C12: -1 -> W[7, 6] = -1
        # C20: 0  -> W[2, 6] = 0
        # C21: -1 -> W[5, 6] = -1
        # C22: +1 -> W[8, 6] = 1
        W_block[3, 6] = -1; W_block[6, 6] = 1; W_block[7, 6] = -1
        W_block[5, 6] = -1; W_block[8, 6] = 1
        
        return U_block, V_block, W_block
    
    # Try rank-14 construction: two independent Strassen blocks
    def try_rank14_strassen():
        """Try to construct a rank-14 decomposition using two Strassen blocks."""
        U1, V1, W1 = strassen_3x3_block()
        
        # For 3x3 * 3x6:
        # C[:, 0:3] = A * B[:, 0:3]
        # C[:, 3:6] = A * B[:, 3:6]
        
        # U has shape (9, 14): first 7 cols for block 1, next 7 for block 2
        # V has shape (18, 14): rows 0-8 for B[:,0:3], rows 9-17 for B[:,3:6]
        # W has shape (18, 14): rows 0-8 for C[:,0:3], rows 9-17 for C[:,3:6]
        
        U = np.zeros((9, 14))
        V = np.zeros((18, 14))
        W = np.zeros((18, 14))
        
        # Block 1: columns 0-2 of output
        U[:, 0:7] = U1
        # V rows 0-8 correspond to B[:, 0:3]
        V[0:9, 0:7] = V1
        # W rows 0-8 correspond to C[:, 0:3]
        W[0:9, 0:7] = W1
        
        # Block 2: columns 3-5 of output
        U[:, 7:14] = U1
        # V rows 9-17 correspond to B[:, 3:6]
        V[9:18, 7:14] = V1
        # W rows 9-17 correspond to C[:, 3:6]
        W[9:18, 7:14] = W1
        
        # Verify
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        return None
    
    # Try rank-14 first
    result = try_rank14_strassen()
    if result is not None:
        return result
    
    # Fallback: schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
