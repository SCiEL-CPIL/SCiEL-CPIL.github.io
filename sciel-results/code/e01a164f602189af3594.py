"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2_2x2():
    """Return U, V, W for 2x2x2 matrix mult, rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7)
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2*2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U: rows indexed by i*2+j (i=row of A, j=col of A)
    # V: rows indexed by j*2+k (j=row of B, k=col of B)
    # W: rows indexed by k*2+i (k=col of output, i=row of output)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Row indices for A (i*2+j): A00=0, A01=1, A10=2, A11=3
    # Row indices for B (j*2+k): B00=0, B01=1, B10=2, B11=3
    # Row indices for C (k*2+i): C00=0, C01=1, C10=2, C11=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1  # C00 gets +M1, C11 gets +M1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1  # C10 gets +M2, C11 gets -M2
    
    # M3 = A00(B01-B11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1  # C01 gets +M3, C11 gets +M3
    
    # M4 = A11(B10-B00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1  # C00 gets +M4, C10 gets +M4
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1  # C00 gets -M5, C01 gets +M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1  # C11 gets +M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1  # C00 gets +M7
    
    return U, V, W

def _kron_decomp_4x4_4x4():
    """Return U, V, W for 4x4x4 matrix mult, rank 49.
    U shape (16,49), V shape (16,49), W shape (16,49)
    Using Kronecker product of two 2x2x2 Strassen decompositions.
    """
    U2, V2, W2 = _strassen_2x2_2x2()
    
    # Kronecker product: for 4x4 = 2x2 ⊗ 2x2
    # A is 4x4, indexed as A[(i1,i2),(j1,j2)] where i1,i2,j1,j2 in {0,1}
    # B is 4x4, indexed as B[(j1,j2),(k1,k2)]
    # C is 4x4, indexed as C[(i1,i2),(k1,k2)]
    # 
    # The tensor factors: U has rows (i1,i2,j1,j2) -> index i1*4+i2*2+j1*2+j2? 
    # Actually let's use: A row index = i1*2 + i2, col index = j1*2 + j2
    # So A[(i1,i2),(j1,j2)] has row i1*2+i2, col j1*2+j2
    # U row index = (i1*2+i2)*4 + (j1*2+j2) = i1*8+i2*4+j1*2+j2
    # V row index = (j1*2+j2)*4 + (k1*2+k2) = j1*8+j2*4+k1*2+k2
    # W row index = (k1*2+k2)*4 + (i1*2+i2) = k1*8+k2*4+i1*2+i2
    
    # For Kronecker: U_4x4[(i1,i2,j1,j2), (c1,c2)] = U2[i1,j1,c1] * U2[i2,j2,c2]
    # where c1,c2 are the 7 Strassen indices for the two 2x2 sub-blocks
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    rowU = i1*8 + i2*4 + j1*2 + j2
                    for c1 in range(7):
                        for c2 in range(7):
                            col = c1*7 + c2
                            uval = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
                            if uval != 0:
                                U4[rowU, col] = uval
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    rowV = j1*8 + j2*4 + k1*2 + k2
                    for c1 in range(7):
                        for c2 in range(7):
                            col = c1*7 + c2
                            vval = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
                            if vval != 0:
                                V4[rowV, col] = vval
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    rowW = k1*8 + k2*4 + i1*2 + i2
                    for c1 in range(7):
                        for c2 in range(7):
                            col = c1*7 + c2
                            wval = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
                            if wval != 0:
                                W4[rowW, col] = wval
    
    return U4, V4, W4

def construct():
    """Construct rank-98 decomposition for 4x4x8 matrix multiplication."""
    deadline = time.time() + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get 4x4x4 decomposition (rank 49)
    U4, V4, W4 = _kron_decomp_4x4_4x4()
    
    # For 4x4x8: B is 4x8, so B = [B_left | B_right] where each is 4x4
    # C = A*B = [A*B_left | A*B_right], each 4x4
    # 
    # U has shape (n*m, rank) = (16, 98)
    # V has shape (m*p, rank) = (32, 98)  
    # W has shape (p*n, rank) = (32, 98)
    #
    # For the left block (k=0..3):
    #   U_left: rows (i,j) for i,j in 0..3, from U4
    #   V_left: rows (j,k) for j in 0..3, k in 0..3
    #   W_left: rows (k,i) for k in 0..3, i in 0..3
    #
    # For the right block (k=4..7):
    #   U_right: same U4 (A is the same)
    #   V_right: rows (j,k) for j in 0..3, k in 4..7
    #   W_right: rows (k,i) for k in 4..7, i in 0..3
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left block: columns 0-48
    U[:, 0:49] = U4
    # V_left: V[j*4+k, c] for j,k in 0..3
    for j in range(4):
        for k in range(4):
            V[j*4+k, 0:49] = V4[j*4+k, 0:49]
    # W_left: W[k*4+i, c] for k,i in 0..3
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Right block: columns 49-97
    U[:, 49:98] = U4
    # V_right: V[j*4+k, c] for j in 0..3, k in 4..7
    for j in range(4):
        for k in range(4, 8):
            V[j*4+k, 49:98] = V4[j*4+(k-4), 0:49]
    # W_right: W[k*4+i, c] for k in 4..7, i in 0..3
    for k in range(4, 8):
        for i in range(4):
            W[k*4+i, 49:98] = W4[(k-4)*4+i, 0:49]
    
    # Verify decomposition
    # C[i,k] = sum_j sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # But our convention is W[k*n+i, c], so W index is k*4+i
    
    # Convert to lists for return
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    # Quick verification with random matrices
    if time.time() < deadline:
        np.random.seed(42)
        A = np.random.randn(4, 4)
        B = np.random.randn(4, 8)
        C_expected = A @ B
        
        U_arr = np.array(U_list)
        V_arr = np.array(V_list)
        W_arr = np.array(W_list)
        
        C_computed = np.zeros((4, 8))
        for i in range(4):
            for k in range(8):
                s = 0.0
                for j in range(4):
                    for c in range(98):
                        s += U_arr[i*4+j, c] * V_arr[j*4+k, c] * W_arr[k*4+i, c]
                C_computed[i, k] = s
        
        error = np.max(np.abs(C_computed - C_expected))
        if error > 1e-10:
            # Fallback to schoolbook if verification fails
            rank_fb = 128
            U_fb = [[0]*rank_fb for _ in range(16)]
            V_fb = [[0]*rank_fb for _ in range(32)]
            W_fb = [[0]*rank_fb for _ in range(32)]
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(8):
                        U_fb[i*4+j][col] = 1
                        V_fb[j*4+k][col] = 1
                        W_fb[k*4+i][col] = 1
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
