"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Returns U, V, W matrices for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indexing:
    U[i*2+j, r] corresponds to A[i,j]
    V[j*2+k, r] corresponds to B[j,k]
    W[k*2+i, r] corresponds to C[i,k]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Strassen's 7 products
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00 += M1, C11 += M1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10 += M2
    
    # M3 = A00 * (B10 + B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = 1
    W[1, 2] = 1 # C01 += M3
    
    # M4 = (A00 - A10) * (B00 + B01)
    U[0, 3] = 1; U[2, 3] = -1
    V[0, 3] = 1; V[1, 3] = 1
    W[0, 3] = 1; W[3, 3] = -1 # C00 += M4, C11 -= M4
    
    # M5 = (A01 - A11) * (B10 + B11)
    U[1, 4] = 1; U[3, 4] = -1
    V[2, 4] = 1; V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1 # C00 -= M5, C11 += M5
    # Wait, standard Strassen:
    # C00 = M1 + M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M4 + M6
    # Let's re-verify M5 and M7 assignments.
    
    # M5 = (A01 - A11) * (B10 + B11)
    # Contribution to C00: -M5?
    # C00 = A00B00 + A01B10.
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A00B00 + A00B01 - A10B00 - A10B01
    # M5 = A01B10 + A01B11 - A11B10 - A11B11
    # M7 = (A01 + A11) * (B10 - B01)? No.
    
    # Let's use the standard explicit formulas for U, V, W.
    
    # Reset W
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # M2: (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1
    
    # M3: A00(B10+B11)
    U[0,2]=1
    V[2,2]=1; V[3,2]=1
    W[1,2]=1
    
    # M4: (A00-A10)(B00+B01)
    U[0,3]=1; U[2,3]=-1
    V[0,3]=1; V[1,3]=1
    W[0,3]=1; W[3,3]=-1
    
    # M5: (A01-A11)(B10+B11)
    U[1,4]=1; U[3,4]=-1
    V[2,4]=1; V[3,4]=1
    W[0,4]=-1; W[3,4]=1
    
    # M6: (A10+A00)(B01-B11)
    U[0,5]=1; U[2,5]=1
    V[1,5]=1; V[3,5]=-1
    W[2,5]=1
    
    # M7: (A10+A01)(B11-B01)
    U[1,6]=1; U[2,6]=1
    V[3,6]=1; V[1,6]=-1
    W[1,6]=1; W[0,6]=-1
    
    return U, V, W

def _build_2x4x2():
    """
    Builds decomposition for 2x4x2 matrix multiplication.
    Rank 14.
    Splits into two 2x2x2 blocks.
    """
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _strassen_2x2x2()
    
    # U shape (8, 14)
    U = np.zeros((8, 14), dtype=float)
    # V shape (8, 14)
    V = np.zeros((8, 14), dtype=float)
    # W shape (4, 14)
    W = np.zeros((4, 14), dtype=float)
    
    # Block 1: A cols 0,1 and B rows 0,1
    # U indices: i*4+j. For j in {0,1}, row is i*4+j.
    # V indices: j*2+k. For j in {0,1}, row is j*2+k.
    # W indices: k*2+i. For k in {0,1}, row is k*2+i.
    
    # U1 is 4x7. Rows 0..3 correspond to A00, A01, A10, A11.
    # We need to map these to U_global rows.
    # A00 -> U[0, :], A01 -> U[1, :], A10 -> U[4, :], A11 -> U[5, :]
    U[0, 0:7] = U1[0]
    U[1, 0:7] = U1[1]
    U[4, 0:7] = U1[2]
    U[5, 0:7] = U1[3]
    
    # V1 is 4x7. Rows 0..3 correspond to B00, B01, B10, B11.
    # V_global rows: j*2+k.
    # B00 (j=0,k=0) -> V[0, :]
    # B01 (j=0,k=1) -> V[1, :]
    # B10 (j=1,k=0) -> V[2, :]
    # B11 (j=1,k=1) -> V[3, :]
    V[0, 0:7] = V1[0]
    V[1, 0:7] = V1[1]
    V[2, 0:7] = V1[2]
    V[3, 0:7] = V1[3]
    
    # W1 is 4x7. Rows 0..3 correspond to C00, C01, C10, C11.
    # W_global rows: k*2+i.
    # C00 (k=0,i=0) -> W[0, :]
    # C01 (k=1,i=0) -> W[2, :]
    # C10 (k=0,i=1) -> W[1, :]
    # C11 (k=1,i=1) -> W[3, :]
    W[0, 0:7] = W1[0]
    W[2, 0:7] = W1[1]
    W[1, 0:7] = W1[2]
    W[3, 0:7] = W1[3]
    
    # Block 2: A cols 2,3 and B rows 2,3
    # U indices: i*4+j. For j in {2,3}.
    # A20 (i=0,j=2) -> U[2, :]
    # A21 (i=0,j=3) -> U[3, :]
    # A30 (i=1,j=2) -> U[6, :]
    # A31 (i=1,j=3) -> U[7, :]
    
    U[2, 7:14] = U2[0]
    U[3, 7:14] = U2[1]
    U[6, 7:14] = U2[2]
    U[7, 7:14] = U2[3]
    
    # V indices: j*2+k. For j in {2,3}.
    # B20 (j=2,k=0) -> V[4, :]
    # B21 (j=2,k=1) -> V[5, :]
    # B30 (j=3,k=0) -> V[6, :]
    # B31 (j=3,k=1) -> V[7, :]
    V[4, 7:14] = V2[0]
    V[5, 7:14] = V2[1]
    V[6, 7:14] = V2[2]
    V[7, 7:14] = V2[3]
    
    # W indices: k*2+i. For k in {0,1} (since this block contributes to C cols 0,1? No.)
    # Wait, the second block uses B rows 2,3.
    # The output C is 2x2.
    # C = A1 B1 + A2 B2.
    # Both blocks contribute to the SAME C.
    # So W indices are the same: k*2+i.
    W[0, 7:14] = W2[0]
    W[2, 7:14] = W2[1]
    W[1, 7:14] = W2[2]
    W[3, 7:14] = W2[3]
    
    return U, V, W

def _build_2x4x1():
    """
    Builds decomposition for 2x4x1 matrix multiplication.
    Rank 4.
    """
    U = np.zeros((8, 4), dtype=float)
    V = np.zeros((4, 4), dtype=float) # j*1+k -> j
    W = np.zeros((2, 4), dtype=float) # k*2+i -> i
    
    for j in range(4):
        U[0*4+j, j] = 1
        U[1*4+j, j] = 1
        V[j*1+0, j] = 1
        W[0*2+0, j] = 1
        W[0*2+1, j] = 1
        
    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify that the tensor decomposition is correct."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n * m, U_arr.shape[1]):
        return False
    if V_arr.shape != (m * p, V_arr.shape[1]):
        return False
    if W_arr.shape != (p * n, W_arr.shape[1]):
        return False
    
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def construct():
    # Build 2x4x2 decomposition (Rank 14)
    U_242, V_242, W_242 = _build_2x4x2()
    
    # Build 2x4x1 decomposition (Rank 4)
    U_241, V_241, W_241 = _build_2x4x1()
    
    # Combine for 2x4x5
    # C has 5 columns.
    # Block 1 (2x4x2) handles cols 0,1.
    # Block 2 (2x4x2) handles cols 2,3.
    # Block 3 (2x4x1) handles col 4.
    
    # We need two instances of 2x4x2.
    # Instance 1: Cols 0,1.
    # Instance 2: Cols 2,3.
    
    # The U and V matrices for the two 2x4x2 blocks are identical in structure 
    # relative to A and B, but they operate on different parts of B?
    # No, _build_2x4x2 assumes B is 4x2.
    # For Instance 1, B is the first 2 columns of the original 4x5 B?
    # No, the original B is 4x5.
    # My _build_2x4x2 constructs a decomposition for A(2x4) * B(4x2).
    # So V has shape (4*2, 14) = (8, 14).
    # In the global problem, B is 4x5.
    # V_global has shape (4*5, R) = (20, R).
    
    # Let's re-assemble carefully.
    
    R_242 = 14
    R_241 = 4
    Total_Rank = R_242 + R_242 + R_241
    
    U_global = np.zeros((8, Total_Rank), dtype=float)
    V_global = np.zeros((20, Total_Rank), dtype=float)
    W_global = np.zeros((10, Total_Rank), dtype=float)
    
    # Block 1: A(2x4) * B_cols_01(4x2) -> C_cols_01(2x2)
    # U_242 is 8x14.
    U_global[:, 0:R_242] = U_242
    
    # V_242 is 8x14. Rows j*2+k.
    # In global V, rows are j*5+k.
    # For k=0,1:
    # V_242 row 0 (j=0,k=0) -> V_global row 0*5+0 = 0
    # V_242 row 1 (j=0,k=1) -> V_global row 0*5+1 = 1
    # V_242 row 2 (j=1,k=0) -> V_global row 1*5+0 = 5
    # V_242 row 3 (j=1,k=1) -> V_global row 1*5+1 = 6
    # V_242 row 4 (j=2,k=0) -> V_global row 2*5+0 = 10
    # V_242 row 5 (j=2,k=1) -> V_global row 2*5+1 = 11
    # V_242 row 6 (j=3,k=0) -> V_global row 3*5+0 = 15
    # V_242 row 7 (j=3,k=1) -> V_global row 3*5+1 = 16
    
    v_map_1 = [0, 1, 5, 6, 10, 11, 15, 16]
    for i_local, i_global in enumerate(v_map_1):
        V_global[i_global, 0:R_242] = V_242[i_local]
        
    # W_242 is 4x14. Rows k*2+i.
    # In global W, rows are k*2+i.
    # For k=0,1:
    # W_242 row 0 (k=0,i=0) -> W_global row 0
    # W_242 row 1 (k=0,i=1) -> W_global row 1
    # W_242 row 2 (k=1,i=0) -> W_global row 2
    # W_242 row 3 (k=1,i=1) -> W_global row 3
    W_global[0:4, 0:R_242] = W_242
    
    # Block 2: A(2x4) * B_cols_23(4x2) -> C_cols_23(2x2)
    # U is the same.
    U_global[:, R_242:2*R_242] = U_242
    
    # V_242 rows map to global V rows for k=2,3.
    # V_242 row 0 (j=0,k=0) -> V_global row 0*5+2 = 2
    # V_242 row 1 (j=0,k=1) -> V_global row 0*5+3 = 3
    # V_242 row 2 (j=1,k=0) -> V_global row 1*5+2 = 7
    # V_242 row 3 (j=1,k=1) -> V_global row 1*5+3 = 8
    # V_242 row 4 (j=2,k=0) -> V_global row 2*5+2 = 12
    # V_242 row 5 (j=2,k=1) -> V_global row 2*5+3 = 13
    # V_242 row 6 (j=3,k=0) -> V_global row 3*5+2 = 17
    # V_242 row 7 (j=3,k=1) -> V_global row 3*5+3 = 18
    
    v_map_2 = [2, 3, 7, 8, 12, 13, 17, 18]
    for i_local, i_global in enumerate(v_map_2):
        V_global[i_global, R_242:2*R_242] = V_242[i_local]
        
    # W_242 rows map to global W rows for k=2,3.
    # W_242 row 0 (k=0,i=0) -> W_global row 2*2+0 = 4
    # W_242 row 1 (k=0,i=1) -> W_global row 2*2+1 = 5
    # W_242 row 2 (k=1,i=0) -> W_global row 3*2+0 = 6
    # W_242 row 3 (k=1,i=1) -> W_global row 3*2+1 = 7
    W_global[4:8, R_242:2*R_242] = W_242
    
    # Block 3: A(2x4) * B_col_4(4x1) -> C_col_4(2x1)
    # U_241 is 8x4.
    U_global[:, 2*R_242:Total_Rank] = U_241
    
    # V_241 is 4x4. Rows j*1+k = j.
    # In global V, rows are j*5+4.
    # V_241 row 0 (j=0) -> V_global row 0*5+4 = 4
    # V_241 row 1 (j=1) -> V_global row 1*5+4 = 9
    # V_241 row 2 (j=2) -> V_global row 2*5+4 = 14
    # V_241 row 3 (j=3) -> V_global row 3*5+4 = 19
    v_map_3 = [4, 9, 14, 19]
    for i_local, i_global in enumerate(v_map_3):
        V_global[i_global, 2*R_242:Total_Rank] = V_241[i_local]
        
    # W_241 is 2x4. Rows k*2+i = i (since k=0).
    # In global W, rows are k*2+i. For k=4.
    # W_241 row 0 (i=0) -> W_global row 4*2+0 = 8
    # W_241 row 1 (i=1) -> W_global row 4*2+1 = 9
    W_global[8:10, 2*R_242:Total_Rank] = W_241
    
    # Convert to lists for return
    U_list = U_global.tolist()
    V_list = V_global.tolist()
    W_list = W_global.tolist()
    
    assert _verify_decomposition(U_list, V_list, W_list), "Decomposition verification failed"
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
