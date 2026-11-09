"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2():
    """Return (U, V, W) for 2x2 matrix multiplication, rank 7.
    Flat indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Standard Strassen decomposition
    # Factor 0: M1 = (A00+A11)(B00+B11), contributes to C00 and C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Factor 1: M2 = (A10+A11)*B00, contributes to C10
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # Factor 2: M3 = A00*(B10-B11), contributes to C01
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # Factor 3: M4 = A11*(B01-B10), contributes to C01 and C00 (negative)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1
    
    # Factor 4: M5 = (A00+A01)*B11, contributes to C00 and C10
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # Factor 5: M6 = (A10-A00)*(B00+B10), contributes to C11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # Factor 6: M7 = (A01-A11)*(B01+B11), contributes to C00 and C11
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _kron_strassen_4x4x4():
    """Build 4x4x4 via Kronecker product of two Strassen(2,2), rank 49.
    Uses vectorized numpy for speed.
    """
    S_U, S_V, S_W = _strassen_2x2()
    S_U_3d = S_U.reshape(2, 2, 7)
    S_V_3d = S_V.reshape(2, 2, 7)
    S_W_3d = S_W.reshape(2, 2, 7)
    
    rank = 49
    U_44 = np.zeros((16, rank))
    V_44 = np.zeros((16, rank))
    W_44 = np.zeros((16, rank))
    
    # Kronecker product logic:
    # U_44[i,j, r1,r2] = U_2x2[i//2, j//2, r1] * U_2x2[i%2, j%2, r2]
    # We can do this with broadcasting.
    
    U_tensor = S_U_3d[:, None, :, None, :] * S_U_3d[None, :, None, :, :]
    # Shape: (2, 2, 7, 2, 2, 7) -> indices: i_hi, i_lo, r_hi, j_hi, j_lo, r_lo
    # We want shape (4, 4, 49) where i = i_hi*2 + i_lo, j = j_hi*2 + j_lo, r = r_hi*7 + r_lo
    # Transpose to (i_hi, j_hi, i_lo, j_lo, r_hi, r_lo)
    U_tensor = U_tensor.transpose(0, 3, 1, 4, 2, 5)
    U_44 = U_tensor.reshape(16, 49)
    
    V_tensor = S_V_3d[:, None, :, None, :] * S_V_3d[None, :, None, :, :]
    # Shape: (2, 2, 7, 2, 2, 7) -> indices: j_hi, j_lo, r_hi, k_hi, k_lo, r_lo
    # We want shape (4, 4, 49) where j = j_hi*2 + j_lo, k = k_hi*2 + k_lo, r = r_hi*7 + r_lo
    # Transpose to (j_hi, k_hi, j_lo, k_lo, r_hi, r_lo)
    V_tensor = V_tensor.transpose(0, 3, 1, 4, 2, 5)
    V_44 = V_tensor.reshape(16, 49)
    
    W_tensor = S_W_3d[:, None, :, None, :] * S_W_3d[None, :, None, :, :]
    # Shape: (2, 2, 7, 2, 2, 7) -> indices: k_hi, k_lo, r_hi, i_hi, i_lo, r_lo
    # We want shape (4, 4, 49) where k = k_hi*2 + k_lo, i = i_hi*2 + i_lo, r = r_hi*7 + r_lo
    # Transpose to (k_hi, i_hi, k_lo, i_lo, r_hi, r_lo)
    W_tensor = W_tensor.transpose(0, 3, 1, 4, 2, 5)
    W_44 = W_tensor.reshape(16, 49)
    
    return U_44, V_44, W_44

def _verify_decomposition(U, V, W, n, m, p):
    """Verify a full decomposition is correct."""
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(T, 1.0, atol=1e-10):
        max_diff = np.max(np.abs(T - 1.0))
        print(f"Verification failed: max diff = {max_diff}")
        return False
    return True

def _schoolbook_4x4x1():
    """Build 4x4x1 schoolbook decomposition, rank 16."""
    n, m, p = 4, 4, 1
    rank = 16
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
            V[j*p+0, col] = 1
            W[0*n+i, col] = 1
            col += 1
            
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 110  # seconds, with safety margin
    
    try:
        # Get 4x4x4 Strassen decomposition (rank 49)
        U_44, V_44, W_44 = _kron_strassen_4x4x4()
        
        # Get 4x4x1 schoolbook decomposition (rank 16)
        U_441, V_441, W_441 = _schoolbook_4x4x1()
        
        # Total rank: 49 + 49 + 16 = 114
        total_rank = 49 + 49 + 16
        
        U_total = np.zeros((16, total_rank))
        V_total = np.zeros((16, total_rank))
        W_total = np.zeros((36, total_rank))
        
        # Block 1: k=0..3 (first 4 columns of B)
        U_total[:, 0:49] = U_44
        V_total[:, 0:49] = V_44
        W_total[0:16, 0:49] = W_44  # W index k*n+i, k in 0..3 -> 0..15
        
        # Block 2: k=4..7 (next 4 columns of B)
        U_total[:, 49:98] = U_44
        V_total[:, 49:98] = V_44
        # W index for k=4..7: k*4+i. k=4 -> 16..19, k=5 -> 20..23, etc.
        W_total[16:32, 49:98] = W_44
        
        # Block 3: k=8 (last column of B)
        # U_441 is shape (16, 16), V_441 is (4, 16), W_441 is (4, 16)
        U_total[:, 98:114] = U_441
        V_total[:, 98:114] = V_441
        # W index for k=8: 8*4+i = 32..35
        W_total[32:36, 98:114] = W_441
        
        # Verify
        if _verify_decomposition(U_total, V_total, W_total, 4, 4, 9):
            # Convert to lists of ints/floats
            U_list = U_total.tolist()
            V_list = V_total.tolist()
            W_list = W_total.tolist()
            
            # Clean up floats that are effectively integers
            for mat in [U_list, V_list, W_list]:
                for row in mat:
                    for idx in range(len(row)):
                        val = row[idx]
                        if abs(val - round(val)) < 1e-10:
                            row[idx] = int(round(val))
            
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            print("Decomposition verification failed, falling back to schoolbook.")
            
    except Exception as e:
        print(f"Error in construct: {e}")
        
    # Fallback: Pure schoolbook (rank 144)
    n, m, p = 4, 4, 9
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
