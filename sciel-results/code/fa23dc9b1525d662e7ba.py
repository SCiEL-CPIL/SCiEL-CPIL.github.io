"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    Uses Kronecker product of Strassen 2x2x2 (rank 7) and M_1,2,3 (rank 6) to get rank 42.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Strassen 2x2x2 decomposition (rank 7)
    # U_str, V_str, W_str each shape (4, 7)
    # Indexing: U[i,j,r] where i,j in {0,1} -> flat index i*2+j
    #           V[j,k,r] where j,k in {0,1} -> flat index j*2+k
    #           W[k,i,r] where k,i in {0,1} -> flat index k*2+i
    
    U_str = np.array([
        [1, 0, 0, 0, 0, 0, 0],   # (0,0): A00
        [0, 0, 0, 0, 0, 0, 0],   # (0,1): A01 - not used directly in Strassen
        [0, 0, 0, 0, 0, 0, 0],   # (1,0): A10 - not used directly
        [0, 1, 0, 0, 0, 0, 0],   # (1,1): A11
    ], dtype=float)
    
    # Actually, let me use the standard Strassen formulation properly.
    # The tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    # For 2x2x2, we need rank 7.
    
    # Standard Strassen:
    # Let M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # U matrix: rows indexed by (i,j) -> i*2+j, columns by r=0..6
    # U[i,j,r] = coefficient of A_ij in Mr
    U_str = np.zeros((4, 7), dtype=float)
    # M1: A00 + A11
    U_str[0, 0] = 1; U_str[3, 0] = 1  # (0,0), (1,1)
    # M2: A10 + A11
    U_str[2, 1] = 1; U_str[3, 1] = 1  # (1,0), (1,1)
    # M3: A00
    U_str[0, 2] = 1  # (0,0)
    # M4: A11
    U_str[3, 3] = 1  # (1,1)
    # M5: A00 + A01
    U_str[0, 4] = 1; U_str[1, 4] = 1  # (0,0), (0,1)
    # M6: A10 - A00
    U_str[2, 5] = 1; U_str[0, 5] = -1  # (1,0), (0,0)
    # M7: A01 - A11
    U_str[1, 6] = 1; U_str[3, 6] = -1  # (0,1), (1,1)
    
    # V matrix: rows indexed by (j,k) -> j*2+k, columns by r=0..6
    # V[j,k,r] = coefficient of B_jk in Mr
    V_str = np.zeros((4, 7), dtype=float)
    # M1: B00 + B11
    V_str[0, 0] = 1; V_str[3, 0] = 1  # (0,0), (1,1)
    # M2: B00
    V_str[0, 1] = 1  # (0,0)
    # M3: B10 - B11
    V_str[2, 2] = 1; V_str[3, 2] = -1  # (1,0), (1,1)
    # M4: B01 - B10
    V_str[1, 3] = 1; V_str[2, 3] = -1  # (0,1), (1,0)
    # M5: B11
    V_str[3, 4] = 1  # (1,1)
    # M6: B00 + B01
    V_str[0, 5] = 1; V_str[1, 5] = 1  # (0,0), (0,1)
    # M7: B10 + B11
    V_str[2, 6] = 1; V_str[3, 6] = 1  # (1,0), (1,1)
    
    # W matrix: rows indexed by (k,i) -> k*2+i, columns by r=0..6
    # W[k,i,r] = coefficient in Mr that contributes to C_ki
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    W_str = np.zeros((4, 7), dtype=float)
    # (k,i) = (0,0) -> C00: M1+M4-M5-M7
    W_str[0, 0] = 1; W_str[0, 3] = 1; W_str[0, 4] = -1; W_str[0, 6] = -1
    # (k,i) = (0,1) -> C01: M3+M5
    W_str[1, 2] = 1; W_str[1, 4] = 1
    # (k,i) = (1,0) -> C10: M2+M4
    W_str[2, 1] = 1; W_str[2, 3] = 1
    # (k,i) = (1,1) -> C11: M1-M2+M3+M7
    W_str[3, 0] = 1; W_str[3, 1] = -1; W_str[3, 2] = 1; W_str[3, 6] = 1
    
    # Verify Strassen
    def verify_strassen(U, V, W, n2, m2, p2):
        U_t = U.reshape(n2, m2, -1)
        V_t = V.reshape(m2, p2, -1)
        W_t = W.reshape(p2, n2, -1)
        T = np.einsum('ijr,jkr,kin->ijk', U_t, V_t, W_t)
        return np.allclose(T, 1.0, atol=1e-10)
    
    assert verify_strassen(U_str, V_str, W_str, 2, 2, 2), "Strassen verification failed"
    
    # M_1,2,3 schoolbook decomposition (rank 6)
    # n1=1, m1=2, p1=3
    # U_123 shape (1*2, 6) = (2, 6)
    # V_123 shape (2*3, 6) = (6, 6)
    # W_123 shape (3*1, 6) = (3, 6)
    
    U_123 = np.zeros((2, 6), dtype=float)
    V_123 = np.zeros((6, 6), dtype=float)
    W_123 = np.zeros((3, 6), dtype=float)
    
    col = 0
    for i in range(1):
        for j in range(2):
            for k in range(3):
                U_123[i * 2 + j, col] = 1
                V_123[j * 3 + k, col] = 1
                W_123[k * 1 + i, col] = 1
                col += 1
    
    # Verify M_1,2,3
    def verify_123(U, V, W, n1, m1, p1):
        U_t = U.reshape(n1, m1, -1)
        V_t = V.reshape(m1, p1, -1)
        W_t = W.reshape(p1, n1, -1)
        T = np.einsum('ijr,jkr,kin->ijk', U_t, V_t, W_t)
        return np.allclose(T, 1.0, atol=1e-10)
    
    assert verify_123(U_123, V_123, W_123, 1, 2, 3), "M_1,2,3 verification failed"
    
    # Kronecker product:
    # U_total = kron(U_str, U_123) shape (4*2, 7*6) = (8, 42)
    # V_total = kron(V_str, V_123) shape (4*6, 7*6) = (24, 42)
    # W_total = kron(W_str, W_123) shape (4*3, 7*6) = (12, 42)
    
    U_total = np.kron(U_str, U_123)
    V_total = np.kron(V_str, V_123)
    W_total = np.kron(W_str, W_123)
    
    # Verify the full decomposition
    def verify_full(U, V, W, n, m, p):
        r = U.shape[1]
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kin->ijk', U_t, V_t, W_t)
        return np.allclose(T, 1.0, atol=1e-9)
    
    assert verify_full(U_total, V_total, W_total, n, m, p), "Full decomposition verification failed"
    
    rank = U_total.shape[1]
    
    # Convert to integer if possible (values are integers)
    U_out = U_total.astype(int) if np.allclose(U_total, np.round(U_total)) else U_total
    V_out = V_total.astype(int) if np.allclose(V_total, np.round(V_total)) else V_total
    W_out = W_total.astype(int) if np.allclose(W_total, np.round(W_total)) else W_total
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
