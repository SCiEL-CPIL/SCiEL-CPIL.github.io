"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 130  # Safety margin: return by 130s (20s before 150s limit)

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Strategy: 7 Strassen blocks (2 cols each) + 1 schoolbook block (1 col)
    # Rank = 7*7 + 4 = 53
    # 
    # U shape: (n*m, rank) = (4, 53)
    # V shape: (m*p, rank) = (30, 53)  
    # W shape: (p*n, rank) = (30, 53)
    #
    # Indexing:
    #   U[i*m + j, r] : A[i,j]
    #   V[j*p + k, r] : B[j,k]
    #   W[k*n + i, r] : C[i,k] (trace convention: W[k*n+i])
    
    total_rank = 53
    U = np.zeros((n * m, total_rank))  # 4 x 53
    V = np.zeros((m * p, total_rank))  # 30 x 53
    W = np.zeros((p * n, total_rank))  # 30 x 53
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For A(2x2) * B(2x2) = C(2x2):
    # U_str[i*2+j, r], V_str[j*2+k, r], W_str[k*2+i, r]
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_str = np.zeros((4, 7))
    V_str = np.zeros((4, 7))
    W_str = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U_str[0, 0] = 1; U_str[3, 0] = 1  # A00, A11
    V_str[0, 0] = 1; V_str[3, 0] = 1  # B00, B11
    W_str[0, 0] = 1; W_str[3, 0] = 1  # C00(k=0,i=0), C11(k=1,i=1)
    
    # M2: (A10+A11)B00
    U_str[2, 1] = 1; U_str[3, 1] = 1  # A10, A11
    V_str[0, 1] = 1  # B00
    W_str[1, 1] = 1  # C10(k=0,i=1)
    W_str[3, 1] = -1 # C11(k=1,i=1)
    
    # M3: A00(B01-B11)
    U_str[0, 2] = 1  # A00
    V_str[1, 2] = 1; V_str[3, 2] = -1  # B01, -B11
    W_str[2, 2] = 1  # C01(k=1,i=0)
    W_str[3, 2] = 1  # C11(k=1,i=1)
    
    # M4: A11(B10-B00)
    U_str[3, 3] = 1  # A11
    V_str[2, 3] = 1; V_str[0, 3] = -1  # B10, -B00
    W_str[0, 3] = 1  # C00(k=0,i=0)
    W_str[1, 3] = 1  # C10(k=0,i=1)
    
    # M5: (A00+A01)B11
    U_str[0, 4] = 1; U_str[1, 4] = 1  # A00, A01
    V_str[3, 4] = 1  # B11
    W_str[0, 4] = -1 # C00(k=0,i=0)
    W_str[2, 4] = 1  # C01(k=1,i=0)
    
    # M6: (A10-A00)(B00+B01)
    U_str[2, 5] = 1; U_str[0, 5] = -1  # A10, -A00
    V_str[0, 5] = 1; V_str[1, 5] = 1  # B00, B01
    W_str[3, 5] = 1  # C11(k=1,i=1)
    
    # M7: (A01-A11)(B10+B11)
    U_str[1, 6] = 1; U_str[3, 6] = -1  # A01, -A11
    V_str[2, 6] = 1; V_str[3, 6] = 1  # B10, B11
    W_str[0, 6] = 1  # C00(k=0,i=0)
    
    # Schoolbook 2x2x1 decomposition (rank 4)
    # A(2x2) * B(2x1) = C(2x1)
    # U_sb[i*2+j, r], V_sb[j*1+k, r], W_sb[k*2+i, r]
    U_sb = np.zeros((4, 4))
    V_sb = np.zeros((2, 4))
    W_sb = np.zeros((2, 4))
    
    # Q1 = A00 * B0
    U_sb[0, 0] = 1
    V_sb[0, 0] = 1
    W_sb[0, 0] = 1  # C0(k=0,i=0)
    
    # Q2 = A01 * B1
    U_sb[1, 1] = 1
    V_sb[1, 1] = 1
    W_sb[0, 1] = 1  # C0(k=0,i=0)
    
    # Q3 = A10 * B0
    U_sb[2, 2] = 1
    V_sb[0, 2] = 1
    W_sb[1, 2] = 1  # C1(k=0,i=1)
    
    # Q4 = A11 * B1
    U_sb[3, 3] = 1
    V_sb[1, 3] = 1
    W_sb[1, 3] = 1  # C1(k=0,i=1)
    
    # Place 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for block_idx in range(7):
        if time.time() > _DEADLINE:
            actual_rank = 7 * block_idx
            return {"U": U[:, :actual_rank].tolist(), 
                    "V": V[:, :actual_rank].tolist(), 
                    "W": W[:, :actual_rank].tolist()}
        
        base_r = 7 * block_idx
        k0 = 2 * block_idx
        k1 = 2 * block_idx + 1
        
        # U: same for all blocks
        U[:, base_r:base_r+7] = U_str
        
        # V: map local (j, k_local) to global (j, k_global)
        for j in range(m):
            for k_local in range(2):
                k_global = k0 + k_local
                V[j * p + k_global, base_r:base_r+7] = V_str[j * 2 + k_local, :]
        
        # W: map local (k_local, i) to global (k_global, i), W[k*n + i]
        for k_local in range(2):
            k_global = k0 + k_local
            for i in range(n):
                W[k_global * n + i, base_r:base_r+7] = W_str[k_local * 2 + i, :]
    
    # Place schoolbook block for column 14
    if time.time() > _DEADLINE:
        actual_rank = 49
        return {"U": U[:, :actual_rank].tolist(), 
                "V": V[:, :actual_rank].tolist(), 
                "W": W[:, :actual_rank].tolist()}
    
    base_r = 49
    k_single = 14
    
    U[:, base_r:base_r+4] = U_sb
    
    # V: map local (j, k_local=0) to global (j, k_single)
    for j in range(m):
        V[j * p + k_single, base_r:base_r+4] = V_sb[j * 1 + 0, :]
    
    # W: map local (k_local=0, i) to global (k_single, i), W[k*n + i]
    for i in range(n):
        W[k_single * n + i, base_r:base_r+4] = W_sb[0 * 2 + i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
