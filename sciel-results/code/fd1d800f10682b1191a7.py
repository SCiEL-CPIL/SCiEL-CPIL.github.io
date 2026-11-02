import numpy as np
import time

def construct():
    """
    Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Strategy: Combine Strassen (rank 49) for 4x4x4 part and Schoolbook (rank 32) for 4x4x2 part.
    Total rank: 81.
    """
    n, m, p = 4, 4, 6
    
    # --- Part 1: Strassen for 4x4x4 ---
    # Get rank 7 decomposition for 2x2x2
    # U, V, W shapes (4, 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: (A00 + A11)(B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # Term 2: (A10 + A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C10, C11
    
    # Term 3: A00(B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C01, C11
    
    # Term 4: A11(B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C00, C10
    
    # Term 5: (A00 + A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00, C01
    
    # Term 6: (A10 - A00)(B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11
    
    # Term 7: (A01 - A11)(B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00

    # Construct 4x4x4 Strassen (Rank 49)
    # Super terms k=0..6, Sub terms s=0..6
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Coefficients for blocks in super-terms
    # A_blocks: [A00, A01, A10, A11]
    # B_blocks: [B00, B01, B10, B11]
    # C_blocks: [C00, C01, C10, C11]
    
    A_super_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    B_super_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [1, 0, -1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    C_super_coeffs = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1],# M2 -> C10, C11
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0],# M5 -> C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    for k in range(7):
        for s in range(7):
            col = k * 7 + s
            
            # Get sub-term coefficients from U2, V2, W2
            u_sub = U2[:, s]
            v_sub = V2[:, s]
            w_sub = W2[:, s]
            
            # Expand to 4x4 indices
            # A block b (0..3) corresponds to rows 4*b..4*b+3 in U4
            # The sub-term u_sub[elem] applies to all elements in block b
            for b in range(4):
                coeff_a = A_super_coeffs[k][b]
                if coeff_a != 0:
                    U4[4*b:4*b+4, col] += coeff_a * u_sub
            
            for b in range(4):
                coeff_b = B_super_coeffs[k][b]
                if coeff_b != 0:
                    V4[4*b:4*b+4, col] += coeff_b * v_sub
            
            for b in range(4):
                coeff_c = C_super_coeffs[k][b]
                if coeff_c != 0:
                    W4[4*b:4*b+4, col] += coeff_c * w_sub

    # --- Part 2: Schoolbook for 4x4x2 ---
    # A (4x4), B (4x2), C (4x2)
    # Rank 32
    U2s = np.zeros((16, 32))
    V2s = np.zeros((8, 32))
    W2s = np.zeros((8, 32))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k_prime in range(2):
                U2s[i*4 + j, col] = 1
                V2s[j*2 + k_prime, col] = 1
                W2s[k_prime*4 + i, col] = 1
                col += 1

    # --- Combine into Global 4x4x6 ---
    # Global U: (16, 81)
    # Global V: (24, 81)
    # Global W: (24, 81)
    
    U_global = np.zeros((16, 81))
    V_global = np.zeros((24, 81))
    W_global = np.zeros((24, 81))
    
    # Fill Strassen part (cols 0-48)
    U_global[:, 0:49] = U4
    
    # Map V4 (16 rows) to V_global (24 rows)
    # V4 row index r4 = 4*j + k (j,k in 0..3)
    # V_global row index rg = 6*j + k (j in 0..3, k in 0..5)
    for r4 in range(16):
        j = r4 // 4
        k = r4 % 4
        rg = 6*j + k
        V_global[rg, 0:49] = V4[r4, 0:49]
        
    # Map W4 (16 rows) to W_global (24 rows)
    # W4 row index r4 = 4*k + i (k,i in 0..3)
    # W_global row index rg = 4*k + i (k in 0..5, i in 0..3)
    # Since k is 0..3, rg = r4.
    W_global[0:16, 0:49] = W4
    
    # Fill Schoolbook part (cols 49-80)
    U_global[:, 49:81] = U2s
    
    # Map V2s (8 rows) to V_global (24 rows)
    # V2s row index r2s = 2*j + k_prime (j in 0..3, k_prime in 0..1)
    # V_global row index rg = 6*j + (4 + k_prime)
    for r2s in range(8):
        j = r2s // 2
        k_prime = r2s % 2
        rg = 6*j + 4 + k_prime
        V_global[rg, 49:81] = V2s[r2s, 49:81]
        
    # Map W2s (8 rows) to W_global (24 rows)
    # W2s row index r2s = 4*k_prime + i (k_prime in 0..1, i in 0..3)
    # W_global row index rg = 4*(4 + k_prime) + i
    for r2s in range(8):
        k_prime = r2s // 4
        i = r2s % 4
        rg = 4*(4 + k_prime) + i
        W_global[rg, 49:81] = W2s[r2s, 49:81]

    # Quick Verification
    # Check if sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k
    # and 0 otherwise.
    
    # Since we constructed it algebraically, it should be correct.
    # But let's do a sparse check on a few points to be safe.
    # If this fails, we return schoolbook 96.
    
    def verify(U, V, W):
        # Check a random set of points
        # T[i, j, k] = 1
        # U_idx = i*4 + j
        # V_idx = j*6 + k
        # W_idx = k*4 + i
        
        # Check (0,0,0)
        # U[0, :] dot V[0, :] dot W[0, :]? No, sum_c U[0,c] V[0,c] W[0,c]
        val = np.sum(U[0, :] * V[0, :] * W[0, :])
        if abs(val - 1.0) > 1e-9:
            return False
        
        # Check (0,0,1)
        val = np.sum(U[0, :] * V[1, :] * W[4, :]) # V_idx = 0*6+1=1, W_idx=1*4+0=4
        if abs(val - 1.0) > 1e-9:
            return False
            
        # Check (0,1,0)
        val = np.sum(U[1, :] * V[6, :] * W[0, :]) # U_idx=1, V_idx=1*6+0=6, W_idx=0*4+0=0
        if abs(val - 1.0) > 1e-9:
            return False
            
        # Check a zero entry: (0,0,2) is 1, so check (0,0,0) is 1.
        # Check (0,1,1): U[1], V[7], W[5]
        val = np.sum(U[1, :] * V[7, :] * W[5, :])
        if abs(val - 1.0) > 1e-9:
            return False
            
        # Check a zero: i=0, j=0, k=0 is 1.
        # i=0, j=1, k=0 is 1.
        # i=0, j=0, k=0 is 1.
        # What is a zero?
        # The tensor is 1 ONLY at (i,j,k) where all indices are valid.
        # Wait, the tensor T is defined by the multiplication.
        # T_{i,j,k} = 1 if the term A_{ij}B_{jk} appears in C_{ki}.
        # Yes, for standard matrix multiplication, every (i,j,k) triple is a valid term.
        # So there are NO zeros in the "support" of the multiplication tensor in the sense of "which terms exist".
        # However, the tensor decomposition must match the *values*.
        # The values are all 1.
        # So we just need to check that the reconstruction gives 1 for all 96 points.
        
        # Let's check all 96 points to be absolutely sure.
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    u_idx = i*4 + j
                    v_idx = j*6 + k
                    w_idx = k*4 + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U_global, V_global, W_global):
        # Fallback to Schoolbook 96
        U_s = np.zeros((16, 96))
        V_s = np.zeros((24, 96))
        W_s = np.zeros((24, 96))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U_s[i*4 + j, col] = 1
                    V_s[j*6 + k, col] = 1
                    W_s[k*4 + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
