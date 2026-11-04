"""Direct approach for 6x6 matrix multiplication tensor decomposition using Kronecker product of Strassen 2x2 and Schoolbook 3x3."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # 1. Build Strassen 2x2 decomposition
    # Rank 7
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # Indices for 2x2:
    # 0: 11, 1: 12, 2: 21, 3: 22
    
    # k=0: M1
    # A: 11(1), 22(1) -> idx 0, 3
    # B: 11(1), 22(1) -> idx 0, 3
    # C: 11(1), 12(1), 21(1), 22(1) with coeffs from combination
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define the coefficients for A, B, C for each of the 7 multiplications
    # A_coeffs[k, idx_A]
    # B_coeffs[k, idx_B]
    # C_coeffs[k, idx_C]
    
    A_coeffs = np.zeros((7, 4), dtype=int)
    B_coeffs = np.zeros((7, 4), dtype=int)
    C_coeffs = np.zeros((7, 4), dtype=int)
    
    # k=0: M1 = (A11 + A22)(B11 + B22)
    A_coeffs[0, 0] = 1; A_coeffs[0, 3] = 1
    B_coeffs[0, 0] = 1; B_coeffs[0, 3] = 1
    C_coeffs[0, 0] = 1; C_coeffs[0, 3] = 1 # M1 contributes to C11 and C22
    # Wait, C_coeffs should reflect the final C combination.
    # It's easier to define the 7 intermediate multiplications and then the final combination.
    # But the tensor decomposition is linear.
    # T = sum_k (sum_a alpha_ka A_a) (sum_b beta_kb B_b) placed in C via gamma_ck.
    # This is exactly what the Kronecker product of the "block" tensor and "local" tensor does.
    # The "block" tensor is the tensor of the 2x2 matrix multiplication defined by Strassen.
    # The "local" tensor is the tensor of the 3x3 matrix multiplication (schoolbook).
    
    # So we just need the U2, V2, W2 for the Strassen 2x2 decomposition.
    # And U3, V3, W3 for the Schoolbook 3x3 decomposition.
    
    # Strassen 2x2 U, V, W:
    # Each of the 7 terms is a rank-1 tensor in the 2x2 space.
    # Term k is defined by A_vec, B_vec, C_vec.
    # A_vec has non-zeros at the blocks involved.
    
    # k=0
    A_coeffs[0, 0] = 1; A_coeffs[0, 3] = 1
    B_coeffs[0, 0] = 1; B_coeffs[0, 3] = 1
    # C for M1 is used in C11 and C22.
    # C11 += M1, C22 += M1
    # So C_vec for this term in the 2x2 output space:
    # C11(1), C22(1)
    C_coeffs[0, 0] = 1; C_coeffs[0, 3] = 1
    
    # k=1
    A_coeffs[1, 2] = 1; A_coeffs[1, 3] = 1
    B_coeffs[1, 0] = 1
    C_coeffs[1, 2] = 1 # C21 += M2
    # Wait, C21 is index 2.
    # C22 -= M2
    C_coeffs[1, 3] = -1
    
    # k=2
    A_coeffs[2, 0] = 1
    B_coeffs[2, 1] = 1; B_coeffs[2, 3] = -1
    C_coeffs[2, 1] = 1 # C12 += M3
    C_coeffs[2, 3] = 1 # C22 += M3
    
    # k=3
    A_coeffs[3, 3] = 1
    B_coeffs[3, 2] = 1; B_coeffs[3, 0] = -1
    C_coeffs[3, 0] = 1 # C11 += M4
    C_coeffs[3, 2] = 1 # C21 += M4
    
    # k=4
    A_coeffs[4, 0] = 1; A_coeffs[4, 1] = 1
    B_coeffs[4, 3] = 1
    C_coeffs[4, 0] = -1 # C11 -= M5
    C_coeffs[4, 1] = 1 # C12 += M5
    
    # k=5
    A_coeffs[5, 2] = 1; A_coeffs[5, 0] = -1
    B_coeffs[5, 0] = 1; B_coeffs[5, 1] = 1
    C_coeffs[5, 3] = 1 # C22 += M6
    
    # k=6
    A_coeffs[6, 1] = 1; A_coeffs[6, 3] = -1
    B_coeffs[6, 2] = 1; B_coeffs[6, 3] = 1
    C_coeffs[6, 0] = 1 # C11 += M7
    
    # Fill U2, V2, W2
    for k in range(7):
        for idx in range(4):
            U2[idx, k] = A_coeffs[k, idx]
            V2[idx, k] = B_coeffs[k, idx]
            W2[idx, k] = C_coeffs[k, idx]
            
    # 2. Build Schoolbook 3x3 decomposition
    # Rank 27
    # U3, V3, W3 are 9x27
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                U3[i * 3 + j, r] = 1
                V3[j * 3 + k, r] = 1
                W3[i * 3 + k, r] = 1
                
    # 3. Construct 6x6 decomposition
    # Rank = 7 * 27 = 189
    rank = 189
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)
    
    for r2 in range(7):
        for r3 in range(27):
            r = r2 * 27 + r3
            
            # Iterate over non-zero entries in U2[:, r2] and U3[:, r3]
            # U2 has at most 2 non-zeros, U3 has 1.
            # So at most 2 non-zeros in U6 for this r.
            
            for idx2 in range(4):
                if U2[idx2, r2] != 0:
                    val2 = U2[idx2, r2]
                    i2 = idx2 // 2
                    j2 = idx2 % 2
                    for idx3 in range(9):
                        if U3[idx3, r3] != 0:
                            val3 = U3[idx3, r3]
                            i3 = idx3 // 3
                            j3 = idx3 % 3
                            
                            iA = i2 * 3 + i3
                            jA = j2 * 3 + j3
                            idxA = iA * 6 + jA
                            
                            U6[idxA, r] = val2 * val3
                            
                            # V
                            # V2 idx must have same row block (j2) as U2
                            # V2 idx2_v = j2 * 2 + k2
                            for idx2_v in range(4):
                                if V2[idx2_v, r2] != 0:
                                    val2_v = V2[idx2_v, r2]
                                    j2_v = idx2_v // 2
                                    k2 = idx2_v % 2
                                    if j2_v == j2:
                                        for idx3_v in range(9):
                                            if V3[idx3_v, r3] != 0:
                                                val3_v = V3[idx3_v, r3]
                                                j3_v = idx3_v // 3
                                                k3 = idx3_v % 3
                                                if j3_v == j3:
                                                    jB = j2 * 3 + j3
                                                    kB = k2 * 3 + k3
                                                    idxB = jB * 6 + kB
                                                    V6[idxB, r] = val2_v * val3_v
                                                    
                            # W
                            # W2 idx must have same row block (i2) as U2
                            # W2 idx2_w = i2 * 2 + k2
                            for idx2_w in range(4):
                                if W2[idx2_w, r2] != 0:
                                    val2_w = W2[idx2_w, r2]
                                    i2_w = idx2_w // 2
                                    k2 = idx2_w % 2
                                    if i2_w == i2:
                                        for idx3_w in range(9):
                                            if W3[idx3_w, r3] != 0:
                                                val3_w = W3[idx3_w, r3]
                                                i3_w = idx3_w // 3
                                                k3 = idx3_w % 3
                                                if i3_w == i3:
                                                    iC = i2 * 3 + i3
                                                    kC = k2 * 3 + k3
                                                    idxC = iC * 6 + kC
                                                    W6[idxC, r] = val2_w * val3_w

    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
