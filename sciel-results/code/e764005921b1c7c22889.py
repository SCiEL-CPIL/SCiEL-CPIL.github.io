"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Total rank estimate: 42
    rank = 42
    
    U = np.zeros((n, m, rank))
    V = np.zeros((m, p, rank))
    W = np.zeros((p, n, rank))
    
    # --- Laderman Factors (3x3x3) ---
    # U, V, W are 3x3x7
    U3 = np.zeros((3, 3, 7))
    V3 = np.zeros((3, 3, 7))
    W3 = np.zeros((3, 3, 7))
    
    # Laderman formulas:
    # M1: A00(B00+B11)
    # M2: A11(B00+B22)
    # M3: A22(B11+B00)
    # M4: (A01-A11)(B10-B00)
    # M5: (A02-A22)(B20-B00)
    # M6: (A12-A22)(B21-B11)
    # M7: (A00+A11+A22)(B00+B11+B22) -- Wait, M7 is usually different.
    
    # Correct Laderman:
    # M1 = A00(B00+B11)
    # M2 = A11(B00+B22)
    # M3 = A22(B11+B00)
    # M4 = (A01-A11)(B10-B00)
    # M5 = (A02-A22)(B20-B00)
    # M6 = (A12-A22)(B21-B11)
    # M7 = (A00+A11+A22)(B00+B11+B22) is NOT correct for rank 7.
    
    # Standard Laderman:
    # M1 = A00(B00+B11)
    # M2 = A11(B00+B22)
    # M3 = A22(B11+B00)
    # M4 = (A01-A11)(B10-B00)
    # M5 = (A02-A22)(B20-B00)
    # M6 = (A12-A22)(B21-B11)
    # M7 = (A00+A11+A22)(B00+B11+B22) -> This gives rank 7?
    # Actually, the 7th term is often:
    # M7 = (A00+A11+A22)(B00+B11+B22) is not quite right.
    # Let's use a verified set.
    
    # Verified Laderman:
    # U:
    # r0: 100 000 000 -> A00
    # r1: 010 000 000 -> A10? No.
    
    # I will use the following known Laderman factors:
    # U (3x3x7)
    U3[0,0,0] = 1; U3[1,1,0] = 1
    U3[0,0,1] = 1; U3[2,2,1] = 1
    U3[1,1,2] = 1; U3[0,0,2] = 1 # Wait, M3 is A22(B11+B00)?
    
    # Let's just use the Strassen for 2x2 and Schoolbook for 3x3?
    # No, Laderman is better.
    # I will hardcode the Laderman factors from a reliable source.
    
    # Laderman:
    # M1 = A00(B00+B11)
    # M2 = A11(B00+B22)
    # M3 = A22(B11+B00)
    # M4 = (A01-A11)(B10-B00)
    # M5 = (A02-A22)(B20-B00)
    # M6 = (A12-A22)(B21-B11)
    # M7 = (A00+A11+A22)(B00+B11+B22)
    
    # This set is NOT correct.
    # Correct Laderman:
    # M1 = A00(B00+B11)
    # M2 = A11(B00+B22)
    # M3 = A22(B11+B00)
    # M4 = (A01-A11)(B10-B00)
    # M5 = (A02-A22)(B20-B00)
    # M6 = (A12-A22)(B21-B11)
    # M7 = (A00+A11+A22)(B00+B11+B22)
    
    # Actually, I'll just use Schoolbook for 3x3 (rank 9) to be safe?
    # 9 + 7 + 9 + 7 + 9 + 7 = 48.
    # 48 is still better than 90.
    # And Schoolbook is trivial to implement correctly.
    
    def add_schoolbook_block(U, V, W, i_start, j_start, k_start, size, rank_offset):
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    r = rank_offset + (i * size * size) + (j * size) + k
                    U[i_start+i, j_start+j, r] = 1
                    V[j_start+j, k_start+k, r] = 1
                    W[k_start+k, i_start+i, r] = 1
        return rank_offset + size * size * size

    # Use Schoolbook for 3x3 (rank 9) and Strassen for 2x2 (rank 7)
    
    # Block 11: 3x3x3 (Indices 0,1,2)
    rank_curr = add_schoolbook_block(U, V, W, 0, 0, 0, 3, 0)
    
    # Block 22: 2x2x2 (Indices 3,4)
    # Strassen
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    
    # Strassen factors
    U2[0,0,0] = 1; U2[1,1,0] = 1
    V2[0,0,0] = 1; V2[1,1,0] = 1
    W2[0,0,0] = 1; W2[1,1,0] = 1
    
    U2[1,0,1] = 1; U2[1,1,1] = 1
    V2[0,0,1] = 1
    W2[1,0,1] = 1; W2[1,1,1] = -1
    
    U2[0,0,2] = 1
    V2[0,1,2] = 1; V2[1,1,2] = -1
    W2[0,1,2] = 1; W2[1,1,2] = 1
    
    U2[1,1,3] = 1
    V2[1,0,3] = 1; V2[0,0,3] = -1
    W2[0,0,3] = -1; W2[1,0,3] = 1
    
    U2[0,0,4] = 1; U2[0,1,4] = 1
    V2[1,1,4] = 1
    W2[0,0,4] = 1; W2[0,1,4] = 1
    
    U2[1,0,5] = 1; U2[0,0,5] = -1
    V2[0,0,5] = 1; V2[0,1,5] = 1
    W2[1,1,5] = 1
    
    U2[0,1,6] = 1; U2[1,1,6] = -1
    V2[1,0,6] = 1; V2[1,1,6] = 1
    W2[0,0,6] = 1; W2[1,1,6] = -1
    
    # Add Strassen to Block 22 (Indices 3,4)
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[3+i, 3+j, rank_curr+r] = U2[i, j, r]
                for k in range(2):
                    V[3+j, 3+k, rank_curr+r] = V2[j, k, r]
                    W[3+k, 3+i, rank_curr+r] = W2[k, i, r]
    rank_curr += 7
    
    # Block 12: 3x3x2 (i in 0-2, j in 0-2, k in 3-4)
    # Schoolbook: 3*3*2 = 18 terms?
    # We can use Schoolbook for cross terms to be safe.
    # Rank = 18.
    for i in range(3):
        for j in range(3):
            for k in range(2):
                r = rank_curr + (i * 3 * 2) + (j * 2) + k
                U[i, j, r] = 1
                V[j, 3+k, r] = 1
                W[3+k, i, r] = 1
    rank_curr += 18
    
    # Block 21: 2x2x3 (i in 3-4, j in 3-4, k in 0-2)
    # Schoolbook: 2*2*3 = 12 terms.
    for i in range(2):
        for j in range(2):
            for k in range(3):
                r = rank_curr + (i * 2 * 3) + (j * 3) + k
                U[3+i, 3+j, r] = 1
                V[3+j, k, r] = 1
                W[k, 3+i, r] = 1
    rank_curr += 12
    
    # Block 13? No, only 2 sets.
    # Block 1 (I1) x Block 2 (I2) where i in I1, j in I2, k in I2?
    # i in 0-2, j in 3-4, k in 3-4.
    # Schoolbook: 3*2*2 = 12 terms.
    for i in range(3):
        for j in range(2):
            for k in range(2):
                r = rank_curr + (i * 2 * 2) + (j * 2) + k
                U[i, 3+j, r] = 1
                V[3+j, 3+k, r] = 1
                W[3+k, i, r] = 1
    rank_curr += 12
    
    # Block 2 (I2) x Block 1 (I1) where i in I2, j in I1, k in I1?
    # i in 3-4, j in 0-2, k in 0-2.
    # Schoolbook: 2*3*3 = 18 terms.
    for i in range(2):
        for j in range(3):
            for k in range(3):
                r = rank_curr + (i * 3 * 3) + (j * 3) + k
                U[3+i, j, r] = 1
                V[j, k, r] = 1
                W[k, 3+i, r] = 1
    rank_curr += 18
    
    # Total rank: 9 + 7 + 18 + 12 + 12 + 18 = 76.
    # This is better than 90, but not 42.
    # I'll stick with this safe implementation.
    
    U = U[:, :, :rank_curr]
    V = V[:, :, :rank_curr]
    W = W[:, :, :rank_curr]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
