import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_blocks = p // 2
    rank_per_block = 7
    rank = num_blocks * rank_per_block
    
    U = np.zeros((n * m, rank)) # 4 x 42
    V = np.zeros((m * p, rank)) # 24 x 42
    W = np.zeros((p * n, rank)) # 24 x 42
    
    # Standard Strassen coefficients for 2x2x2
    # U_coeffs[t][i, j]
    U_coeffs = [
        np.array([[1, 0], [0, 1]]),  # T0: A00 + A11
        np.array([[0, 0], [1, 1]]),  # T1: A10 + A11
        np.array([[1, 0], [0, 0]]),  # T2: A00
        np.array([[0, 0], [0, 1]]),  # T3: A11
        np.array([[1, 1], [0, 0]]),  # T4: A00 + A01
        np.array([[-1, 0], [1, 0]]), # T5: -A00 + A10
        np.array([[0, 1], [0, -1]])  # T6: A01 - A11
    ]
    
    # V_coeffs[t][j, k]
    V_coeffs = [
        np.array([[1, 0], [0, 1]]),  # T0: B00 + B11
        np.array([[1, 0], [0, 0]]),  # T1: B00
        np.array([[0, 0], [1, 0]]),  # T2: B10
        np.array([[0, 1], [0, -1]]), # T3: B01 - B11
        np.array([[0, 0], [0, 1]]),  # T4: B11
        np.array([[1, 1], [0, 0]]),  # T5: B00 + B01
        np.array([[0, 0], [1, 1]])   # T6: B10 + B11
    ]
    
    # W_coeffs[t][k, i]
    # C_ik += W[k, i] * Product
    W_coeffs = [
        np.array([[1, 0], [0, 1]]),  # T0: C00 += 1, C11 += 1
        np.array([[0, 1], [0, -1]]), # T1: C10 += 1, C11 -= 1
        np.array([[0, 0], [1, 1]]),  # T2: C01 += 1, C11 += 1
        np.array([[1, 1], [0, 0]]),  # T3: C00 += 1, C10 += 1
        np.array([[-1, 0], [1, 0]]), # T4: C00 -= 1, C01 += 1
        np.array([[0, 0], [0, 1]]),  # T5: C11 += 1
        np.array([[1, 0], [0, 0]])   # T6: C00 += 1
    ]

    # Verify Strassen logic for one block
    # C00 = T0 + T3 - T4 + T6 = 1 + 1 - 1 + 1 = 2? 
    # Wait.
    # Standard Strassen:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # My T indices:
    # T0 = P1
    # T1 = P2
    # T2 = P3
    # T3 = P4
    # T4 = P5
    # T5 = P6
    # T6 = P7
    
    # Let's check W_coeffs against this.
    # T0 (P1): C00+=1, C11+=1. Correct.
    # T1 (P2): C10+=1, C11-=1. Correct.
    # T2 (P3): C01+=1, C11+=1. Correct.
    # T3 (P4): C00+=1, C10+=1. Correct.
    # T4 (P5): C00-=1, C01+=1. Correct.
    # T5 (P6): C11+=1. Correct.
    # T6 (P7): C00+=1. Correct.
    
    # Check U and V consistency with P definitions.
    # P1 = (A00+A11)(B00+B11). U0: [[1,0],[0,1]]. V0: [[1,0],[0,1]]. Correct.
    # P2 = (A10+A11)B00. U1: [[0,0],[1,1]]. V1: [[1,0],[0,0]]. Correct.
    # P3 = A00(B10-B00). U2: [[1,0],[0,0]]. V2: [[0,0],[1,0]]. 
    # Wait, V2 in my list is [[0,0],[1,0]]. This is B10. Where is -B00?
    # Ah, V2 should be B10 - B00.
    # My V_coeffs[2] is [[0, 0], [1, 0]]. This is just B10.
    # I missed the -B00 term in V2.
    # Let's re-verify P3.
    # P3 = A00 * (B10 - B00).
    # V3 should be: j=0, k=0 -> -1. j=1, k=0 -> 1.
    # V3 = [[-1, 0], [1, 0]].
    
    # Let's re-verify P4.
    # P4 = A11 * (B01 - B11).
    # V4 should be: j=0, k=1 -> 1. j=1, k=1 -> -1.
    # V4 = [[0, 1], [0, -1]].
    # My V_coeffs[3] is [[0, 1], [0, -1]]. Correct.
    
    # Let's re-verify P5.
    # P5 = (A00+A01) * B11.
    # V5 should be: j=1, k=1 -> 1.
    # V5 = [[0, 0], [0, 1]].
    # My V_coeffs[4] is [[0, 0], [0, 1]]. Correct.
    
    # Let's re-verify P6.
    # P6 = (A10-A00) * (B00+B01).
    # V6 should be: j=0, k=0 -> 1. j=0, k=1 -> 1.
    # V6 = [[1, 1], [0, 0]].
    # My V_coeffs[5] is [[1, 1], [0, 0]]. Correct.
    
    # Let's re-verify P7.
    # P7 = (A01-A11) * (B10+B11).
    # V7 should be: j=1, k=0 -> 1. j=1, k=1 -> 1.
    # V7 = [[0, 0], [1, 1]].
    # My V_coeffs[6] is [[0, 0], [1, 1]]. Correct.
    
    # So only V_coeffs[2] was wrong in my quick derivation above?
    # In the code block above, I wrote:
    # V_coeffs = [
    # ...
    # np.array([[0, 0], [1, 0]]),  # T2: B10
    # ...
    # ]
    # It should be np.array([[-1, 0], [1, 0]]) for T2.
    
    # Let's update V_coeffs[2] in the final code.

    U_coeffs[2] = np.array([[1, 0], [0, 0]]) # A00
    V_coeffs[2] = np.array([[-1, 0], [1, 0]]) # -B00 + B10
    
    # Fill matrices
    for b in range(num_blocks):
        for t in range(rank_per_block):
            r = b * rank_per_block + t
            
            # U: independent of block
            for i in range(n):
                for j in range(m):
                    U[i * m + j, r] = U_coeffs[t][i, j]
                    
            # V: depends on block b
            for j in range(m):
                for k_loc in range(2):
                    k_glob = b * 2 + k_loc
                    V[j * p + k_glob, r] = V_coeffs[t][j, k_loc]
                    
            # W: depends on block b
            for k_loc in range(2):
                k_glob = b * 2 + k_loc
                for i in range(n):
                    W[k_glob * n + i, r] = W_coeffs[t][k_loc, i]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
