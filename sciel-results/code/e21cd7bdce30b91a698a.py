import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We construct a rank 32 decomposition.
    # Split j index: j=0 (rank 14) and j in {1,2} (rank 18).
    
    R_total = 32
    
    U = np.zeros((n * m, R_total))
    V = np.zeros((m * p, R_total))
    W = np.zeros((p * n, R_total))
    
    # --- Part 1: j = 0 ---
    # A is 2x3. A[:, 0] is 2x1.
    # B is 3x7. B[0, :] is 1x7.
    # C contribution: A[:, 0] * B[0, :]
    # This is a 2x7 matrix.
    # Schoolbook decomposition: 14 terms.
    # Each term corresponds to (i, 0, k).
    
    r_offset = 0
    for i in range(n):
        for k in range(p):
            # Term r = r_offset
            U[i * m + 0, r_offset] = 1.0
            V[0 * p + k, r_offset] = 1.0
            W[k * n + i, r_offset] = 1.0
            r_offset += 1
            
    # --- Part 2: j in {1, 2} ---
    # A_L is A[:, 1:3] (2x2).
    # B_T is B[1:3, :] (2x7).
    # Split B_T columns:
    # B_t1: cols 0,1 (of B_T) -> global cols 0,1
    # B_t2: cols 2,3 (of B_T) -> global cols 2,3
    # B_t3: col 4 (of B_T) -> global col 4
    # Wait, B is 3x7.
    # B_T is rows 1,2.
    # B_T[:, 0:2] is 2x2.
    # B_T[:, 2:4] is 2x2.
    # B_T[:, 4:5] is 2x1.
    # Missing col 5,6?
    # B has 7 columns.
    # B_T has 7 columns.
    # Split into: (0,1), (2,3), (4,5), (6).
    # Ranks: 7 + 7 + 7 + 4 = 25.
    # Total rank = 14 + 25 = 39.
    
    # Let's try the previous split:
    # j=0: Rank 14.
    # j in {1,2}:
    # B_T (2x7).
    # Split into (0,1), (2,3), (4,5,6)?
    # 2x2 (Rank 7) + 2x2 (Rank 7) + 2x3 (Rank ?)
    # 2x3 matrix multiplication (A_L * B_3) where B_3 is 2x3.
    # Rank(2x2x3) is at most 6? No, 2*2*3=12 schoolbook.
    # Strassen on 2x2 blocks?
    # B_3 = [B_31 (2x2) | B_32 (2x1)].
    # Rank(2x2x2) = 7.
    # Rank(2x2x1) = 4.
    # Total for 2x2x3 is 11.
    # So j in {1,2} rank = 7 (for cols 0,1) + 11 (for cols 2,3,4,5,6? No, 2,3,4,5,6 is 5 cols).
    
    # Let's re-calculate carefully.
    # B_T is 2x7.
    # Split into 2x2, 2x2, 2x2, 2x1.
    # Ranks: 7, 7, 7, 4. Sum = 25.
    # Total 14 + 25 = 39.
    
    # Can we do better for j in {1,2}?
    # What if we use Strassen on the whole 2x2x7?
    # Rank(2x2x7) is not simply 7*7/2.
    # It is at most 7 * Rank(2x2x1)? No.
    # It is bounded by 7 * Rank(2x2x1) if we treat it as 7 independent 2x2x1?
    # No, Strassen shares linear combinations.
    # Rank(2x2xp) <= 7 * ceil(p/2)?
    # For p=7, 7 * 4 = 28?
    # Or 7 * 3 + 4 = 25.
    
    # Let's stick with 39. It is better than 42.
    # Is there a rank 30 construction?
    # Split j=0,1,2.
    # j=0: Rank 7? (1x1x7). No, 2x1x7 is 14.
    # If n=1, rank is 7. But n=2.
    
    # Let's implement the 39 rank construction.
    
    R_total = 39
    U = np.zeros((n * m, R_total))
    V = np.zeros((m * p, R_total))
    W = np.zeros((p * n, R_total))
    
    r_offset = 0
    
    # Part 1: j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0, r_offset] = 1.0
            V[0 * p + k, r_offset] = 1.0
            W[k * n + i, r_offset] = 1.0
            r_offset += 1
            
    # Part 2: j in {1, 2}
    # A_L indices: i in {0,1}, j in {1,2}
    # B_T indices: j in {1,2}, k in {0..6}
    
    # Block 1: B_T[:, 0:2] (cols 0,1)
    # Strassen on 2x2 * 2x2
    # A_L = [A01 A02; A11 A12]
    # B_B1 = [B10 B11; B20 B21]
    
    # Strassen formulas:
    # M1 = (A01+A12) * (B10+B21)
    # M2 = (A11+A12) * B10
    # M3 = A01 * (B11-B21)
    # M4 = A12 * (B20+B21)
    # M5 = (A01+A02) * B21
    # M6 = (A11-A01) * (B10+B11)
    # M7 = (A02-A12) * (B20+B21)
    
    # We map local indices (li, lj) to global (i, j)
    # Local i=0 -> Global i=0
    # Local i=1 -> Global i=1
    # Local j=0 -> Global j=1
    # Local j=1 -> Global j=2
    
    # Local k=0 -> Global k=0
    # Local k=1 -> Global k=1
    
    def add_strassen_block(global_i_offset, global_j_offset, global_k_offset, r_start):
        # global_i_offset is 0 (since n=2, local i is 0,1 -> global 0,1)
        # global_j_offset is 1 (since local j is 0,1 -> global 1,2)
        # global_k_offset is start col
        
        # M1
        # (A01+A12) * (B10+B21)
        # U: A01, A12
        # V: B10, B21
        # W: Output C00, C01, C10, C11
        
        # Term 1: (A01+A12) * (B10+B21)
        # U[0, j=0] (A01) coeff 1. U[1, j=1] (A12) coeff 1.
        # V[j=0, k=0] (B10) coeff 1. V[j=1, k=1] (B21) coeff 1.
        # W[k=0, i=0] (C00) coeff 1. W[k=1, i=0] (C01) coeff -1.
        # W[k=0, i=1] (C10) coeff 1. W[k=1, i=1] (C11) coeff -1.
        
        # Note: W index is k*n + i.
        
        # r = r_start
        U[0*m + (global_j_offset+0), r_start] = 1.0  # A01
        U[1*m + (global_j_offset+1), r_start] = 1.0  # A12
        V[(global_j_offset+0)*p + (global_k_offset+0), r_start] = 1.0 # B10
        V[(global_j_offset+1)*p + (global_k_offset+1), r_start] = 1.0 # B21
        W[(global_k_offset+0)*n + 0, r_start] = 1.0  # C00
        W[(global_k_offset+1)*n + 0, r_start] = -1.0 # C01
        W[(global_k_offset+0)*n + 1, r_start] = 1.0  # C10
        W[(global_k_offset+1)*n + 1, r_start] = -1.0 # C11
        r_start += 1
        
        # M2: (A11+A12) * B10
        # U: A11, A12. V: B10.
        # W: C10, C11
        U[1*m + (global_j_offset+0), r_start] = 1.0 # A11
        U[1*m + (global_j_offset+1), r_start] = 1.0 # A12
        V[(global_j_offset+0)*p + (global_k_offset+0), r_start] = 1.0 # B10
        W[(global_k_offset+0)*n + 1, r_start] = 1.0 # C10
        W[(global_k_offset+1)*n + 1, r_start] = 1.0 # C11
        r_start += 1
        
        # M3: A01 * (B11-B21)
        U[0*m + (global_j_offset+0), r_start] = 1.0 # A01
        V[(global_j_offset+0)*p + (global_k_offset+1), r_start] = 1.0 # B11
        V[(global_j_offset+1)*p + (global_k_offset+1), r_start] = -1.0 # B21
        W[(global_k_offset+1)*n + 0, r_start] = 1.0 # C01
        W[(global_k_offset+1)*n + 1, r_start] = 1.0 # C11
        r_start += 1
        
        # M4: A12 * (B20+B21)
        U[1*m + (global_j_offset+1), r_start] = 1.0 # A12
        V[(global_j_offset+1)*p + (global_k_offset+0), r_start] = 1.0 # B20
        V[(global_j_offset+1)*p + (global_k_offset+1), r_start] = 1.0 # B21
        W[(global_k_offset+0)*n + 1, r_start] = 1.0 # C10
        W[(global_k_offset+1)*n + 1, r_start] = 1.0 # C11
        r_start += 1
        
        # M5: (A01+A02) * B21
        U[0*m + (global_j_offset+0), r_start] = 1.0 # A01
        U[0*m + (global_j_offset+1), r_start] = 1.0 # A02
        V[(global_j_offset+1)*p + (global_k_offset+1), r_start] = 1.0 # B21
        W[(global_k_offset+1)*n + 0, r_start] = 1.0 # C01
        r_start += 1
        
        # M6: (A11-A01) * (B10+B11)
        U[1*m + (global_j_offset+0), r_start] = 1.0 # A11
        U[0*m + (global_j_offset+0), r_start] = -1.0 # A01
        V[(global_j_offset+0)*p + (global_k_offset+0), r_start] = 1.0 # B10
        V[(global_j_offset+0)*p + (global_k_offset+1), r_start] = 1.0 # B11
        W[(global_k_offset+0)*n + 1, r_start] = 1.0 # C10
        W[(global_k_offset+1)*n + 1, r_start] = 1.0 # C11
        r_start += 1
        
        # M7: (A02-A12) * (B20+B21)
        U[0*m + (global_j_offset+1), r_start] = 1.0 # A02
        U[1*m + (global_j_offset+1), r_start] = -1.0 # A12
        V[(global_j_offset+1)*p + (global_k_offset+0), r_start] = 1.0 # B20
        V[(global_j_offset+1)*p + (global_k_offset+1), r_start] = 1.0 # B21
        W[(global_k_offset+0)*n + 0, r_start] = 1.0 # C00
        W[(global_k_offset+1)*n + 0, r_start] = 1.0 # C01
        r_start += 1
        
        return r_start

    # Block 1: cols 0,1
    r_offset = add_strassen_block(0, 1, 0, r_offset)
    # Block 2: cols 2,3
    r_offset = add_strassen_block(0, 1, 2, r_offset)
    # Block 3: cols 4,5
    r_offset = add_strassen_block(0, 1, 4, r_offset)
    
    # Block 4: col 6 (2x2x1)
    # A_L * B_col6
    # C06 = A01*B16 + A02*B26
    # C16 = A11*B16 + A12*B26
    
    # Schoolbook 4 terms:
    # Term 1: A01 * B16 -> C06
    U[0*m + (1+0), r_offset] = 1.0 # A01
    V[(1+0)*p + 6, r_offset] = 1.0 # B16
    W[6*n + 0, r_offset] = 1.0 # C06
    r_offset += 1
    
    # Term 2: A02 * B26 -> C06
    U[0*m + (1+1), r_offset] = 1.0 # A02
    V[(1+1)*p + 6, r_offset] = 1.0 # B26
    W[6*n + 0, r_offset] = 1.0 # C06
    r_offset += 1
    
    # Term 3: A11 * B16 -> C16
    U[1*m + (1+0), r_offset] = 1.0 # A11
    V[(1+0)*p + 6, r_offset] = 1.0 # B16
    W[6*n + 1, r_offset] = 1.0 # C16
    r_offset += 1
    
    # Term 4: A12 * B26 -> C16
    U[1*m + (1+1), r_offset] = 1.0 # A12
    V[(1+1)*p + 6, r_offset] = 1.0 # B26
    W[6*n + 1, r_offset] = 1.0 # C16
    r_offset += 1
    
    assert r_offset == R_total, f"Rank mismatch: {r_offset} != {R_total}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
