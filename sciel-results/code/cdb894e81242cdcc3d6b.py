import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # We use a block decomposition:
    # A (2x4) = [A_L (2x2), A_R (2x2)]
    # B (4x6) = [B_T (2x6); B_B (2x6)]
    # C = A_L B_T + A_R B_B
    # B_T is split into 3 blocks of 2x2: B_T1, B_T2, B_T3
    # B_B is split into 3 blocks of 2x2: B_B1, B_B2, B_B3
    # C = sum_{i=1 to 3} A_L B_Ti + sum_{j=1 to 3} A_R B_Bj
    # Each 2x2x2 multiplication uses Strassen (rank 7).
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen matrices for 2x2 multiplication
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    # U coefficients for X (2x2)
    # X indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
    # But A is 2x4.
    # For A_L (cols 0,1): indices 0,1,2,3 in A (i*m+j)
    # For A_R (cols 2,3): indices 4,5,6,7 in A
    #   A_L: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    #   A_R: (0,2)->4, (0,3)->5, (1,2)->6, (1,3)->7
    
    strassen_U = [
        [1, 0, 0, 1], # M1: X11+X22
        [0, 0, 1, 1], # M2: X21+X22
        [1, 0, 0, 0], # M3: X11
        [0, 0, 0, 1], # M4: X22
        [1, 1, 0, 0], # M5: X11+X12
        [-1, 0, 1, 0],# M6: X21-X11
        [0, 1, -1, 0] # M7: X12-X22
    ]
    
    # V coefficients for Y (2x2)
    strassen_V = [
        [1, 1, 1, 1], # M1: Y11+Y22
        [1, 0, 0, 0], # M2: Y11
        [0, 1, 0, -1],# M3: Y12-Y22
        [0, -1, 1, 0],# M4: Y21-Y11
        [0, 0, 0, 1], # M5: Y22
        [1, 1, 0, 0], # M6: Y11+Y12
        [0, 0, 1, 1]  # M7: Y21+Y22
    ]
    
    # W coefficients for Z (2x2) output
    # Z11 = M1+M4-M5+M7
    # Z12 = M3+M5
    # Z21 = M2+M4
    # Z22 = M1-M2+M3+M6
    # Z indices: (0,0), (0,1), (1,0), (1,1)
    # Output C is 2x6.
    # For each block of 2 cols in C, the Z matrix contributes.
    # C[i, k] where k is global col.
    # For block 0 (cols 0,1): Z00 -> C[0,0], Z01 -> C[0,1], Z10 -> C[1,0], Z11 -> C[1,1]
    # W[k*n+i, r]
    
    strassen_W = [
        [1, 0, 0, -1], # M1: Z11+1, Z22+1 -> Wait, Z11 has +1, Z22 has +1
        # Let's map carefully.
        # W is (6*2) x 7? No, W is (p*n) x rank.
        # For a specific Strassen call contributing to C[2x2 block],
        # The coefficients for the 4 Z entries are:
        # Z00 (C[0,0]): M1+1, M4+1, M5-1, M7+1
        # Z01 (C[0,1]): M3+1, M5+1
        # Z10 (C[1,0]): M2+1, M4+1
        # Z11 (C[1,1]): M1+1, M2-1, M3+1, M6+1
        
        # We store these as vectors of length 7 for each of the 4 output positions.
        # But W is a single matrix. We will add to the correct rows.
    ]
    
    # Define the 6 Strassen calls
    # Call 0: A_L * B_T1 (Cols 0-1 of B_T)
    # Call 1: A_L * B_T2 (Cols 2-3 of B_T)
    # Call 2: A_L * B_T3 (Cols 4-5 of B_T)
    # Call 3: A_R * B_B1 (Cols 0-1 of B_B)
    # Call 4: A_R * B_B2 (Cols 2-3 of B_B)
    # Call 5: A_R * B_B3 (Cols 4-5 of B_B)
    
    calls = [
        # (A_offset_j, B_offset_j, B_offset_k, C_offset_k)
        # A_offset_j: 0 for A_L, 2 for A_R
        # B_offset_j: 0 for B_T, 2 for B_B
        # B_offset_k: 0, 2, 4 for the 3 blocks of cols
        # C_offset_k: same as B_offset_k
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4)
    ]
    
    col = 0
    for A_off, B_off_j, B_off_k, C_off_k in calls:
        for s in range(7):
            # Fill U
            # U is (2*4) x rank. Index i*m+j.
            # A_L: j in [0,1], A_R: j in [2,3]
            # strassen_U[s] is [X11, X12, X21, X22]
            # X11 = A[0, A_off], X12 = A[0, A_off+1]
            # X21 = A[1, A_off], X22 = A[1, A_off+1]
            
            U[0*m + (A_off+0), col] += strassen_U[s][0]
            U[0*m + (A_off+1), col] += strassen_U[s][1]
            U[1*m + (A_off+0), col] += strassen_U[s][2]
            U[1*m + (A_off+1), col] += strassen_U[s][3]
            
            # Fill V
            # V is (4*6) x rank. Index j*p+k.
            # B_T: j in [0,1], B_B: j in [2,3]
            # B block cols: B_off_k, B_off_k+1
            # strassen_V[s] is [Y11, Y12, Y21, Y22]
            # Y11 = B[B_off_j, B_off_k]
            # Y12 = B[B_off_j, B_off_k+1]
            # Y21 = B[B_off_j+1, B_off_k]
            # Y22 = B[B_off_j+1, B_off_k+1]
            
            V[(B_off_j+0)*p + (B_off_k+0), col] += strassen_V[s][0]
            V[(B_off_j+0)*p + (B_off_k+1), col] += strassen_V[s][1]
            V[(B_off_j+1)*p + (B_off_k+0), col] += strassen_V[s][2]
            V[(B_off_j+1)*p + (B_off_k+1), col] += strassen_V[s][3]
            
            # Fill W
            # W is (6*2) x rank. Index k*n+i.
            # Output C block cols: C_off_k, C_off_k+1
            # Z00 -> C[0, C_off_k]
            # Z01 -> C[0, C_off_k+1]
            # Z10 -> C[1, C_off_k]
            # Z11 -> C[1, C_off_k+1]
            
            # M1
            if s == 0:
                W[(C_off_k+0)*n + 0, col] += 1  # Z11
                W[(C_off_k+0)*n + 1, col] += 0  # Z21 (No, Z21 is M2+M4)
                # Let's just use the explicit formulas for each s to be safe.
                # This is getting messy. Let's define W contributions per s.
            
            # Clear the current col for W and re-add correctly
            # It is easier to define the W coefficients for each s explicitly.
            
            W[(C_off_k+0)*n + 0, col] = 0
            W[(C_off_k+1)*n + 0, col] = 0
            W[(C_off_k+0)*n + 1, col] = 0
            W[(C_off_k+1)*n + 1, col] = 0
            
            if s == 0: # M1
                W[(C_off_k+0)*n + 0, col] += 1 # Z00
                W[(C_off_k+1)*n + 0, col] += 0 # Z01
                W[(C_off_k+0)*n + 1, col] += 0 # Z10
                W[(C_off_k+1)*n + 1, col] += 1 # Z11
            elif s == 1: # M2
                W[(C_off_k+0)*n + 0, col] += 0
                W[(C_off_k+1)*n + 0, col] += 0
                W[(C_off_k+0)*n + 1, col] += 1 # Z10
                W[(C_off_k+1)*n + 1, col] += -1 # Z11
            elif s == 2: # M3
                W[(C_off_k+0)*n + 0, col] += 0
                W[(C_off_k+1)*n + 0, col] += 1 # Z01
                W[(C_off_k+0)*n + 1, col] += 0
                W[(C_off_k+1)*n + 1, col] += 1 # Z11
            elif s == 3: # M4
                W[(C_off_k+0)*n + 0, col] += 1 # Z00
                W[(C_off_k+1)*n + 0, col] += 0
                W[(C_off_k+0)*n + 1, col] += 1 # Z10
                W[(C_off_k+1)*n + 1, col] += 0
            elif s == 4: # M5
                W[(C_off_k+0)*n + 0, col] += -1 # Z00
                W[(C_off_k+1)*n + 0, col] += 1 # Z01
                W[(C_off_k+0)*n + 1, col] += 0
                W[(C_off_k+1)*n + 1, col] += 0
            elif s == 5: # M6
                W[(C_off_k+0)*n + 0, col] += 0
                W[(C_off_k+1)*n + 0, col] += 0
                W[(C_off_k+0)*n + 1, col] += 0
                W[(C_off_k+1)*n + 1, col] += 1 # Z11
            elif s == 6: # M7
                W[(C_off_k+0)*n + 0, col] += 1 # Z00
                W[(C_off_k+1)*n + 0, col] += 0
                W[(C_off_k+0)*n + 1, col] += 0
                W[(C_off_k+1)*n + 1, col] += 0
            
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

