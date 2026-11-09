import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 4, 4, 6
    R = 77
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Strassen coefficients for 2x2 matrices
    # alpha[i] corresponds to A-blocks, beta[i] to B-blocks
    # i=0..6
    alpha = [
        [1, 0, 0, 1],  # A00 + A11
        [0, 0, 1, 1],  # A10 + A11
        [1, 0, 0, 0],  # A00
        [0, 0, 0, 1],  # A11
        [1, 1, 0, 0],  # A00 + A01
        [-1, 0, 1, 0], # -A10 + A11  (Note: Strassen M6 is (A11-A10) * B00)
        [0, 1, 0, -1]  # A01 - A11    (Note: Strassen M7 is A11 * (B11-B10))
    ]
    
    beta = [
        [1, 0, 0, 1],  # B00 + B11
        [1, 0, 0, 0],  # B00
        [0, 1, 0, -1], # B10 - B11
        [-1, 0, 1, 0], # -B00 + B01 ? No, standard Strassen varies.
        # Let's use standard Strassen M1..M7:
        # M1 = (A11+A22)(B11+B22) -> alpha[0], beta[0]
        # M2 = (A21+A22)B11       -> alpha[1], beta[1] (A21+A22 = A10+A11? Indices: A11 is 00, A12 is 01, A21 is 10, A22 is 11)
        # Let's map 2x2 block indices:
        # 0: 00, 1: 01, 2: 10, 3: 11
        # M1: A[0]+A[3] * B[0]+B[3]
        # M2: (A[2]+A[3]) * B[0]
        # M3: A[0] * (B[1]-B[3])
        # M4: A[3] * (B[2]-B[0])
        # M5: (A[0]+A[1]) * B[2]
        # M6: (A[3]-A[2]) * B[1]
        # M7: (A[1]-A[3]) * B[2]  <- Wait, standard M7 is (A[1]-A[3]) B[2]?
        # Let's verify standard Strassen reconstruction:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 + M7
        #
        # Alpha (A coeffs):
        # M1: 1, 0, 0, 1
        # M2: 0, 0, 1, 1  (A21+A22 -> indices 2,3)
        # M3: 1, 0, 0, 0
        # M4: 0, 0, 0, 1
        # M5: 1, 1, 0, 0
        # M6: 0, 0, -1, 1 (A22-A21 -> indices 3,2)
        # M7: 0, 1, 0, -1 (A01-A11? No, A12-A22 -> indices 1,3)
        
        # Beta (B coeffs):
        # M1: 1, 0, 0, 1
        # M2: 1, 0, 0, 0
        # M3: 0, 1, 0, -1
        # M4: -1, 0, 1, 0
        # M5: 0, 0, 1, 0
        # M6: 0, 1, 0, 0
        # M7: 0, 0, 1, 0
        
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ]
    
    # Reconstruction coefficients for C blocks
    # C00: M1 + M4 - M5 - M7
    # C01: M3 + M5
    # C10: M2 + M4
    # C11: M1 - M2 + M6 + M7
    recon = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, -1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (5, 1), (6, 1)]
    }

    col = 0
    # Outer loop: Top-level Strassen on 4x4 A?
    # Actually, the current structure is:
    # Split A into 4 2x2 blocks. Use Strassen to get 7 intermediates S1..S7 (2x2 each).
    # For each S_k, we need to multiply by B?
    # No, Strassen combines A-blocks into linear forms, multiplies by B-block linear forms,
    # and reconstructs C.
    # This works for square B. For 4x6 B, we need to adapt.
    
    # The current code's approach:
    # It seems to perform a full Strassen decomposition on the 4x4 part of the tensor?
    # Rank 49 is for 4x4x4.
    # The current code generates 77 columns.
    
    # I will replicate the exact logic of the previous code but refactored.
    
    # Part 1: Inner Strassen for 2x2 x 2x2 logic applied to the 4x4 A and first 4 cols of B?
    # The previous code had a complex nested loop.
    
    # Let's use the explicit construction from the previous code, just cleaned up.
    
    for s in range(7):
        # Part 1: 7x7 terms
        for si in range(7):
            # U
            for a in range(4):
                va = alpha[si][a]
                if va != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        vr = alpha[s][r]
                        if vr != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += va * vr
            # V
            for b in range(4):
                vb = beta[si][b]
                if vb != 0:
                    y_row, y_ic = b // 2, b % 2
                    for c in range(4):
                        vc = beta[s][c]
                        if vc != 0:
                            ac, bc = c // 2, c % 2
                            # v_idx mapping for 4x6 B
                            # The previous code used: v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_ic)
                            # This maps (row 0:2, col 0:3) and (row 2:4, col 3:6)?
                            # bc=0 -> col 0:3 (y_ic 0,1 -> 0,1? No, bc*3 is 0 or 3)
                            # If bc=0, col is y_ic (0 or 1).
                            # If bc=1, col is 3+y_ic (3 or 4).
                            # This only covers cols 0,1,3,4. Missing 2,5.
                            # Wait, p=6. B is 4x6.
                            # The previous code handled the "rest" in Part 2.
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_ic)
                            V[v_idx, col] += vb * vc
            # W
            for (ar_r, br_r, gamma) in [ (r//2, r%2, 1) for r in range(4)]: # Placeholder, needs recon[s]
                pass # Previous code used recon[s] for W
            
            # Correct W logic from previous code:
            # recon[s] gives which C-blocks and coeffs.
            # For the inner si, it also gives structure.
            # The previous code:
            # for a_r, b_r, gamma in recon[s]:
            #    for za, zb, zgamma in recon[si]:
            #        w_idx = (b_r * 3 + zb) * 4 + (a_r * 2 + za)
            #        W[w_idx, col] += gamma * zgamma
            
            # I need `recon` for `s` and `si`.
            # The `recon` defined above is for 2x2 Strassen.
            # `s` is outer Strassen index, `si` is inner Strassen index.
            # This implies the previous code was doing something like "Strassen on Strassen"?
            # Or just double Strassen for 4x4x4 part?
            
            col += 1
            
        # Part 2: Remaining columns for B
        # Previous code: 2x2 schoolbook for the rest?
        # It iterated u in 0..1, v in 0..1. 4 cols.
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    vr = alpha[s][r]
                    if vr != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += vr
                for c in range(4):
                    vc = beta[s][c]
                    if vc != 0:
                        ac, bc = c // 2, c % 2
                        # Previous mapping for V in Part 2:
                        # v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        # bc=0 -> col 2. bc=1 -> col 5.
                        # Covers cols 2 and 5.
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += vc
                for a_r, b_r, gamma in [(r//2, r%2, 1) for r in range(4)]: # Placeholder
                    pass
                # Correct W logic from previous code:
                # for a_r, b_r, gamma in recon[s]:
                #     w_idx = (b_r * 3 + 2) * 4 + (a_r * 2 + u)
                #     W[w_idx, col] += gamma
                # Note: The col index for B in W logic is fixed to 2?
                # And u/v determine the row/col in C?
                
                col += 1

    # This reconstruction of the previous code's logic is tricky without the original `recon` definition
    # which was local.
    # I will define `recon` locally and use it.

    return {"U": U, "V": V, "W": W} # Return numpy arrays? The task says list?
    # Task says: "factor matrices of shapes... with integer/half-integer entries"
    # "Interface: construct() -> {'U', 'V', 'W'}"
    # Usually lists are safer if strict type checking, but numpy arrays are often accepted.
    # The previous code converted to list with `to_list`. I will do the same.

def to_list(mat):
    if isinstance(mat, np.ndarray):
        return mat.tolist()
    return mat.tolist()

# Refined construct
def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    alpha = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=np.int8)
    
    beta = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.int8)
    
    # Recon for 2x2 Strassen: (row_block, col_block, coeff)
    # M1=0, M2=1, M3=2, M4=3, M5=4, M6=5, M7=6
    recon = [
        [(0, 0, 1), (3, 3, -1)], # M1: C00+=1, C11+=-1? No.
        # Standard:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 + M7
        # So for index k (0..6):
        # k=0 (M1): C00 +1, C11 +1
        # k=1 (M2): C10 +1, C11 -1
        # k=2 (M3): C01 +1
        # k=3 (M4): C00 +1, C10 +1
        # k=4 (M5): C00 -1, C01 +1
        # k=5 (M6): C11 +1
        # k=6 (M7): C00 -1, C11 +1
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,-1), (1,1,1)]
    ]
    
    col = 0
    for s in range(7):
        for si in range(7):
            # U
            for a in range(4):
                va = alpha[si, a]
                if va != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        vr = alpha[s, r]
                        if vr != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += va * vr
            # V
            for b in range(4):
                vb = beta[si, b]
                if vb != 0:
                    y_row, y_ic = b // 2, b % 2
                    for c in range(4):
                        vc = beta[s, c]
                        if vc != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_ic)
                            V[v_idx, col] += vb * vc
            # W
            for a_r, b_r, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b_r * 3 + zb) * 4 + (a_r * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1
            
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    vr = alpha[s, r]
                    if vr != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += vr
                for c in range(4):
                    vc = beta[s, c]
                    if vc != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += vc
                for a_r, b_r, gamma in recon[s]:
                    w_idx = (b_r * 3 + 2) * 4 + (a_r * 2 + u)
                    W[w_idx, col] += gamma
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
