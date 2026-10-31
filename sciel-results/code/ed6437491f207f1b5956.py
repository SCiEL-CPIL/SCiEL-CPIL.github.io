import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # U, V, W will be of shape (16, 49)
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Helper to get index in flattened 4x4 matrix
    def idx4(r, c):
        return r * 4 + c
        
    # Helper to get index in flattened 2x2 matrix (local)
    # But we need to map to global 4x4 indices.
    
    # Define the 7 Strassen terms for 2x2 matrices.
    # Each term is defined by:
    # A-part: list of (row, col, coeff) in local 2x2 coords
    # B-part: list of (row, col, coeff) in local 2x2 coords
    # C-part: list of (row, col, coeff) in local 2x2 coords (contribution to C)
    
    # Strassen 2x2 terms:
    # M1 = (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
    # M2 = (A21+A22)B11      -> C21 += 1
    # M3 = A11(B12-B22)      -> C12 += 1, C22 -= 1
    # M4 = A22(B21-B11)      -> C11 += 1, C21 += 1
    # M5 = (A11+A12)B22      -> C12 += 1, C11 -= 1
    # M6 = (A21-A11)(B11+B12)-> C22 += 1
    # M7 = (A12-A22)(B21+B22)-> C22 += 1, C11 -= 1
    
    # Let's refine the C contributions based on:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 + M7
    
    strassen_2x2 = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: list of (r, c, val)
        (
            [(0,0,1), (1,1,1)], 
            [(0,0,1), (1,1,1)], 
            [(0,0,1), (1,1,1)]
        ),
        (
            [(1,0,1), (1,1,1)], 
            [(0,0,1)], 
            [(1,0,1)]
        ),
        (
            [(0,0,1)], 
            [(0,1,1), (1,1,-1)], 
            [(0,1,1), (1,1,-1)]
        ),
        (
            [(1,1,1)], 
            [(1,0,1), (0,0,-1)], 
            [(0,0,1), (1,0,1)]
        ),
        (
            [(0,0,1), (0,1,1)], 
            [(1,1,1)], 
            [(0,1,1), (0,0,-1)]
        ),
        (
            [(1,0,1), (0,0,-1)], 
            [(0,0,1), (0,1,1)], 
            [(1,1,1)]
        ),
        (
            [(0,1,1), (1,1,-1)], 
            [(1,0,1), (1,1,1)], 
            [(1,1,1), (0,0,-1)]
        ),
    ]
    
    # Outer Strassen 4x4 structure:
    # We treat 4x4 as 2x2 blocks of 2x2 matrices.
    # Block indices:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # The 7 outer terms P1..P7 are products of 2x2 matrices.
    # Each P_k is formed by linear combinations of the 4x4 blocks.
    # We need to express the A-part and B-part of each P_k in terms of 4x4 entries.
    
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # For each P_k, we will apply the 2x2 Strassen algorithm.
    # This means we need to identify the "local" 2x2 entries of the matrices being multiplied in P_k.
    # Let X_k and Y_k be the two 2x2 matrices in P_k = X_k Y_k.
    # The entries of X_k are linear forms in the 4x4 blocks of A.
    # The entries of Y_k are linear forms in the 4x4 blocks of B.
    
    # We will iterate over the 7 outer terms.
    # For each outer term, we define the mapping from local 2x2 coords to global 4x4 coords for X_k and Y_k.
    
    # Outer Term 1: X = A11 + A22, Y = B11 + B22
    # X[0,0] = A[0,0] + A[2,2]
    # X[0,1] = A[0,1] + A[2,3]
    # X[1,0] = A[1,0] + A[3,2]
    # X[1,1] = A[1,1] + A[3,3]
    
    # We represent this as a list of 4 entries, each being a list of (global_row, global_col, coeff).
    
    def get_block_entries(block_type, offset_r, offset_c):
        # Returns list of 4 entries, each is a list of (r, c, coeff)
        # block_type: 'A' or 'B'
        # This is just a helper to think clearly, we'll hardcode the lists.
        pass

    # Outer Term 1: X = A11 + A22
    X1 = [
        [(0,0,1), (2,2,1)], # X[0,0]
        [(0,1,1), (2,3,1)], # X[0,1]
        [(1,0,1), (3,2,1)], # X[1,0]
        [(1,1,1), (3,3,1)]  # X[1,1]
    ]
    Y1 = [
        [(0,0,1), (2,2,1)], # Y[0,0]
        [(0,1,1), (2,3,1)], # Y[0,1]
        [(1,0,1), (3,2,1)], # Y[1,0]
        [(1,1,1), (3,3,1)]  # Y[1,1]
    ]
    
    # Outer Term 2: X = A21 + A22, Y = B11
    X2 = [
        [(2,0,1), (2,2,1)], # X[0,0] (A21[0,0] is A[2,0], A22[0,0] is A[2,2])
        [(2,1,1), (2,3,1)], # X[0,1]
        [(3,0,1), (3,2,1)], # X[1,0]
        [(3,1,1), (3,3,1)]  # X[1,1]
    ]
    Y2 = [
        [(0,0,1)], # Y[0,0]
        [(0,1,1)], # Y[0,1]
        [(1,0,1)], # Y[1,0]
        [(1,1,1)]  # Y[1,1]
    ]
    
    # Outer Term 3: X = A11, Y = B12 - B22
    X3 = [
        [(0,0,1)], # X[0,0]
        [(0,1,1)], # X[0,1]
        [(1,0,1)], # X[1,0]
        [(1,1,1)]  # X[1,1]
    ]
    Y3 = [
        [(2,0,1), (2,2,-1)], # Y[0,0] (B12[0,0] is B[2,0], B22[0,0] is B[2,2])
        [(2,1,1), (2,3,-1)], # Y[0,1]
        [(3,0,1), (3,2,-1)], # Y[1,0]
        [(3,1,1), (3,3,-1)]  # Y[1,1]
    ]
    
    # Outer Term 4: X = A22, Y = B21 - B11
    X4 = [
        [(2,0,1)], # X[0,0]
        [(2,1,1)], # X[0,1]
        [(3,0,1)], # X[1,0]
        [(3,1,1)]  # X[1,1]
    ]
    Y4 = [
        [(0,0,1), (0,0,-1)], # Wait, B21[0,0] is B[2,0], B11[0,0] is B[0,0]
        # Correct Y4:
        [(2,0,1), (0,0,-1)], # Y[0,0]
        [(2,1,1), (0,1,-1)], # Y[0,1]
        [(3,0,1), (1,0,-1)], # Y[1,0]
        [(3,1,1), (1,1,-1)]  # Y[1,1]
    ]
    
    # Outer Term 5: X = A11 + A12, Y = B22
    X5 = [
        [(0,0,1), (0,2,1)], # X[0,0] (A11[0,0] is A[0,0], A12[0,0] is A[0,2])
        [(0,1,1), (0,3,1)], # X[0,1]
        [(1,0,1), (1,2,1)], # X[1,0]
        [(1,1,1), (1,3,1)]  # X[1,1]
    ]
    Y5 = [
        [(2,0,1)], # Y[0,0]
        [(2,1,1)], # Y[0,1]
        [(3,0,1)], # Y[1,0]
        [(3,1,1)]  # Y[1,1]
    ]
    
    # Outer Term 6: X = A21 - A11, Y = B11 + B12
    X6 = [
        [(2,0,1), (0,0,-1)], # X[0,0]
        [(2,1,1), (0,1,-1)], # X[0,1]
        [(3,0,1), (1,0,-1)], # X[1,0]
        [(3,1,1), (1,1,-1)]  # X[1,1]
    ]
    Y6 = [
        [(0,0,1), (2,0,1)], # Y[0,0] (B11[0,0] is B[0,0], B12[0,0] is B[2,0])
        [(0,1,1), (2,1,1)], # Y[0,1]
        [(1,0,1), (3,0,1)], # Y[1,0]
        [(1,1,1), (3,1,1)]  # Y[1,1]
    ]
    
    # Outer Term 7: X = A12 - A22, Y = B21 + B22
    X7 = [
        [(0,2,1), (2,0,-1)], # X[0,0] (A12[0,0] is A[0,2], A22[0,0] is A[2,0])
        [(0,3,1), (2,1,-1)], # X[0,1]
        [(1,2,1), (3,0,-1)], # X[1,0]
        [(1,3,1), (3,1,-1)]  # X[1,1]
    ]
    Y7 = [
        [(2,0,1), (2,2,1)], # Y[0,0] (B21[0,0] is B[2,0], B22[0,0] is B[2,2])
        [(2,1,1), (2,3,1)], # Y[0,1]
        [(3,0,1), (3,2,1)], # Y[1,0]
        [(3,1,1), (3,3,1)]  # Y[1,1]
    ]
    
    outer_terms = [
        (X1, Y1),
        (X2, Y2),
        (X3, Y3),
        (X4, Y4),
        (X5, Y5),
        (X6, Y6),
        (X7, Y7)
    ]
    
    # C contributions from P_k to C
    # C11 = P11 + P41 - P51 - P71  (using P_k[0,0] etc)
    # C12 = P31 + P51
    # C21 = P21 + P41
    # C22 = P11 - P31 + P61 + P71
    
    # We need to map local 2x2 indices of P_k to global 4x4 indices of C.
    # P_k is a 2x2 block.
    # P1 is at C[0:2, 0:2] (Top-Left)
    # P2 is at C[2:4, 0:2] (Bottom-Left) -> Wait, Strassen formula:
    # C11 (Top-Left 2x2) = P1 + P4 - P5 - P7
    # C12 (Top-Right 2x2) = P3 + P5
    # C21 (Bottom-Left 2x2) = P2 + P4
    # C22 (Bottom-Right 2x2) = P1 - P3 + P6 + P7
    
    # So:
    # P1 contributes to C[0:2, 0:2] (coeff 1) and C[2:4, 2:4] (coeff 1)
    # P2 contributes to C[2:4, 0:2] (coeff 1)
    # P3 contributes to C[0:2, 2:4] (coeff 1) and C[2:4, 2:4] (coeff -1)
    # P4 contributes to C[0:2, 0:2] (coeff 1) and C[2:4, 0:2] (coeff 1)
    # P5 contributes to C[0:2, 2:4] (coeff 1) and C[0:2, 0:2] (coeff -1)
    # P6 contributes to C[2:4, 2:4] (coeff 1)
    # P7 contributes to C[2:4, 2:4] (coeff 1) and C[0:2, 0:2] (coeff -1)
    
    # c_contribs[k] is a list of (global_row_offset, global_col_offset, coeff)
    # where the local 2x2 entry (lr, lc) of P_k adds coeff * M_{k, lr, lc} to C[global_row_offset + lr, global_col_offset + lc]
    
    c_contribs = [
        # P1
        [(0,0,1), (2,2,1)],
        # P2
        [(2,0,1)],
        # P3
        [(0,2,1), (2,2,-1)],
        # P4
        [(0,0,1), (2,0,1)],
        # P5
        [(0,2,1), (0,0,-1)],
        # P6
        [(2,2,1)],
        # P7
        [(2,2,1), (0,0,-1)]
    ]
    
    term_idx = 0
    for k in range(7):
        X_k, Y_k = outer_terms[k]
        # Apply inner Strassen to X_k and Y_k
        for l in range(7):
            a_coeffs, b_coeffs, c_coeffs = strassen_2x2[l]
            
            # Construct u and v for this term
            # u is vector of length 16
            # v is vector of length 16
            
            # For each entry in a_coeffs (local 2x2 coords of X_k)
            for (lr, lc, coeff_a) in a_coeffs:
                # Get global entries for X_k[lr, lc]
                for (gr, gc, coeff_x) in X_k[lr*2+lc]: # Indexing X_k as flat? No, X_k is list of 4 lists.
                    # X_k is indexed by [lr*2+lc]?
                    # My X_k definition:
                    # X_k[0] is [0,0]
                    # X_k[1] is [0,1]
                    # X_k[2] is [1,0]
                    # X_k[3] is [1,1]
                    # So index is lr*2+lc.
                    u_idx = gr * 4 + gc
                    U[u_idx, term_idx] += coeff_a * coeff_x
            
            for (lr, lc, coeff_b) in b_coeffs:
                for (gr, gc, coeff_y) in Y_k[lr*2+lc]:
                    v_idx = gr * 4 + gc
                    V[v_idx, term_idx] += coeff_b * coeff_y
            
            # Construct w
            # c_coeffs are local 2x2 coords of the result of the inner Strassen term (which is an entry of P_k)
            # This entry contributes to C via c_contribs[k]
            for (lr, lc, coeff_c_inner) in c_coeffs:
                # This local entry (lr, lc) of P_k contributes to C
                for (orow, ocol, coeff_c_outer) in c_contribs[k]:
                    gr = orow + lr
                    gc = ocol + lc
                    w_idx = gr * 4 + gc
                    W[w_idx, term_idx] += coeff_c_inner * coeff_c_outer
            
            term_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

