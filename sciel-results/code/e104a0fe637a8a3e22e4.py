import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 instances of Strassen (rank 7) = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U coeffs for A[i][j]
    # V coeffs for B[k][l]
    # W coeffs for C[l][i] (index k*n+i in W matrix)
    
    # The 7 terms of Strassen
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Contributions to C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the U, V, W patterns for each M
    # U pattern: dict (i,j) -> coeff
    # V pattern: dict (k,l) -> coeff
    # W pattern: dict (l,i) -> coeff (Note: W index is k*n+i, so we need to map (l,i) to index? 
    # Wait, W index is k*n+i. In C[l][i], l is row of B (k), i is row of A (i).
    # So W index for C[l][i] is l*n + i.
    
    strassen_terms = [
        {
            "U": {(0,0): 1, (1,1): 1},
            "V": {(0,0): 1, (1,1): 1},
            "W": {(0,0): 1, (1,1): 1} # C00, C11
        },
        {
            "U": {(1,0): 1, (1,1): 1},
            "V": {(0,0): 1},
            "W": {(0,1): 1} # C10
        },
        {
            "U": {(0,0): 1},
            "V": {(0,1): 1, (1,1): -1},
            "W": {(1,0): 1, (1,1): 1} # C01, C11
        },
        {
            "U": {(1,1): 1},
            "V": {(1,0): 1, (0,0): -1},
            "W": {(0,0): 1, (0,1): 1} # C00, C10
        },
        {
            "U": {(0,0): 1, (0,1): 1},
            "V": {(1,1): 1},
            "W": {(0,0): -1, (1,0): 1} # C00(-1), C01(1)
        },
        {
            "U": {(1,0): 1, (0,0): -1},
            "V": {(0,0): 1, (0,1): 1},
            "W": {(1,1): 1} # C11
        },
        {
            "U": {(0,1): 1, (1,1): -1},
            "V": {(1,0): 1, (1,1): 1},
            "W": {(0,0): 1} # C00
        }
    ]
    
    # Instances:
    # 1. j in 0,1; k in 0,1; l in 0,1
    # 2. j in 0,1; k in 0,1; l in 2,3
    # 3. j in 2,3; k in 2,3; l in 0,1
    # 4. j in 2,3; k in 2,3; l in 2,3
    
    instances = [
        (0, 0, 0), # j_offset, k_offset, l_offset
        (0, 0, 2),
        (2, 2, 0),
        (2, 2, 2)
    ]
    
    col_idx = 0
    for j_off, k_off, l_off in instances:
        for term in strassen_terms:
            # Fill U
            for (i, j), coeff in term["U"].items():
                global_j = j_off + j
                U[i * m + global_j][col_idx] += coeff
            
            # Fill V
            for (k, l), coeff in term["V"].items():
                global_k = k_off + k
                global_l = l_off + l
                V[global_k * p + global_l][col_idx] += coeff
            
            # Fill W
            # W index is k*n + i.
            # The term contributes to C[l][i].
            # So W index is l*n + i.
            for (l, i), coeff in term["W"].items():
                global_l = l_off + l
                global_k = k_off + k # Wait, W depends on k (row of B) and i (row of A/C).
                # In the decomposition C[i][k] = sum ... W[k*n+i]
                # The output index is (i, k).
                # In Strassen, the output is C_local[l][i].
                # So global output index is (i, global_k).
                # Wait, l in local Strassen corresponds to column of B.
                # In global B, column is l_off + l.
                # But W index is k*n + i.
                # Here k is the row of B.
                # In the term V[k][l], k is row of B, l is col of B.
                # The product A[i][j] B[k][l] contributes to C[i][k]?
                # No, A is n x m, B is m x p.
                # A[i][j] B[j][k] -> C[i][k].
                # So the shared index is j (col of A, row of B).
                # In Strassen local:
                # A[i'][j'] B[j'][k'] -> C[i'][k'].
                # So the output index is (i', k').
                # In global:
                # i' -> i.
                # k' -> global_k = k_off + k'.
                # So W index is global_k * n + i.
                
                # Let's re-verify the W mapping in the prompt.
                # "W[k*n+i,r]"
                # So W row index is k*n + i.
                # k is row of B. i is row of A.
                # So for a local term contributing to C[i'][k'],
                # the W entry is at row (k_off + k') * n + i'.
                
                # My term["W"] keys are (l, i) in my previous thought process?
                # Let's check term 1: C00 and C11.
                # C00: i=0, k=0. W index 0*n + 0 = 0.
                # C11: i=1, k=1. W index 1*n + 1 = 3 (if n=2, 1*2+1=3).
                # In my strassen_terms[0]["W"], I had {(0,0): 1, (1,1): 1}.
                # If key is (k, i), then (0,0) is k=0,i=0. (1,1) is k=1,i=1.
                # So I should use (k, i) as keys in W.
                
                pass # Logic needs to be applied in the loop below

            # Corrected W filling
            for (k, i), coeff in term["W"].items():
                global_k = k_off + k
                W[global_k * n + i][col_idx] += coeff
                
            col_idx += 1

    return {"U": U, "V": V, "W": W}
