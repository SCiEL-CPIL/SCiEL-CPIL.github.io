"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total standard terms
    total_terms = n * m * p
    # We will use rank 59
    rank = total_terms - 1
    
    # Initialize matrices with zeros
    # Using lists of lists for the output format as requested
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the specific 2x2x2 block to replace with Strassen
    # i in {0, 1}, j in {0, 1}, k in {0, 1}
    strassen_block = {(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)}
    
    col = 0
    
    # Add all standard terms except those in the Strassen block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_block:
                    continue
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    # Add Strassen terms for the block i in {0,1}, j in {0,1}, k in {0,1}
    # Indices mapping:
    # u(i,j) = i*5 + j
    # v(j,k) = j*6 + k
    # w(k,i) = k*2 + i
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for u_idx, coeff in u_coeffs:
            U[u_idx][col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx][col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx][col] += coeff
        col += 1
        
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11) -> C00 +1, C11 +1
    # M2 = (A10 + A11)B00        -> C10 +1, C11 -1
    # M3 = A00(B01 - B11)        -> C01 +1, C11 +1
    # M4 = A11(B10 - B00)        -> C00 +1, C10 +1
    # M5 = (A00 + A01)B11        -> C00 -1, C01 +1
    # M6 = (A10 - A00)(B00 + B01)-> C11 +1
    # M7 = (A01 - A11)(B10 + B11)-> C00 +1
    
    # Pre-calculate indices
    u00, u01 = 0*5+0, 0*5+1
    u10, u11 = 1*5+0, 1*5+1
    
    v00, v01 = 0*6+0, 0*6+1
    v10, v11 = 1*6+0, 1*6+1
    
    w00, w01 = 0*2+0, 1*2+0
    w10, w11 = 0*2+1, 1*2+1
    
    # Term 1: M1
    add_term(
        [(u00, 1), (u11, 1)],
        [(v00, 1), (v11, 1)],
        [(w00, 1), (w11, 1)]
    )
    
    # Term 2: M2
    add_term(
        [(u10, 1), (u11, 1)],
        [(v00, 1)],
        [(w10, 1), (w11, -1)]
    )
    
    # Term 3: M3
    add_term(
        [(u00, 1)],
        [(v01, 1), (v11, -1)],
        [(w01, 1), (w11, 1)]
    )
    
    # Term 4: M4
    add_term(
        [(u11, 1)],
        [(v10, 1), (v00, -1)],
        [(w00, 1), (w10, 1)]
    )
    
    # Term 5: M5
    add_term(
        [(u00, 1), (u01, 1)],
        [(v11, 1)],
        [(w00, -1), (w01, 1)]
    )
    
    # Term 6: M6
    add_term(
        [(u10, 1), (u00, -1)],
        [(v00, 1), (v01, 1)],
        [(w11, 1)]
    )
    
    # Term 7: M7
    add_term(
        [(u01, 1), (u11, -1)],
        [(v10, 1), (v11, 1)],
        [(w00, 1)]
    )
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
