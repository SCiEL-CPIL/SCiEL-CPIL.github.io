"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=7
    # U: (n*m, rank) = (15, rank)
    # V: (m*p, rank) = (35, rank)
    # W: (p*n, rank) = (21, rank)
    
    # We will build the columns one by one.
    # First 7 columns: Strassen for top-left 2x2x2 block
    # Next 97 columns: Schoolbook for the rest
    
    num_strassen = 7
    # Schoolbook terms: all (i,j,k) except i in {0,1}, j in {0,1}, k in {0,1}
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                schoolbook_terms.append((i, j, k))
                
    total_rank = num_strassen + len(schoolbook_terms)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Define Strassen terms
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B11 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # W contributions:
    # C00 <- M1 + M4 - M5 + M7
    # C01 <- M3 + M5
    # C10 <- M2 + M4
    # C11 <- M1 - M2 + M3 + M6
    
    # Indices in W: k*n + i
    # C00: k=0, i=0 -> 0
    # C01: k=1, i=0 -> 3
    # C10: k=0, i=1 -> 1
    # C11: k=1, i=1 -> 4
    
    strassen_defs = [
        # (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
        # M1
        (
            [(0*5+0, 1), (1*5+1, 1)],  # A00, A11
            [(0*7+0, 1), (1*7+1, 1)],  # B00, B11
            [(0, 1), (4, 1)]           # C00 +1, C11 +1
        ),
        # M2
        (
            [(1*5+0, 1), (1*5+1, 1)],  # A10, A11
            [(0*7+0, 1)],              # B00
            [(1, 1), (4, -1)]          # C10 +1, C11 -1
        ),
        # M3
        (
            [(0*5+0, 1)],              # A00
            [(1*7+0, 1), (0*7+0, -1)], # B10, -B00
            [(3, 1), (4, 1)]           # C01 +1, C11 +1
        ),
        # M4
        (
            [(1*5+1, 1)],              # A11
            [(1*7+1, 1), (0*7+0, -1)], # B11, -B00
            [(0, 1), (1, 1)]           # C00 +1, C10 +1
        ),
        # M5
        (
            [(0*5+0, 1), (0*5+1, 1)],  # A00, A01
            [(1*7+1, 1)],              # B11
            [(0, -1), (3, 1)]          # C00 -1, C01 +1
        ),
        # M6
        (
            [(1*5+0, 1), (0*5+0, -1)], # A10, -A00
            [(0*7+0, 1), (0*7+1, 1)],  # B00, B01
            [(4, 1)]                   # C11 +1
        ),
        # M7
        (
            [(0*5+1, 1), (1*5+1, -1)], # A01, -A11
            [(1*7+0, 1), (1*7+1, 1)],  # B10, B11
            [(0, 1)]                   # C00 +1
        )
    ]
    
    col_idx = 0
    for u_defs, v_defs, w_defs in strassen_defs:
        for idx, val in u_defs:
            U[idx][col_idx] += val
        for idx, val in v_defs:
            V[idx][col_idx] += val
        for idx, val in w_defs:
            W[idx][col_idx] += val
        col_idx += 1
        
    # Schoolbook terms
    for i, j, k in schoolbook_terms:
        U[i * m + j][col_idx] = 1
        V[j * p + k][col_idx] = 1
        W[k * n + i][col_idx] = 1
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
