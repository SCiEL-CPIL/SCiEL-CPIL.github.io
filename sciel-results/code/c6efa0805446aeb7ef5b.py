"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Hybrid decomposition: Strassen for 2x2x2 core + schoolbook for the rest
    # Strassen core: i in {0,1}, j in {0,1}, k in {0,1} -> 7 terms
    # Schoolbook: all other (i,j,k) combinations -> 54 - 8 = 46 terms
    # Total rank: 7 + 46 = 53
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in u_vals.items():
            U[idx][col] += val
        for idx, val in v_vals.items():
            V[idx][col] += val
        for idx, val in w_vals.items():
            W[idx][col] += val
        col += 1
    
    # Strassen 2x2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A indices: U[i*m+j] = U[i*3+j]
    # B indices: V[j*p+k] = V[j*6+k]
    # C indices: W[k*n+i] = W[k*3+i]
    
    # M1 = (A00+A11)(B00+B11)
    add_term({0: 1, 4: 1}, {0: 1, 7: 1}, {0: 1, 4: 1})
    
    # M2 = (A10+A11)(B00)
    add_term({3: 1, 4: 1}, {0: 1}, {1: 1, 4: 1})
    
    # M3 = (A00)(B10+B11)
    add_term({0: 1}, {6: 1, 7: 1}, {3: 1, 4: 1})
    
    # M4 = (A11)(B10-B11)
    add_term({4: 1}, {6: 1, 7: -1}, {0: 1, 1: 1})
    
    # M5 = (A00+A10)(B11)
    add_term({0: 1, 3: 1}, {7: 1}, {0: -1, 3: 1})
    
    # M6 = (A10-A11)(B00+B10)
    add_term({3: 1, 4: -1}, {0: 1, 6: 1}, {4: 1})
    
    # M7 = (A01-A10)(B01+B10)
    add_term({1: 1, 3: -1}, {1: 1, 6: 1}, {0: 1})
    
    # Schoolbook for all other (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
