"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for (i, j), val in u_vals.items():
            if val != 0:
                U[i * m + j][col] += val
        for (j, k), val in v_vals.items():
            if val != 0:
                V[j * p + k][col] += val
        for (k, i), val in w_vals.items():
            if val != 0:
                W[k * n + i][col] += val
        col += 1

    # Part 1: j=0 (1x7 part)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            add_term(
                {(i, 0): 1},
                {(0, k): 1},
                {(k, i): 1}
            )
            
    # Part 2: j in {1, 2} (2x7 part)
    # Split k into {0}, {1,2}, {3,4}, {5,6}
    
    # Sub-part 2.1: k=0 (2x2x1)
    # Schoolbook for 2x2x1
    for i in range(n):
        for j_prime in range(2): # j_prime 0, 1 corresponds to global j 1, 2
            j = j_prime + 1
            k = 0
            add_term(
                {(i, j): 1},
                {(j, k): 1},
                {(k, i): 1}
            )

    # Sub-part 2.2: k in {1, 2} (2x2x2)
    # Strassen
    # Local indices: i in 0,1; j_prime in 0,1 (global j 1,2); k_prime in 0,1 (global k 1,2)
    # A[i, j_prime], B[j_prime, k_prime], C[i, k_prime]
    
    # M1 = (A00+A11)(B00+B11)
    add_term(
        {(0, 1): 1, (1, 2): 1},
        {(1, 1): 1, (2, 2): 1},
        {(1, 0): 1, (2, 0): 1, (1, 1): 1, (2, 1): 1} # Wait, W is (k,i).
        # C00 = M1... C01=M1... C10=M1... C11=M1...
        # So W[1,0]=1, W[2,0]=1, W[1,1]=1, W[2,1]=1
    )
    
    # M2 = (A10+A11)B00
    add_term(
        {(1, 1): 1, (1, 2): 1},
        {(1, 1): 1},
        {(1, 1): 1, (2, 1): 1} # C10, C11
    )
    
    # M3 = A00(B01-B11)
    add_term(
        {(0, 1): 1},
        {(1, 2): 1, (2, 2): -1},
        {(1, 0): 1, (1, 1): 1} # C00, C01
    )
    
    # M4 = A11(B10-B00)
    add_term(
        {(1, 2): 1},
        {(2, 1): 1, (1, 1): -1},
        {(1, 0): 1, (2, 0): 1} # C00, C10
    )
    
    # M5 = (A00+A10)B11
    add_term(
        {(0, 1): 1, (1, 1): 1},
        {(2, 2): 1},
        {(1, 0): -1, (2, 0): 1, (1, 1): 1} # C00(-1), C10(1), C01(1)
    )
    
    # M6 = (A11-A00)(B00+B01)
    add_term(
        {(1, 2): 1, (0, 1): -1},
        {(1, 1): 1, (1, 2): 1},
        {(2, 1): 1} # C11
    )
    
    # M7 = (A01-A11)(B10+B11)
    # Wait, A01 is (0,2) in global? No, local j_prime 1 is global j 2.
    # A[i, j_prime].
    # A01 is i=0, j_prime=1 -> (0,2) global.
    # A11 is i=1, j_prime=1 -> (1,2) global.
    add_term(
        {(0, 2): 1, (1, 2): -1},
        {(2, 1): 1, (2, 2): 1},
        {(1, 0): 1, (2, 1): 1} # C00, C11
    )

    # Sub-part 2.3: k in {3, 4} (2x2x2)
    # Same Strassen, just offset k by 2
    k_offset = 3
    
    # M1
    add_term(
        {(0, 1): 1, (1, 2): 1},
        {(1, k_offset): 1, (2, k_offset+1): 1},
        {(k_offset, 0): 1, (k_offset+1, 0): 1, (k_offset, 1): 1, (k_offset+1, 1): 1}
    )
    
    # M2
    add_term(
        {(1, 1): 1, (1, 2): 1},
        {(1, k_offset): 1},
        {(k_offset, 1): 1, (k_offset+1, 1): 1}
    )
    
    # M3
    add_term(
        {(0, 1): 1},
        {(1, k_offset+1): 1, (2, k_offset+1): -1},
        {(k_offset, 0): 1, (k_offset, 1): 1}
    )
    
    # M4
    add_term(
        {(1, 2): 1},
        {(2, k_offset): 1, (1, k_offset): -1},
        {(k_offset, 0): 1, (k_offset+1, 0): 1}
    )
    
    # M5
    add_term(
        {(0, 1): 1, (1, 1): 1},
        {(2, k_offset+1): 1},
        {(k_offset, 0): -1, (k_offset+1, 0): 1, (k_offset, 1): 1}
    )
    
    # M6
    add_term(
        {(1, 2): 1, (0, 1): -1},
        {(1, k_offset): 1, (1, k_offset+1): 1},
        {(k_offset+1, 1): 1}
    )
    
    # M7
    add_term(
        {(0, 2): 1, (1, 2): -1},
        {(2, k_offset): 1, (2, k_offset+1): 1},
        {(k_offset, 0): 1, (k_offset+1, 1): 1}
    )

    # Sub-part 2.4: k in {5, 6} (2x2x2)
    k_offset = 5
    
    # M1
    add_term(
        {(0, 1): 1, (1, 2): 1},
        {(1, k_offset): 1, (2, k_offset+1): 1},
        {(k_offset, 0): 1, (k_offset+1, 0): 1, (k_offset, 1): 1, (k_offset+1, 1): 1}
    )
    
    # M2
    add_term(
        {(1, 1): 1, (1, 2): 1},
        {(1, k_offset): 1},
        {(k_offset, 1): 1, (k_offset+1, 1): 1}
    )
    
    # M3
    add_term(
        {(0, 1): 1},
        {(1, k_offset+1): 1, (2, k_offset+1): -1},
        {(k_offset, 0): 1, (k_offset, 1): 1}
    )
    
    # M4
    add_term(
        {(1, 2): 1},
        {(2, k_offset): 1, (1, k_offset): -1},
        {(k_offset, 0): 1, (k_offset+1, 0): 1}
    )
    
    # M5
    add_term(
        {(0, 1): 1, (1, 1): 1},
        {(2, k_offset+1): 1},
        {(k_offset, 0): -1, (k_offset+1, 0): 1, (k_offset, 1): 1}
    )
    
    # M6
    add_term(
        {(1, 2): 1, (0, 1): -1},
        {(1, k_offset): 1, (1, k_offset+1): 1},
        {(k_offset+1, 1): 1}
    )
    
    # M7
    add_term(
        {(0, 2): 1, (1, 2): -1},
        {(2, k_offset): 1, (2, k_offset+1): 1},
        {(k_offset, 0): 1, (k_offset+1, 1): 1}
    )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
