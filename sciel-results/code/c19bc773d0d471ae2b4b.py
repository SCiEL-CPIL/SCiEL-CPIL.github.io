import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We will build the rank 44 decomposition.
    # Total rank = 44.
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        """
        Adds Strassen decomposition for A[i0:i1+1, j0:j1+1] * B[j0:j1+1, k0:k1+1]
        to C[i0:i1+1, k0:k1+1].
        Assumes i1=i0+1, j1=j0+1, k1=k0+1.
        """
        nonlocal current_rank
        # Local indices 0, 1 correspond to global i0, i1 etc.
        # A elements: (i0, j0), (i0, j1), (i1, j0), (i1, j1)
        # B elements: (j0, k0), (j0, k1), (j1, k0), (j1, k1)
        # C elements: (i0, k0), (i0, k1), (i1, k0), (i1, k1)
        
        # Strassen M1-M7
        # M1 = (a00 + a11)(b00 + b11)
        # M2 = (a10 + a11)b00
        # M3 = a00(b01 - b11)
        # M4 = a11(b10 - b00)
        # M5 = (a00 + a01)b11
        # M6 = (a10 - a00)(b00 + b01)
        # M7 = (a01 - a11)(b10 + b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        def add_term(r, u_coeffs, v_coeffs, w_coeffs):
            # u_coeffs: dict {(i,j): coeff}
            # v_coeffs: dict {(j,k): coeff}
            # w_coeffs: dict {(i,k): coeff}
            for (i, j), c in u_coeffs.items():
                U[i * m + j][r] += c
            for (j, k), c in v_coeffs.items():
                V[j * p + k][r] += c
            for (i, k), c in w_coeffs.items():
                W[k * n + i][r] += c

        # M1
        add_term(current_rank, 
                 {(i0, j0): 1, (i1, j1): 1}, 
                 {(j0, k0): 1, (j1, k1): 1}, 
                 {(i0, k0): 1, (i1, k1): 1})
        current_rank += 1
        
        # M2
        add_term(current_rank, 
                 {(i1, j0): 1, (i1, j1): 1}, 
                 {(j0, k0): 1}, 
                 {(i1, k0): 1})
        current_rank += 1
        
        # M3
        add_term(current_rank, 
                 {(i0, j0): 1}, 
                 {(j0, k1): 1, (j1, k1): -1}, 
                 {(i0, k1): 1})
        current_rank += 1
        
        # M4
        add_term(current_rank, 
                 {(i1, j1): 1}, 
                 {(j1, k0): 1, (j0, k0): -1}, 
                 {(i0, k0): 1, (i1, k0): 1})
        current_rank += 1
        
        # M5
        add_term(current_rank, 
                 {(i0, j0): 1, (i0, j1): 1}, 
                 {(j1, k1): 1}, 
                 {(i0, k0): -1, (i0, k1): 1})
        current_rank += 1
        
        # M6
        add_term(current_rank, 
                 {(i1, j0): 1, (i0, j0): -1}, 
                 {(j0, k0): 1, (j0, k1): 1}, 
                 {(i1, k1): 1})
        current_rank += 1
        
        # M7
        add_term(current_rank, 
                 {(i0, j1): 1, (i1, j1): -1}, 
                 {(j1, k0): 1, (j1, k1): 1}, 
                 {(i0, k0): 1})
        current_rank += 1

    def add_standard_block(i0, j0, j1, k0, k1):
        """
        Adds standard decomposition for A[i0, j0:j1+1] * B[j0:j1+1, k0:k1+1]
        to C[i0, k0:k1+1].
        Assumes j1=j0+1, k1=k0+1.
        """
        nonlocal current_rank
        # A elements: (i0, j0), (i0, j1)
        # B elements: (j0, k0), (j0, k1), (j1, k0), (j1, k1)
        # C elements: (i0, k0), (i0, k1)
        
        # C0 = a00*b00 + a01*b10
        # C1 = a00*b01 + a01*b11
        
        def add_term(r, u_coeffs, v_coeffs, w_coeffs):
            for (i, j), c in u_coeffs.items():
                U[i * m + j][r] += c
            for (j, k), c in v_coeffs.items():
                V[j * p + k][r] += c
            for (i, k), c in w_coeffs.items():
                W[k * n + i][r] += c

        # M1: a00 * b00 -> C0
        add_term(current_rank, 
                 {(i0, j0): 1}, 
                 {(j0, k0): 1}, 
                 {(i0, k0): 1})
        current_rank += 1
        
        # M2: a01 * b10 -> C0
        add_term(current_rank, 
                 {(i0, j1): 1}, 
                 {(j1, k0): 1}, 
                 {(i0, k0): 1})
        current_rank += 1
        
        # M3: a00 * b01 -> C1
        add_term(current_rank, 
                 {(i0, j0): 1}, 
                 {(j0, k1): 1}, 
                 {(i0, k1): 1})
        current_rank += 1
        
        # M4: a01 * b11 -> C1
        add_term(current_rank, 
                 {(i0, j1): 1}, 
                 {(j1, k1): 1}, 
                 {(i0, k1): 1})
        current_rank += 1

    # Blocks:
    # Top-Left Output (i=0,1; k=0,1)
    add_strassen_block(0, 1, 0, 1, 0, 1) # A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(0, 1, 2, 3, 0, 1) # A[0:2, 2:4] * B[2:4, 0:2]
    
    # Top-Right Output (i=0,1; k=2,3)
    add_strassen_block(0, 1, 0, 1, 2, 3) # A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(0, 1, 2, 3, 2, 3) # A[0:2, 2:4] * B[2:4, 2:4]
    
    # Bottom-Left Output (i=2; k=0,1)
    add_standard_block(2, 0, 1, 0, 1) # A[2, 0:2] * B[0:2, 0:2]
    add_standard_block(2, 2, 3, 0, 1) # A[2, 2:4] * B[2:4, 0:2]
    
    # Bottom-Right Output (i=2; k=2,3)
    add_standard_block(2, 0, 1, 2, 3) # A[2, 0:2] * B[0:2, 2:4]
    add_standard_block(2, 2, 3, 2, 3) # A[2, 2:4] * B[2:4, 2:4]
    
    # Verify rank
    if current_rank != rank:
        # This should not happen
        pass
        
    return {"U": U, "V": V, "W": W}
