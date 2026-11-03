def construct():
    import time
    _t0 = time.time()
    _DEADLINE = 130.0

    n, m, p = 3, 4, 6
    terms = []  # Each term: (u_dict, v_dict, w_dict)

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Add Strassen decomposition for A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1:k2+1] -> C[i1:i2+1, k1:k2+1]"""
        a = (i1, j1)
        b = (i1, j2)
        c = (i2, j1)
        d = (i2, j2)
        e = (j1, k1)
        f = (j1, k2)
        g = (j2, k1)
        h = (j2, k2)

        C11 = (i1, k1)
        C12 = (i1, k2)
        C21 = (i2, k1)
        C22 = (i2, k2)

        terms.append((
            {a: 1, d: 1},
            {e: 1, h: 1},
            {C11: 1, C22: 1}
        ))
        terms.append((
            {c: 1, d: 1},
            {e: 1},
            {C21: 1, C22: -1}
        ))
        terms.append((
            {a: 1},
            {f: 1, h: -1},
            {C12: 1, C22: 1}
        ))
        terms.append((
            {d: 1},
            {g: 1, e: -1},
            {C11: 1, C21: 1}
        ))
        terms.append((
            {a: 1, b: 1},
            {h: 1},
            {C11: -1, C12: 1}
        ))
        terms.append((
            {c: 1, a: -1},
            {e: 1, f: 1},
            {C22: 1}
        ))
        terms.append((
            {b: 1, d: -1},
            {g: 1, h: 1},
            {C11: 1}
        ))

    def add_schoolbook_term(i, j, k):
        terms.append((
            {(i, j): 1},
            {(j, k): 1},
            {(i, k): 1}
        ))

    # Strategy: Split B into two 4x3 blocks (cols 0-2 and 3-5)
    # For rows 1-2 of A, use structured decomposition derived from Strassen
    # For row 0, use schoolbook
    #
    # The key insight: decompose 3x4x3 sub-tensor using patterns from 4x4x4 Strassen
    # Zero out the 4th index and apply linear combinations
    
    # Row 0: schoolbook (24 terms)
    for j in range(4):
        for k in range(6):
            add_schoolbook_term(0, j, k)
    
    # Rows 1-2: Use Strassen-like decomposition on 2x4x3 blocks
    # Split columns of B into two groups of 3
    # For each group, decompose A[1:3, :] * B[:, group] using structured approach
    
    # Group 1: B columns 0,1,2
    # Group 2: B columns 3,4,5
    
    for k_start in [0, 3]:
        k1, k2, k3 = k_start, k_start + 1, k_start + 2
        
        # Decompose 2x4x3: A rows 1,2; B all 4 rows; C cols k1,k2,k3
        # Use Strassen on 2x2x2 sub-blocks and combine
        
        # Left half of B (cols j=0,1): 2x2x3
        # Right half of B (cols j=2,3): 2x2x3
        
        # For left half (j=0,1), k=k1,k2,k3:
        # M1 = (A[1,0]+A[2,1])*(B[0,k1]+B[1,k3]) -> C[1,k1], C[2,k3]
        # M2 = (A[2,0]+A[2,1])*B[0,k1] -> C[2,k1], C[2,k3]-
        # M3 = A[1,0]*(B[0,k2]-B[1,k3]) -> C[1,k2], C[2,k3]
        # M4 = A[2,1]*(B[1,k1]-B[0,k2]) -> C[1,k1], C[2,k1]
        # M5 = (A[1,0]+A[1,1])*B[1,k3] -> C[1,k1]-, C[1,k2]
        # M6 = (A[2,0]-A[1,0])*(B[0,k1]+B[0,k2]) -> C[2,k3]
        # M7 = (A[1,1]-A[2,1])*(B[1,k1]+B[1,k3]) -> C[1,k1]
        
        # For right half (j=2,3), k=k1,k2,k3:
        # Similar pattern with j=2,3
        
        # Left block (j=0,1):
        terms.append((
            {(1, 0): 1, (2, 1): 1},
            {(0, k1): 1, (1, k3): 1},
            {(1, k1): 1, (2, k3): 1}
        ))
        terms.append((
            {(2, 0): 1, (2, 1): 1},
            {(0, k1): 1},
            {(2, k1): 1, (2, k3): -1}
        ))
        terms.append((
            {(1, 0): 1},
            {(0, k2): 1, (1, k3): -1},
            {(1, k2): 1, (2, k3): 1}
        ))
        terms.append((
            {(2, 1): 1},
            {(1, k1): 1, (0, k2): -1},
            {(1, k1): 1, (2, k1): 1}
        ))
        terms.append((
            {(1, 0): 1, (1, 1): 1},
            {(1, k3): 1},
            {(1, k1): -1, (1, k2): 1}
        ))
        terms.append((
            {(2, 0): 1, (1, 0): -1},
            {(0, k1): 1, (0, k2): 1},
            {(2, k3): 1}
        ))
        terms.append((
            {(1, 1): 1, (2, 1): -1},
            {(1, k1): 1, (1, k3): 1},
            {(1, k1): 1}
        ))
        
        # Right block (j=2,3):
        terms.append((
            {(1, 2): 1, (2, 3): 1},
            {(2, k1): 1, (3, k3): 1},
            {(1, k1): 1, (2, k3): 1}
        ))
        terms.append((
            {(2, 2): 1, (2, 3): 1},
            {(2, k1): 1},
            {(2, k1): 1, (2, k3): -1}
        ))
        terms.append((
            {(1, 2): 1},
            {(2, k2): 1, (3, k3): -1},
            {(1, k2): 1, (2, k3): 1}
        ))
        terms.append((
            {(2, 3): 1},
            {(3, k1): 1, (2, k2): -1},
            {(1, k1): 1, (2, k1): 1}
        ))
        terms.append((
            {(1, 2): 1, (1, 3): 1},
            {(3, k3): 1},
            {(1, k1): -1, (1, k2): 1}
        ))
        terms.append((
            {(2, 2): 1, (1, 2): -1},
            {(2, k1): 1, (2, k2): 1},
            {(2, k3): 1}
        ))
        terms.append((
            {(1, 3): 1, (2, 3): -1},
            {(3, k1): 1, (3, k3): 1},
            {(1, k1): 1}
        ))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff

    if time.time() - _t0 > _DEADLINE:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
