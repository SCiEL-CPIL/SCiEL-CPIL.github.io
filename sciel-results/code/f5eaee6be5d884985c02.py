"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS # 3, 4, 4
    
    # Target Rank: 44
    # Decomposition:
    # Split A (3x4) into A1 (3x2, cols 0-1) and A2 (3x2, cols 2-3)
    # Split B (4x4) into B1 (2x4, rows 0-1) and B2 (2x4, rows 2-3)
    # C = A1*B1 + A2*B2
    # Each A_i*B_j is 3x2x4.
    # Split 3x2x4 into two 3x2x2 blocks (output cols 0-1 and 2-3).
    # Each 3x2x2 is split into 1x2x2 (top row of A) and 2x2x2 (bottom rows of A).
    # 1x2x2 rank 4 (schoolbook).
    # 2x2x2 rank 7 (Strassen).
    # Total per 3x2x2: 11.
    # Total per 3x2x4: 22.
    # Total: 44.

    rank = 44
    U = [[0] * rank for _ in range(n * m)] # 12 x 44
    V = [[0] * rank for _ in range(m * p)] # 16 x 44
    W = [[0] * rank for _ in range(p * n)] # 12 x 44
    
    col = 0

    def add_term(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
        """
        Add a term to the decomposition.
        u_indices: list of U row indices (i*m + j)
        v_indices: list of V row indices (j*p + k)
        w_indices: list of W row indices (k*n + i)
        u_coeffs: list of coefficients for u
        v_coeffs: list of coefficients for v
        w_coeffs: list of coefficients for w
        """
        nonlocal col
        if col >= rank:
            return
        
        for idx, coeff in zip(u_indices, u_coeffs):
            if coeff != 0:
                U[idx][col] += coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            if coeff != 0:
                V[idx][col] += coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            if coeff != 0:
                W[idx][col] += coeff
        
        col += 1

    # Helper to get U index for A[i, j]
    def u_idx(i, j):
        return i * m + j

    # Helper to get V index for B[j, k]
    def v_idx(j, k):
        return j * p + k

    # Helper to get W index for C[k, i]
    def w_idx(k, i):
        return k * n + i

    # Function to add Schoolbook 1x2x2 terms
    # A is 1x2: [a0, a1] at A[ai, j0], A[ai, j1]
    # B is 2x2: B[j0, k0], B[j0, k1], B[j1, k0], B[j1, k1]
    # C is 1x2: C[k0, ai], C[k1, ai]
    def add_1x2x2(ai, j0, j1, k0, k1):
        # C[k0, ai] = A[ai, j0]B[j0, k0] + A[ai, j1]B[j1, k0]
        # C[k1, ai] = A[ai, j0]B[j0, k1] + A[ai, j1]B[j1, k1]
        
        # Term 1: A[ai, j0] * B[j0, k0] -> C[k0, ai]
        add_term([u_idx(ai, j0)], [v_idx(j0, k0)], [w_idx(k0, ai)], [1], [1], [1])
        # Term 2: A[ai, j1] * B[j1, k0] -> C[k0, ai]
        add_term([u_idx(ai, j1)], [v_idx(j1, k0)], [w_idx(k0, ai)], [1], [1], [1])
        # Term 3: A[ai, j0] * B[j0, k1] -> C[k1, ai]
        add_term([u_idx(ai, j0)], [v_idx(j0, k1)], [w_idx(k1, ai)], [1], [1], [1])
        # Term 4: A[ai, j1] * B[j1, k1] -> C[k1, ai]
        add_term([u_idx(ai, j1)], [v_idx(j1, k1)], [w_idx(k1, ai)], [1], [1], [1])

    # Function to add Strassen 2x2x2 terms
    # A is 2x2: rows ai, ai+1; cols j0, j1
    # B is 2x2: rows j0, j1; cols k0, k1
    # C is 2x2: rows k0, k1; cols ai, ai+1
    def add_2x2x2(ai, j0, j1, k0, k1):
        # A indices:
        # a00 = A[ai, j0], a01 = A[ai, j1]
        # a10 = A[ai+1, j0], a11 = A[ai+1, j1]
        
        # B indices:
        # b00 = B[j0, k0], b01 = B[j0, k1]
        # b10 = B[j1, k0], b11 = B[j1, k1]
        
        # C indices:
        # c00 = C[k0, ai], c01 = C[k0, ai+1]
        # c10 = C[k1, ai], c11 = C[k1, ai+1]

        u00 = u_idx(ai, j0)
        u01 = u_idx(ai, j1)
        u10 = u_idx(ai+1, j0)
        u11 = u_idx(ai+1, j1)

        v00 = v_idx(j0, k0)
        v01 = v_idx(j0, k1)
        v10 = v_idx(j1, k0)
        v11 = v_idx(j1, k1)

        w00 = w_idx(k0, ai)
        w01 = w_idx(k0, ai+1)
        w10 = w_idx(k1, ai)
        w11 = w_idx(k1, ai+1)

        # m0 = (a00 + a11)(b00 + b11)
        # Contributes to c00 (+1), c11 (+1)
        add_term([u00, u11], [v00, v11], [w00, w11], [1, 1], [1, 1], [1, 1])

        # m1 = (a10 + a11)b00
        # Contributes to c10 (+1), c11 (+1)
        add_term([u10, u11], [v00], [w10, w11], [1, 1], [1], [1, 1])

        # m2 = a00(b01 - b11)
        # Contributes to c01 (+1), c11 (-1)
        add_term([u00], [v01, v11], [w01, w11], [1], [1, -1], [1, -1])

        # m3 = a11(b10 - b00)
        # Contributes to c00 (-1), c10 (+1)
        add_term([u11], [v10, v00], [w00, w10], [1], [1, -1], [-1, 1])

        # m4 = (a00 + a01)b11
        # Contributes to c01 (+1), c11 (-1)
        add_term([u00, u01], [v11], [w01, w11], [1, 1], [1], [1, -1])

        # m5 = (a10 - a00)(b00 + b01)
        # Contributes to c00 (+1), c10 (+1)
        add_term([u10, u00], [v00, v01], [w00, w10], [1, -1], [1, 1], [1, 1])

        # m6 = (a01 - a11)(b10 + b11)
        # Contributes to c00 (-1)
        add_term([u01, u11], [v10, v11], [w00], [1, -1], [1, 1], [-1])

    # 
    # 1. Process A1 (cols 0-1) * B1 (rows 0-1)
    # A1 is A[0:3, 0:2]. B1 is B[0:2, 0:4].
    # Split B1 into B1a (cols 0-1) and B1b (cols 2-3).
    
    # 1a. A1 * B1a (3x2x2)
    # Top row of A1 (i=0) * B1a (j=0,1; k=0,1)
    add_1x2x2(0, 0, 1, 0, 1)
    # Bottom rows of A1 (i=1,2) * B1a (j=0,1; k=0,1)
    add_2x2x2(1, 0, 1, 0, 1)

    # 1b. A1 * B1b (3x2x2)
    # Top row of A1 (i=0) * B1b (j=0,1; k=2,3)
    add_1x2x2(0, 0, 1, 2, 3)
    # Bottom rows of A1 (i=1,2) * B1b (j=0,1; k=2,3)
    add_2x2x2(1, 0, 1, 2, 3)

    # 
    # 2. Process A2 (cols 2-3) * B2 (rows 2-3)
    # A2 is A[0:3, 2:4]. B2 is B[2:4, 0:4].
    # Split B2 into B2a (cols 0-1) and B2b (cols 2-3).
    
    # 2a. A2 * B2a (3x2x2)
    # Top row of A2 (i=0) * B2a (j=2,3; k=0,1)
    add_1x2x2(0, 2, 3, 0, 1)
    # Bottom rows of A2 (i=1,2) * B2a (j=2,3; k=0,1)
    add_2x2x2(1, 2, 3, 0, 1)

    # 2b. A2 * B2b (3x2x2)
    # Top row of A2 (i=0) * B2b (j=2,3; k=2,3)
    add_1x2x2(0, 2, 3, 2, 3)
    # Bottom rows of A2 (i=1,2) * B2b (j=2,3; k=2,3)
    add_2x2x2(1, 2, 3, 2, 3)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
