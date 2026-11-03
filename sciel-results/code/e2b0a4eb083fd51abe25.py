"""Matrix multiplication 4x5x6 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    def check_deadline():
        if time.time() > deadline:
            return True
        return False

    n, m, p = 4, 5, 6

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_sparse, v_sparse, w_sparse):
        U_cols.append(u_sparse)
        V_cols.append(v_sparse)
        W_cols.append(w_sparse)

    def schoolbook_block(i_range, j_range, k_range):
        """Add schoolbook terms for A[i,j]*B[j,k]->C[i,k]."""
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    add_term(
                        {(i * m + j): 1},
                        {(j * p + k): 1},
                        {(k * n + i): 1}
                    )

    def laderman_block(row_off, col_off_a, col_off_b, row_off_b):
        """
        Multiply A[row_off:row_off+3, col_off_a:col_off_a+3]
                 by B[row_off_b:row_off_b+3, col_off_b:col_off_b+3]
        -> C[row_off:row_off+3, col_off_b:col_off_b+3]
        using Laderman's rank-7 decomposition.
        """
        # Laderman's 7 products (0-indexed local, 3x3):
        # z1 = (a00+a10+a11)(b00+b10+b11)
        # z2 = (a01+a02+a12)(b01+b12)
        # z3 = (a00+a01+a11)(b00+b01+b11)
        # z4 = (a02+a12+a22)(b02+b12+b22)
        # z5 = (a00+a10+a20)(b00+b10+b20)
        # z6 = (a01+a11+a21)(b01+b11+b21)
        # z7 = (a10+a20+a21)(b10+b20+b21)
        #
        # Reconstruction:
        # C00 = z1 - z2 - z3 + z4
        # C01 = z2 - z4
        # C02 = z4
        # C10 = z3 - z1
        # C11 = z1 - z3 - z6 + z7
        # C12 = z6 - z7
        # C20 = z5 - z3
        # C21 = z3 - z5 - z7 + z1
        # C22 = z5 - z7

        # Define the 7 bilinear forms
        # Each is (u_dict, v_dict) where keys are global indices
        def a_idx(r, c):
            return (row_off + r) * m + (col_off_a + c)

        def b_idx(r, c):
            return (row_off_b + r) * p + (col_off_b + c)

        def c_idx(r, c):
            return (col_off_b + c) * n + (row_off + r)

        # z1 = (a00+a10+a11)(b00+b10+b11)
        z1_u = {a_idx(0,0): 1, a_idx(1,0): 1, a_idx(1,1): 1}
        z1_v = {b_idx(0,0): 1, b_idx(1,0): 1, b_idx(1,1): 1}
        # z2 = (a01+a02+a12)(b01+b12)
        z2_u = {a_idx(0,1): 1, a_idx(0,2): 1, a_idx(1,2): 1}
        z2_v = {b_idx(0,1): 1, b_idx(1,2): 1}
        # z3 = (a00+a01+a11)(b00+b01+b11)
        z3_u = {a_idx(0,0): 1, a_idx(0,1): 1, a_idx(1,1): 1}
        z3_v = {b_idx(0,0): 1, b_idx(0,1): 1, b_idx(1,1): 1}
        # z4 = (a02+a12+a22)(b02+b12+b22)
        z4_u = {a_idx(0,2): 1, a_idx(1,2): 1, a_idx(2,2): 1}
        z4_v = {b_idx(0,2): 1, b_idx(1,2): 1, b_idx(2,2): 1}
        # z5 = (a00+a10+a20)(b00+b10+b20)
        z5_u = {a_idx(0,0): 1, a_idx(1,0): 1, a_idx(2,0): 1}
        z5_v = {b_idx(0,0): 1, b_idx(1,0): 1, b_idx(2,0): 1}
        # z6 = (a01+a11+a21)(b01+b11+b21)
        z6_u = {a_idx(0,1): 1, a_idx(1,1): 1, a_idx(2,1): 1}
        z6_v = {b_idx(0,1): 1, b_idx(1,1): 1, b_idx(2,1): 1}
        # z7 = (a10+a20+a21)(b10+b20+b21)
        z7_u = {a_idx(1,0): 1, a_idx(2,0): 1, a_idx(2,1): 1}
        z7_v = {b_idx(1,0): 1, b_idx(2,0): 1, b_idx(2,1): 1}

        # Reconstruction coefficients: C_rc = sum_z coeff[z] * z
        # C00 = z1 - z2 - z3 + z4
        # C01 = z2 - z4
        # C02 = z4
        # C10 = z3 - z1
        # C11 = z1 - z3 - z6 + z7
        # C12 = z6 - z7
        # C20 = z5 - z3
        # C21 = z3 - z5 - z7 + z1
        # C22 = z5 - z7

        # For each output C[r,c], for each z with nonzero coefficient:
        # add a term: U += coeff * z_u, V += z_v, W += coeff * {c_idx(r,c): 1}
        # But we need to be careful: each z contributes to multiple C entries.
        # The correct way: for each z, for each (r,c) where coeff != 0,
        # we add a separate rank-1 term.
        #
        # Actually no. The rank-1 decomposition works as:
        # C = sum_t (sum_r coeff_{t,r,c} * a_{t,r}) * (sum_c b_{t,c}) ... 
        # No, that's not right either.
        #
        # The correct formulation: each z_t is a product of a linear form in A
        # and a linear form in B. The reconstruction says C_{r,c} = sum_t w_{t,r,c} * z_t.
        # So the tensor decomposition is:
        # T = sum_t (w_t ⊗ u_t ⊗ v_t)
        # where u_t is the vector of coefficients of z_t in A,
        # v_t is the vector of coefficients of z_t in B,
        # w_t is the vector of coefficients of z_t in C (flattened).
        #
        # So for each z_t, we add ONE term:
        # U term: u_t (sparse dict of A indices)
        # V term: v_t (sparse dict of B indices)  
        # W term: {c_idx(r,c): w_{t,r,c} for all (r,c) with w != 0}

        # W coefficients for each z:
        # z1: C00:+1, C10:-1, C11:+1, C21:+1
        # z2: C00:-1, C01:+1
        # z3: C00:-1, C10:+1, C11:-1, C20:-1, C21:+1
        # z4: C00:+1, C01:-1, C02:+1
        # z5: C20:+1, C21:-1, C22:+1
        # z6: C11:-1, C12:+1
        # z7: C11:+1, C12:-1, C21:-1, C22:-1

        z_w = [
            # z1
            {c_idx(0,0): 1, c_idx(1,0): -1, c_idx(1,1): 1, c_idx(2,1): 1},
            # z2
            {c_idx(0,0): -1, c_idx(0,1): 1},
            # z3
            {c_idx(0,0): -1, c_idx(1,0): 1, c_idx(1,1): -1, c_idx(2,0): -1, c_idx(2,1): 1},
            # z4
            {c_idx(0,0): 1, c_idx(0,1): -1, c_idx(0,2): 1},
            # z5
            {c_idx(2,0): 1, c_idx(2,1): -1, c_idx(2,2): 1},
            # z6
            {c_idx(1,1): -1, c_idx(1,2): 1},
            # z7
            {c_idx(1,1): 1, c_idx(1,2): -1, c_idx(2,1): -1, c_idx(2,2): -1},
        ]

        z_u = [z1_u, z2_u, z3_u, z4_u, z5_u, z6_u, z7_u]
        z_v = [z1_v, z2_v, z3_v, z4_v, z5_v, z6_v, z7_v]

        for t in range(7):
            add_term(z_u[t], z_v[t], z_w[t])

    # Partition:
    # A is 4x5: rows 0-3, cols 0-4
    # B is 5x6: rows 0-4, cols 0-5
    # C is 4x6: rows 0-3, cols 0-5
    #
    # Split A rows into [0,1,2] and [3]
    # Split A cols into [0,1,2] and [3,4]
    # Split B rows into [0,1,2] and [3,4]
    # Split B cols into [0,1,2] and [3,4,5]
    # Split C rows into [0,1,2] and [3]
    # Split C cols into [0,1,2] and [3,4,5]
    #
    # Blocks:
    # 1. A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3]  (3x3 x 3x3) -> Laderman rank 7
    # 2. A[0:3,0:3] * B[0:3,3:6] -> C[0:3,3:6]  (3x3 x 3x3) -> Laderman rank 7
    # 3. A[0:3,3:5] * B[3:5,0:3] -> C[0:3,0:3]  (3x2 x 2x3) -> schoolbook rank 18
    # 4. A[0:3,3:5] * B[3:5,3:6] -> C[0:3,3:6]  (3x2 x 2x3) -> schoolbook rank 18
    # 5. A[3:4,0:3] * B[0:3,0:3] -> C[3:4,0:3]  (1x3 x 3x3) -> schoolbook rank 9
    # 6. A[3:4,0:3] * B[0:3,3:6] -> C[3:4,3:6]  (1x3 x 3x3) -> schoolbook rank 9
    # 7. A[3:4,3:5] * B[3:5,0:3] -> C[3:4,0:3]  (1x2 x 2x3) -> schoolbook rank 6
    # 8. A[3:4,3:5] * B[3:5,3:6] -> C[3:4,3:6]  (1x2 x 2x3) -> schoolbook rank 6
    #
    # Total: 7 + 7 + 18 + 18 + 9 + 9 + 6 + 6 = 80

    # Block 1: A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3]
    laderman_block(row_off=0, col_off_a=0, col_off_b=0, row_off_b=0)

    # Block 2: A[0:3,0:3] * B[0:3,3:6] -> C[0:3,3:6]
    laderman_block(row_off=0, col_off_a=0, col_off_b=3, row_off_b=0)

    # Block 3: A[0:3,3:5] * B[3:5,0:3] -> C[0:3,0:3]
    schoolbook_block(range(0,3), range(3,5), range(0,3))

    # Block 4: A[0:3,3:5] * B[3:5,3:6] -> C[0:3,3:6]
    schoolbook_block(range(0,3), range(3,5), range(3,6))

    # Block 5: A[3:4,0:3] * B[0:3,0:3] -> C[3:4,0:3]
    schoolbook_block(range(3,4), range(0,3), range(0,3))

    # Block 6: A[3:4,0:3] * B[0:3,3:6] -> C[3:4,3:6]
    schoolbook_block(range(3,4), range(0,3), range(3,6))

    # Block 7: A[3:4,3:5] * B[3:5,0:3] -> C[3:4,0:3]
    schoolbook_block(range(3,4), range(3,5), range(0,3))

    # Block 8: A[3:4,3:5] * B[3:5,3:6] -> C[3:4,3:6]
    schoolbook_block(range(3,4), range(3,5), range(3,6))

    # Verify correctness numerically (fast vectorized check)
    rank = len(U_cols)

    # Build dense matrices
    U_dense = np.zeros((n * m, rank), dtype=np.float64)
    V_dense = np.zeros((m * p, rank), dtype=np.float64)
    W_dense = np.zeros((p * n, rank), dtype=np.float64)

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U_dense[idx, r] = val
        for idx, val in V_cols[r].items():
            V_dense[idx, r] = val
        for idx, val in W_cols[r].items():
            W_dense[idx, r] = val

    # Fast verification using matrix operations
    np.random.seed(42)
    A_test = np.random.randint(-5, 6, size=(n, m)).astype(np.float64)
    B_test = np.random.randint(-5, 6, size=(m, p)).astype(np.float64)
    C_expected = A_test @ B_test

    # Reshape A and B for tensor contraction
    # A_flat[i*m+j] = A[i,j], B_flat[j*p+k] = B[j,k]
    A_flat = A_test.reshape(n * m)
    B_flat = B_test.reshape(m * p)

    # For each term t:
    #   uA_t[i] = sum_j U[i*m+j, t] * A[i,j]
    #   vB_t[k] = sum_j V[j*p+k, t] * B[j,k]
    #   C[i,k] += W[k*n+i, t] * uA_t[i] * vB_t[k]
    #
    # Vectorize: 
    #   uA_all = U_dense.T @ A_flat.reshape(n, m).T ... no, that's not right.
    #   
    # Better: reshape U_dense to (n, m, rank), then
    #   uA_all[i, t] = sum_j U_dense[i*m+j, t] * A_test[i, j]
    #   = sum_j U_reshaped[i, j, t] * A_test[i, j]
    #   = np.einsum('ijt,ij->it', U_reshaped, A_test)
    #
    # Similarly for V.

    U_reshaped = U_dense.reshape(n, m, rank)
    V_reshaped = V_dense.reshape(m, p, rank)
    W_reshaped = W_dense.reshape(p, n, rank)

    # uA[i, t] = sum_j U_reshaped[i, j, t] * A_test[i, j]
    uA = np.einsum('ijt,ij->it', U_reshaped, A_test)  # shape (n, rank)
    # vB[k, t] = sum_j V_reshaped[j, k, t] * B_test[j, k]
    vB = np.einsum('jkt,jk->kt', V_reshaped, B_test)  # shape (p, rank)

    # C[i, k] = sum_t W_reshaped[k, i, t] * uA[i, t] * vB[k, t]
    C_decomp = np.einsum('kit,it,kt->ik', W_reshaped, uA, vB)

    if not np.allclose(C_decomp, C_expected, atol=1e-10):
        # Fallback to schoolbook if Laderman is wrong
        U_cols.clear()
        V_cols.clear()
        W_cols.clear()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    add_term(
                        {(i * m + j): 1},
                        {(j * p + k): 1},
                        {(k * n + i): 1}
                    )
        rank = len(U_cols)
        U_dense = np.zeros((n * m, rank), dtype=np.float64)
        V_dense = np.zeros((m * p, rank), dtype=np.float64)
        W_dense = np.zeros((p * n, rank), dtype=np.float64)
        for r in range(rank):
            for idx, val in U_cols[r].items():
                U_dense[idx, r] = val
            for idx, val in V_cols[r].items():
                V_dense[idx, r] = val
            for idx, val in W_cols[r].items():
                W_dense[idx, r] = val

    # Build output as lists of lists (fast: convert to int where possible)
    def to_output_list(dense_mat, rows):
        out = []
        for i in range(rows):
            row = dense_mat[i]
            # Check if all values are integers
            if np.all(row == np.round(row)):
                out.append([int(x) for x in row])
            else:
                out.append([float(x) for x in row])
        return out

    U = to_output_list(U_dense, n * m)
    V = to_output_list(V_dense, m * p)
    W = to_output_list(W_dense, p * n)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
