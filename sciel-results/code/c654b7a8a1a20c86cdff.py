# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0 - 2.0  # 2 second safety margin

    n, m, p = 4, 4, 4
    rank = 49

    # Element-level Strassen data for 2x2 matrices
    U_elem = np.array([
        [1, 0, 0, 1],   # t=0: a00+a11
        [0, 0, 1, 1],   # t=1: a10+a11
        [1, 0, 0, 0],   # t=2: a00
        [0, 0, 0, 1],   # t=3: a11
        [1, 1, 0, 0],   # t=4: a00+a01
        [-1, 0, 1, 0],  # t=5: a10-a00
        [0, 1, 0, -1],  # t=6: a01-a11
    ], dtype=int)
    V_elem = np.array([
        [1, 0, 0, 1],   # t=0: b00+b11
        [1, 0, 0, 0],   # t=1: b00
        [0, 1, 0, -1],  # t=2: b01-b11
        [-1, 0, 1, 0],  # t=3: b10-b00
        [0, 0, 0, 1],   # t=4: b11
        [1, 1, 0, 0],   # t=5: b00+b01
        [0, 0, 1, 1],   # t=6: b10+b11
    ], dtype=int)
    W_elem = np.array([
        [[1, 0], [0, 1]],   # t=0
        [[0, 0], [1, -1]],  # t=1
        [[0, 1], [0, 1]],   # t=2
        [[1, 0], [1, 0]],   # t=3
        [[-1, 1], [0, 0]],  # t=4
        [[0, 0], [0, 1]],   # t=5
        [[1, 0], [0, 0]],   # t=6
    ], dtype=int)

    # Block-level Strassen data
    A_blocks = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)],
    ]
    B_blocks = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)],
    ]
    C_blocks = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]

    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)

    # Pre-compute indices for speed
    # row_a = (2 * bi + ei) * 4 + (2 * bj + ej)
    # row_b = (2 * bj + ej) * 4 + (2 * bk + ek)
    # row_c = (2 * bk + ek) * 4 + (2 * bi + ei)

    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U
            for (bi, bj, c_a) in A_blocks[s]:
                base_a = (2 * bi) * 4 + (2 * bj)
                for ei in range(2):
                    for ej in range(2):
                        row_a = base_a + ei * 4 + ej
                        val = c_a * U_elem[t, ei * 2 + ej]
                        if val != 0:
                            U[row_a, col] += val

            # V
            for (bj, bk, c_b) in B_blocks[s]:
                base_b = (2 * bj) * 4 + (2 * bk)
                for ej in range(2):
                    for ek in range(2):
                        row_b = base_b + ej * 4 + ek
                        val = c_b * V_elem[t, ej * 2 + ek]
                        if val != 0:
                            V[row_b, col] += val

            # W
            for (bi, bk, c_c) in C_blocks[s]:
                base_c = (2 * bk) * 4 + (2 * bi)
                for ei in range(2):
                    for ek in range(2):
                        row_c = base_c + ei * 4 + ek
                        val = c_c * W_elem[t, ei, ek]
                        if val != 0:
                            W[row_c, col] += val

    # Verification
    try:
        if time.time() < deadline:
            A_test = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=int)
            B_test = np.array([[16,15,14,13],[12,11,10,9],[8,7,6,5],[4,3,2,1]], dtype=int)
            AB = A_test @ B_test
            A_flat = A_test.flatten()
            B_flat = B_test.flatten()
            u_vals = U.T @ A_flat
            v_vals = V.T @ B_flat
            scalars = u_vals * v_vals
            W_flat_decomp = W @ scalars
            AB_decomp = W_flat_decomp.reshape(4, 4).T
            if not np.array_equal(AB, AB_decomp):
                # Fallback
                rank_fb = 64
                U_fb = np.zeros((16, rank_fb), dtype=int)
                V_fb = np.zeros((16, rank_fb), dtype=int)
                W_fb = np.zeros((16, rank_fb), dtype=int)
                col_fb = 0
                for i2 in range(4):
                    for j2 in range(4):
                        for k2 in range(4):
                            U_fb[i2 * 4 + j2, col_fb] = 1
                            V_fb[j2 * 4 + k2, col_fb] = 1
                            W_fb[k2 * 4 + i2, col_fb] = 1
                            col_fb += 1
                return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
