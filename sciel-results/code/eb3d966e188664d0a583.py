"""
Rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.

Strategy:
1. Build the known rank-28 decomposition using 4 Strassen 2x2x2 blocks.
2. Run a local search to attempt rank reduction below 28.
3. Return the best valid decomposition found.
"""

import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Build the target tensor T
    # T[x, y, z] = 1 if x=4i+k, y=4k+j, z=2j+i
    # x in 0..7 (A indices: i*4+k), y in 0..15 (B indices: k*4+j), z in 0..7 (C indices: j*2+i)
    T = np.zeros((8, 16, 8), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1.0

    def build_strassen_block(U, V, W, col_offset, X_rows, Y_rows, Z_rows):
        """Add one Strassen 2x2x2 block starting at col_offset.
        X_rows: [x00, x01, x10, x11] - 4 A-indices for the 2x2 block
        Y_rows: [y00, y01, y10, y11] - 4 B-indices for the 2x2 block
        Z_rows: [z00, z01, z10, z11] - 4 C-indices for the 2x2 block
        """
        # Strassen formulas for 2x2 * 2x2 = 2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b10-b11)
        # M4 = a11*(b01-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        # c00 = M1+M4-M5+M7
        # c01 = M3+M5
        # c10 = M2+M4
        # c11 = M1-M2+M3+M6

        # M1: (a00+a11)(b00+b11)
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1

        # M2: (a10+a11)*b00
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1

        # M3: a00*(b10-b11)
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[2]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1

        # M4: a11*(b01-b00)
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[1]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1

        # M5: (a00+a01)*b11
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1

        # M6: (a10-a00)*(b00+b01)
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1

        # M7: (a01-a11)*(b10+b11)
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    def verify(U, V, W, rank):
        """Check if the decomposition exactly reproduces T."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += np.einsum('r,s,t->rst', U_np[:, r], V_np[:, r], W_np[:, r])
        return np.allclose(T_est, T, atol=1e-9)

    # Build the rank-28 Strassen decomposition
    # A is 2x4: rows 0-1, cols 0-3
    # B is 4x4: rows 0-3, cols 0-3
    # C is 2x4: rows 0-1, cols 0-3
    #
    # Partition:
    # A = [A_left | A_right] where A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    # B = [[B_tl, B_tr], [B_bl, B_br]] each 2x2
    # C = [C_left | C_right] where C_left = C[:, 0:2], C_right = C[:, 2:4]
    #
    # C_left = A_left * B_tl + A_right * B_bl  (2x2 result)
    # C_right = A_left * B_tr + A_right * B_br  (2x2 result)
    #
    # For C_left: we need A_left(2x2) * B_tl(2x2) + A_right(2x2) * B_bl(2x2)
    # This is NOT a single 2x2*2x2. It's a sum of two 2x2*2x2 products.
    #
    # Actually, let me reconsider the block structure.
    #
    # A[i,k] for i in {0,1}, k in {0,1,2,3}
    # B[k,j] for k in {0,1,2,3}, j in {0,1,2,3}
    # C[i,j] for i in {0,1}, j in {0,1,2,3}
    #
    # C[i,j] = sum_k A[i,k] * B[k,j]
    #
    # Let's partition k into {0,1} and {2,3}:
    # C[i,j] = sum_{k in {0,1}} A[i,k]*B[k,j] + sum_{k in {2,3}} A[i,k]*B[k,j]
    #
    # For fixed j-pair {j1, j2}, define:
    # C_sub[i, {j1,j2}] = A_left[i,:] * B_left[:, {j1,j2}] + A_right[i,:] * B_right[:, {j1,j2}]
    #
    # Where A_left = A[:, {0,1}] (2x2), A_right = A[:, {2,3}] (2x2)
    # B_left = B[{0,1}, {j1,j2}] (2x2), B_right = B[{2,3}, {j1,j2}] (2x2)
    #
    # So for each pair of output columns {j1,j2}, we have:
    # C_sub = A_left * B_left + A_right * B_right
    #
    # This is a 2x2 matrix that is a sum of two 2x2*2x2 products.
    # We can compute this with 14 products (7 for each 2x2*2x2) but that's 14.
    # Or we can find a smarter way.
    #
    # Actually, the known rank-28 decomposition uses 4 independent Strassen blocks.
    # Let me think about which 4 blocks they are.
    #
    # The 4 blocks correspond to the 4 combinations of:
    # - A row pair: {0,1} (only one choice since n=2)
    # - B column pair: {0,1} or {2,3}
    # - B row pair: {0,1} or {2,3}
    #
    # Wait, let me just directly construct 4 Strassen blocks:
    #
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> contributes to C[:, 0:2]
    # Block 2: A[:, 2:4] * B[2:4, 0:2] -> contributes to C[:, 0:2]
    # Block 3: A[:, 0:2] * B[0:2, 2:4] -> contributes to C[:, 2:4]
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> contributes to C[:, 2:4]
    #
    # Each block is a 2x2 * 2x2 = 2x2 multiplication done with Strassen (7 products).
    # Total: 4 * 7 = 28 products.
    #
    # For Block 1: A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    # X_rows for A: A[i, k] where i in {0,1}, k in {0,1}
    #   A[0,0]=x0, A[0,1]=x1, A[1,0]=x4, A[1,1]=x5
    #   So X_rows = [0, 1, 4, 5] mapping to [a00, a01, a10, a11]
    #
    # Y_rows for B: B[k, j] where k in {0,1}, j in {0,1}
    #   B[0,0]=y0, B[0,1]=y1, B[1,0]=y4, B[1,1]=y5
    #   So Y_rows = [0, 1, 4, 5] mapping to [b00, b01, b10, b11]
    #
    # Z_rows for C: C[i, j] where i in {0,1}, j in {0,1}
    #   C[0,0]=z0, C[0,1]=z1, C[1,0]=z2, C[1,1]=z3
    #   Wait, C index: z = j*2 + i
    #   C[0,0]: j=0,i=0 -> z=0
    #   C[0,1]: j=1,i=0 -> z=2
    #   C[1,0]: j=0,i=1 -> z=1
    #   C[1,1]: j=1,i=1 -> z=3
    #   So Z_rows = [0, 2, 1, 3] mapping to [c00, c01, c10, c11]

    rank28 = 28
    U28 = [[0] * rank28 for _ in range(8)]
    V28 = [[0] * rank28 for _ in range(16)]
    W28 = [[0] * rank28 for _ in range(8)]

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    X1 = [0, 1, 4, 5]      # A[0,0], A[0,1], A[1,0], A[1,1]
    Y1 = [0, 1, 4, 5]      # B[0,0], B[0,1], B[1,0], B[1,1]
    Z1 = [0, 2, 1, 3]      # C[0,0], C[0,1], C[1,0], C[1,1]
    build_strassen_block(U28, V28, W28, 0, X1, Y1, Z1)

    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    X2 = [2, 3, 6, 7]      # A[0,2], A[0,3], A[1,2], A[1,3]
    Y2 = [8, 9, 12, 13]    # B[2,0], B[2,1], B[3,0], B[3,1]
    Z2 = [0, 2, 1, 3]      # C[0,0], C[0,1], C[1,0], C[1,1]
    build_strassen_block(U28, V28, W28, 7, X2, Y2, Z2)

    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    X3 = [0, 1, 4, 5]      # A[0,0], A[0,1], A[1,0], A[1,1]
    Y3 = [2, 3, 6, 7]      # B[0,2], B[0,3], B[1,2], B[1,3]
    Z3 = [4, 6, 5, 7]      # C[0,2], C[0,3], C[1,2], C[1,3]
    build_strassen_block(U28, V28, W28, 14, X3, Y3, Z3)

    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    X4 = [2, 3, 6, 7]      # A[0,2], A[0,3], A[1,2], A[1,3]
    Y4 = [10, 11, 14, 15]  # B[2,2], B[2,3], B[3,2], B[3,3]
    Z4 = [4, 6, 5, 7]      # C[0,2], C[0,3], C[1,2], C[1,3]
    build_strassen_block(U28, V28, W28, 21, X4, Y4, Z4)

    # Verify rank-28 solution
    assert verify(U28, V28, W28, 28), "Rank-28 Strassen decomposition failed verification"

    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28

    # Now try to search for lower rank
    # Strategy: Try to find a rank-27 decomposition by local search
    # We'll try various approaches:
    # 1. Try to merge/remove terms from the rank-28 solution
    # 2. Try random restarts with rank 27

    def try_local_search(target_rank, max_iterations=500):
        """Try to find a valid decomposition of target_rank using local search."""
        if time.time() > deadline:
            return None

        # Initialize with the rank-28 solution, then try to reduce
        # Approach: Start with random small integers and use gradient-like updates
        # This is a hard problem; let's try a few random restarts

        for restart in range(10):
            if time.time() > deadline:
                break

            # Random initialization with small integers
            U_try = np.random.randint(-2, 3, (8, target_rank))
            V_try = np.random.randint(-2, 3, (16, target_rank))
            W_try = np.random.randint(-2, 3, (8, target_rank))

            # Compute current error
            T_est = np.zeros((8, 16, 8), dtype=float)
            for r in range(target_rank):
                T_est += np.einsum('r,s,t->rst', U_try[:, r], V_try[:, r], W_try[:, r])
            error = np.sum((T_est - T) ** 2)

            if error < 1e-9:
                return (U_try.tolist(), V_try.tolist(), W_try.tolist(), target_rank)

            # Local search: try small perturbations
            for it in range(max_iterations):
                if time.time() > deadline:
                    break

                # Pick a random entry to perturb
                mat_choice = np.random.randint(0, 3)  # 0=U, 1=V, 2=W
                if mat_choice == 0:
                    row = np.random.randint(0, 8)
                    col = np.random.randint(0, target_rank)
                    delta = np.random.choice([-1, 1])
                    U_try[row, col] += delta
                elif mat_choice == 1:
                    row = np.random.randint(0, 16)
                    col = np.random.randint(0, target_rank)
                    delta = np.random.choice([-1, 1])
                    V_try[row, col] += delta
                else:
                    row = np.random.randint(0, 8)
                    col = np.random.randint(0, target_rank)
                    delta = np.random.choice([-1, 1])
                    W_try[row, col] += delta

                # Recompute error (incremental would be faster, but let's keep it simple)
                T_est = np.zeros((8, 16, 8), dtype=float)
                for r in range(target_rank):
                    T_est += np.einsum('r,s,t->rst', U_try[:, r], V_try[:, r], W_try[:, r])
                new_error = np.sum((T_est - T) ** 2)

                if new_error < 1e-9:
                    return (U_try.tolist(), V_try.tolist(), W_try.tolist(), target_rank)

                # Accept if improved, or with small probability (simulated annealing style)
                if new_error < error:
                    error = new_error
                elif np.random.random() < 0.01:  # small probability of accepting worse
                    error = new_error

        return None

    # Try rank 27
    if time.time() < deadline - 5:
        result = try_local_search(27, max_iterations=200)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            print(f"Found rank {best_rank} solution!")

    # If we still have time, try rank 26
    if time.time() < deadline - 3:
        result = try_local_search(26, max_iterations=100)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            print(f"Found rank {best_rank} solution!")

    # Final verification
    if not verify(best_U, best_V, best_W, best_rank):
        # Fall back to the verified rank-28 solution
        best_U, best_V, best_W, best_rank = U28, V28, W28, 28

    return {"U": best_U, "V": best_V, "W": best_W}
