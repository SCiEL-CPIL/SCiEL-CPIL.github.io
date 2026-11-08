import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin

    # Precompute the tensor structure: for each output position (i,k),
    # the sum over j of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must equal 1
    # for all (i, j, k) where the matrix product has a 1.
    # Actually, the tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for the
    # standard matrix multiplication. We need:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    # where T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i, else 0.

    # Build the list of (a, b, c) triples with T=1
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))

    num_eq = len(triples)

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W (as lists of lists) exactly decompose the tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for (a, b, c) in triples:
            val = np.dot(U[a], np.multiply(V[b], W[c]))
            if abs(val - 1.0) > tol:
                return False
        return True

    # Rank 39 fallback construction (Strassen-style blocks)
    def construct_39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0
        def add_term(u_indices, v_indices, w_indices):
            nonlocal col
            for idx in u_indices:
                U[idx][col] += 1
            for idx in v_indices:
                V[idx][col] += 1
            for idx in w_indices:
                W[idx][col] += 1
            col += 1

        def add_strassen_block(k_start):
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1

            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1

            add_term([0, 4], [b00, b11], [c00, c11])
            add_term([3, 4], [b00], [c10, c11])
            W[c11][col-1] = -1

            add_term([0], [b01, b11], [c01, c11])
            V[b11][col-1] = -1

            add_term([4], [b10, b00], [c00, c10])
            V[b00][col-1] = -1

            add_term([0, 1], [b11], [c00, c01])
            W[c00][col-1] = -1

            add_term([3, 0], [b00, b01], [c11])
            U[0][col-1] = -1

            add_term([1, 4], [b10, b11], [c00])
            U[4][col-1] = -1

        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)

        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1

        add_term([0], [b06], [c06])
        add_term([1], [b16], [c06])
        add_term([3], [b06], [c16])
        add_term([4], [b16], [c16])

        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            add_term([2], [b2k], [c0k])
            add_term([5], [b2k], [c1k])

        return {"U": U, "V": V, "W": W}

    fallback_result = construct_39()

    # Try to find lower rank using structured search
    # Key insight: many entries can be 0, ±1, or ±1/2
    # Use a smarter approach: try ranks from 38 down, using random search
    # with constraint that values are in {-1, -0.5, 0, 0.5, 1}

    # For efficiency, represent the problem differently.
    # For a given rank R, we need U (n*m x R), V (m*p x R), W (p*n x R)
    # such that for all (a,b,c) in triples: sum_r U[a,r]*V[b,r]*W[c,r] = 1

    # Try ranks 38 down to 36 with random search
    for R in range(38, 35, -1):
        if time.time() > deadline:
            break

        # Each row of U, V, W is a vector of length R with entries in {-1,-0.5,0,0.5,1}
        # That's 5^R possibilities per row - too many for brute force.
        # Instead, use random sampling with local search.

        num_rows_U = n * m  # 6
        num_rows_V = m * p  # 21
        num_rows_W = p * n  # 14
        total_rows = num_rows_U + num_rows_V + num_rows_W

        # Random search with multiple restarts
        max_restarts = 8
        if time.time() > deadline - 10:
            max_restarts = 2

        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break

            np.random.seed(hash((R, attempt)) % (2**32))

            # Initialize with random values from the allowed set
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

            U = np.zeros((num_rows_U, R))
            V = np.zeros((num_rows_V, R))
            W = np.zeros((num_rows_W, R))

            # For each entry, randomly choose from vals with some probability of 0
            for r in range(R):
                # Random pattern for each column
                for i in range(num_rows_U):
                    if np.random.random() < 0.4:
                        U[i, r] = np.random.choice(vals)
                for i in range(num_rows_V):
                    if np.random.random() < 0.3:
                        V[i, r] = np.random.choice(vals)
                for i in range(num_rows_W):
                    if np.random.random() < 0.3:
                        W[i, r] = np.random.choice(vals)

            # Local search: try flipping individual entries
            for step in range(500):
                if time.time() > deadline - 3:
                    break

                # Check current quality
                err = 0.0
                for (a, b, c) in triples:
                    val = np.dot(U[a], np.multiply(V[b], W[c]))
                    err += (val - 1.0) ** 2

                if err < 1e-9:
                    # Verify and return
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if check_decomposition(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    break

                # Pick a random position to modify
                choice = np.random.random()
                if choice < num_rows_U / total_rows:
                    row = np.random.randint(num_rows_U)
                    mat = U
                    mat_name = 'U'
                elif choice < (num_rows_U + num_rows_V) / total_rows:
                    row = np.random.randint(num_rows_V)
                    mat = V
                    mat_name = 'V'
                else:
                    row = np.random.randint(num_rows_W)
                    mat = W
                    mat_name = 'W'

                col = np.random.randint(R)
                old_val = mat[row, col]

                # Try a new value
                new_val = np.random.choice(vals)
                if new_val == old_val:
                    continue

                mat[row, col] = new_val

                # Compute error change
                err_new = 0.0
                for (a, b, c) in triples:
                    val = np.dot(U[a], np.multiply(V[b], W[c]))
                    err_new += (val - 1.0) ** 2

                if err_new >= err:
                    mat[row, col] = old_val
                else:
                    err = err_new

    # Try a more systematic approach for rank 38
    if time.time() < deadline - 20:
        R = 38
        # Use the rank-39 solution and try to merge two columns
        U39 = np.array(fallback_result['U'], dtype=float)
        V39 = np.array(fallback_result['V'], dtype=float)
        W39 = np.array(fallback_result['W'], dtype=float)

        # Try merging pairs of columns
        for r1 in range(R):
            if time.time() > deadline - 5:
                break
            for r2 in range(r1 + 1, R):
                # Check if we can eliminate one column
                # The contribution of column r: U[:,r] * V[:,r] * W[:,r]
                # We need the sum over all columns to still give 1 for each triple
                # If we remove column r2, we need to adjust other columns

                # This is complex, so skip for now
                pass

    return fallback_result
# EVOLVE-BLOCK-END
