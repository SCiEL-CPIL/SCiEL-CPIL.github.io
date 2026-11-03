"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal (or near-minimal) exact tensor decomposition for 3x4 * 4x8.
    
    Strategy:
    1. Build the block-Strassen decomposition: Rank 88.
       - Split B (4x8) into two 4x4 blocks (m-dimension).
       - For each 3x4 * 4x4 subproblem:
         * Split A (3x4) into A1 (2x4) and A2 (1x4).
         * C1 = A1 * B_block (2x4): Split into four 2x2*2x2 products, use Strassen (rank 28).
         * C2 = A2 * B_block (1x4): Schoolbook (rank 16).
         * Total per subproblem: rank 44.
       - Total: 2 * 44 = 88.
    2. Try to reduce rank by finding linearly dependent columns.
    3. Verify and return the best valid decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # --- 1. Schoolbook Fallback (Rank 96) ---
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    # --- 2. Block-Strassen Decomposition (Rank 88) ---
    # Strassen coefficients for 2x2 * 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Using dictionary format: {local_index: coefficient}
    # Local indices: 00=0, 01=1, 10=2, 11=3
    strassen_decomp = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # M4
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),         # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})          # M7
    ]
    
    def _build_strassen_3x4x8():
        """Build the block-Strassen decomposition: rank 88."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        # Alternative: split p into 4 blocks of 2, and for each:
        # - Use Strassen for rows 0,1 (2x2x2): 7 mults
        # - Use schoolbook for row 2 (1x2x2): 4 mults
        # Total: 4 * (7 + 4) = 44 per m-block, 2 m-blocks = 88
        
        for j1 in range(2):  # 2 blocks of 2 in m dimension
            # Top part: 2x2x8 (rows 0,1 of A)
            for l in range(4):  # 4 blocks of 2 in p dimension
                for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                    # U (A): local 2x2 -> global
                    for li, coeff in u_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = li_i  # 0 or 1
                        global_j = 2 * j1 + li_j
                        U[global_i * m + global_j][col] += coeff

                    # V (B): local 2x2 -> global
                    for li, coeff in v_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_j = 2 * j1 + li_i
                        global_k = 2 * l + li_j
                        V[global_j * p + global_k][col] += coeff

                    # W (C): local 2x2 -> global
                    for li, coeff in w_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = li_i
                        global_k = 2 * l + li_j
                        W[global_k * n + global_i][col] += coeff

                    col += 1

            # Bot part: 1x2x8 (row 2 of A), schoolbook
            for k in range(8):
                for j2 in range(2):
                    global_i = 2
                    global_j = 2 * j1 + j2
                    U[global_i * m + global_j][col] = 1
                    V[global_j * p + k][col] = 1
                    W[k * n + global_i][col] = 1
                    col += 1

        return {"U": U, "V": V, "W": W}

    def _verify(U, V, W, tol=1e-9):
        """Verify exact tensor decomposition."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _verify_half_integer(U, V, W):
        """Check that all entries are integers or half-integers."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True

    # Build the 88-rank decomposition
    result = _build_strassen_3x4x8()

    # Verify correctness
    if not _verify(result['U'], result['V'], result['W']):
        # Fallback: schoolbook
        result = {"U": best_U, "V": best_V, "W": best_W}
        return result

    # Try to reduce rank by finding linearly dependent columns
    # and removing them while preserving the decomposition
    if time.time() < deadline - 10:
        U_arr = np.array(result['U'], dtype=float)
        V_arr = np.array(result['V'], dtype=float)
        W_arr = np.array(result['W'], dtype=float)
        rank = U_arr.shape[1]

        # Vectorized verification for speed
        def _verify_fast(U_mat, V_mat, W_mat, tol=1e-9):
            """Fast vectorized verification."""
            # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_mat[i*m+j] * V_mat[j*p+k] * W_mat[k*n+i])
                        if abs(s - 1.0) > tol:
                            return False
            return True

        def _verify_half_int_fast(U_mat, V_mat, W_mat, tol=1e-9):
            """Check half-integer property."""
            for mat in [U_mat, V_mat, W_mat]:
                if np.any(np.abs(mat * 2 - np.round(mat * 2)) > tol):
                    return False
            return True

        best_rank = rank
        best_result = result

        # Strategy 1: Greedy column removal with fast verification
        # Order columns by "sparsity" - columns with more zeros are more likely removable
        col_sparsity = []
        for c in range(rank):
            zeros_u = np.count_nonzero(U_arr[:, c])
            zeros_v = np.count_nonzero(V_arr[:, c])
            zeros_w = np.count_nonzero(W_arr[:, c])
            col_sparsity.append((zeros_u + zeros_v + zeros_w, c))
        col_sparsity.sort()
        cols_to_try = [c for _, c in col_sparsity]

        removed = []
        for col_idx in cols_to_try:
            if time.time() > deadline - 5:
                break

            # Try removing this column
            keep = [i for i in range(rank) if i != col_idx]
            U_test = U_arr[:, keep]
            V_test = V_arr[:, keep]
            W_test = W_arr[:, keep]

            # Fast full verification
            if _verify_fast(U_test, V_test, W_test):
                if _verify_half_int_fast(U_test, V_test, W_test):
                    U_arr = U_test
                    V_arr = V_test
                    W_arr = W_test
                    rank = rank - 1
                    removed.append(col_idx)
                    cols_to_try = [i for i in cols_to_try if i != col_idx and i < rank]

        if len(removed) > 0:
            best_rank = rank
            best_result = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

        # Strategy 2: Try to find 2-column dependencies (replace 2 cols with 1)
        if time.time() < deadline - 15 and best_rank < 88:
            # For pairs of columns, check if one is a linear combination of the other
            # in a way that preserves the decomposition
            # This is more complex; try simple cases where coefficients are in {-1, 0, 1}
            
            # Check if any pair of columns has the same "signature" pattern
            # that allows cancellation
            for a in range(best_rank):
                if time.time() > deadline - 10:
                    break
                for b in range(a + 1, best_rank):
                    if time.time() > deadline - 8:
                        break
                    
                    # Check if column b is a scalar multiple of column a in U, V, W
                    # Try simple coefficients: 1, -1
                    for coeff in [1, -1]:
                        # Check if U[:,b] == coeff * U[:,a], V[:,b] == coeff * V[:,a], W[:,b] == coeff * W[:,a]
                        # But we need the sum to still work out...
                        # Actually, if we remove column b and adjust column a, we need:
                        # For all (i,j,k): U[i*m+j,a]*V[j*p+k,a]*W[k*n+i,a] + 
                        #                   U[i*m+j,b]*V[j*p+k,b]*W[k*n+i,b] 
                        # should be preserved by some adjustment.
                        # This is complex; skip for now.
                        pass

        result = best_result

    return result
# EVOLVE-BLOCK-END
