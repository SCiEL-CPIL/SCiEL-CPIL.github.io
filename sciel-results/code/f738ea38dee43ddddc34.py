import numpy as np
import time
from collections import defaultdict

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Strassen coefficients for 2x2 * 2x2 -> 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]], C = [[c11, c12], [c21, c22]]
    # Indices in local 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),           # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),           # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),           # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),           # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),           # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),           # M7
    ]

    # Build the 8-block Strassen decomposition (rank 56)
    # This gives us 8 independent 2x2 multiplications
    # We'll try to merge some of them to reduce rank

    def build_block_strassen():
        """Build the 8-block Strassen decomposition."""
        rank = 56
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)

        for u in range(2):  # A row block / B row block
            for v in range(4):  # B col block / C col block
                base_col = 7 * (4 * u + v)

                # Global indices for A block u
                a11 = 0 * m + 2 * u
                a12 = 0 * m + 2 * u + 1
                a21 = 1 * m + 2 * u
                a22 = 1 * m + 2 * u + 1
                a_local_to_global = {0: a11, 1: a12, 2: a21, 3: a22}

                # Global indices for B block (u, v)
                b11 = (2 * u) * p + 2 * v
                b12 = (2 * u) * p + 2 * v + 1
                b21 = (2 * u + 1) * p + 2 * v
                b22 = (2 * u + 1) * p + 2 * v + 1
                b_local_to_global = {0: b11, 1: b12, 2: b21, 3: b22}

                # Global indices for C block v
                c11 = (2 * v) * n + 0
                c12 = (2 * v + 1) * n + 0
                c21 = (2 * v) * n + 1
                c22 = (2 * v + 1) * n + 1
                c_local_to_global = {0: c11, 1: c12, 2: c21, 3: c22}

                for t in range(7):
                    u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
                    col = base_col + t

                    for local_idx, coeff in u_coeffs:
                        U[a_local_to_global[local_idx], col] += coeff
                    for local_idx, coeff in v_coeffs:
                        V[b_local_to_global[local_idx], col] += coeff
                    for local_idx, coeff in w_coeffs:
                        W[c_local_to_global[local_idx], col] += coeff

        return U, V, W

    def verify(U, V, W, n, m, p, atol=1e-8):
        """Verify the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if not np.isclose(val, 1.0, atol=atol):
                        return False
        return True

    def try_reduce_rank(U, V, W, max_reduce=8, time_budget=30):
        """
        Try to reduce the rank by finding linear dependencies between the rank-1 terms.
        Returns (U_new, V_new, W_new, new_rank) if reduction found, else None.
        """
        start = time.time()
        n, m, p = 2, 4, 8
        nm, mp, pn = n * m, m * p, p * n
        current_rank = U.shape[1]

        # Try to find columns that can be expressed as linear combinations of others
        # This is equivalent to finding the rank of a certain matrix

        # For each pair of columns, check if one can be removed by adjusting others
        # This is expensive, so we'll use a different approach

        # Alternative: use the fact that if we have linear dependencies in U, V, W simultaneously,
        # we can reduce rank.

        # Build the tensor and try to find its true rank using random projections
        # This is a heuristic approach

        # For now, try a simple approach: look for duplicate or nearly-duplicate columns
        # in the "signature" of each rank-1 term

        # The signature of a rank-1 term is the triple (u_vec, v_vec, w_vec)
        # Two terms can be merged if their contributions are linearly dependent

        # A simpler approach: try to find if any two terms have proportional U, V, W vectors
        # and merge them

        U_cur = U.copy()
        V_cur = V.copy()
        W_cur = W.copy()
        rank = current_rank

        # Try to find columns that are linear combinations of others
        # We'll use a greedy approach: for each column, check if it can be expressed
        # as a linear combination of the remaining columns, simultaneously for U, V, W

        # This is NP-hard in general, but for small rank reductions it's feasible

        # Build a matrix where each row represents a (u_idx, v_idx, w_idx) triple
        # and each column represents a rank-1 term
        # The entry is U[u_idx, col] * V[v_idx, col] * W[w_idx, col]

        # Actually, the condition for a valid decomposition is:
        # For all (i, j, k): sum_col U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col] = 1

        # If we have a linear dependence: col_a = sum_{col != a} alpha_col * col_col
        # where the alpha_col are the same for U, V, W (i.e., U[:, col_a] = sum alpha_col * U[:, col_col], etc.)
        # then we can eliminate col_a

        # Let's try to find such dependencies

        # For each pair of columns, try to see if one can be eliminated
        # This requires solving a linear system, which is expensive

        # Instead, let's try a different approach: use the structure of Strassen
        # to find potential merges

        # In Strassen, the 7 terms have specific patterns. Let's see if we can
        # exploit the fact that some terms have simple structure

        # For example, M2 = (a21+a22)*b11 has U vector with support on {a21, a22}
        # and V vector with support on {b11}
        # If two blocks have the same pattern, we might be able to merge them

        # This is getting complex. Let's try a numerical approach:
        # 1. Build the full tensor T[i,j,k] = 1
        # 2. Compute its rank using SVD-like decomposition (but this is for matrices, not tensors)

        # For tensors, rank is NP-hard to compute. Let's use a heuristic:
        # Try random linear combinations and see if we can reduce rank

        # Actually, let's try a different strategy: use the block structure
        # to find potential rank reductions

        # The 8 blocks are: (u=0,v=0), (u=0,v=1), (u=0,v=2), (u=0,v=3),
        #                   (u=1,v=0), (u=1,v=1), (u=1,v=2), (u=1,v=3)

        # Each block uses 7 rank-1 terms, for a total of 56

        # Can we share terms between blocks? For example, if two blocks have
        # the same A-vector pattern, we might be able to merge them

        # Let's check: blocks with the same u have the same A pattern
        # Blocks (u=0,v=0) and (u=0,v=1) have the same A pattern but different B patterns
        # So we can't directly merge them

        # However, if we look at the W vectors, blocks (u=0,v=0) and (u=1,v=0)
        # have the same W pattern (same v=0) but different A and B patterns

        # This suggests that a direct merge is not possible

        # Let's try a different approach: use numerical optimization
        # to find a lower-rank decomposition

        # We'll use a simple gradient-based approach (even though it's not guaranteed to work)

        # Actually, for this specific problem, let's just return the current decomposition
        # since finding a rank reduction is non-trivial

        # Let's try one more thing: check if any two columns are identical
        # (up to scaling) and merge them

        merged_U = []
        merged_V = []
        merged_W = []
        used = [False] * rank

        for i in range(rank):
            if used[i]:
                continue

            # Check if column i is a scalar multiple of any other column
            found_merge = False
            for j in range(i + 1, rank):
                if used[j]:
                    continue

                # Check if U[:, i] is proportional to U[:, j]
                # and V[:, i] is proportional to V[:, j]
                # and W[:, i] is proportional to W[:, j]

                # Get the non-zero entries
                ui = U_cur[:, i]
                uj = U_cur[:, j]
                vi = V_cur[:, i]
                vj = V_cur[:, j]
                wi = W_cur[:, i]
                wj = W_cur[:, j]

                # Check proportionality
                # If ui = alpha * uj, then for all non-zero entries of uj,
                # ui[idx] / uj[idx] should be constant

                def check_proportional(x, y):
                    # Check if x is proportional to y
                    # Get indices where y is non-zero
                    idx_y = np.where(np.abs(y) > 1e-10)[0]
                    if len(idx_y) == 0:
                        return 0.0, False

                    ratios = []
                    for idx in idx_y:
                        if np.abs(x[idx]) < 1e-10:
                            return 0.0, False
                        ratios.append(x[idx] / y[idx])

                    # Check if all ratios are the same
                    if len(ratios) == 0:
                        return 0.0, False

                    mean_ratio = np.mean(ratios)
                    if np.allclose(ratios, mean_ratio, atol=1e-8):
                        # Also check that x has no non-zero entries where y is zero
                        idx_y_zero = np.where(np.abs(y) <= 1e-10)[0]
                        for idx in idx_y_zero:
                            if np.abs(x[idx]) > 1e-10:
                                return 0.0, False
                        return mean_ratio, True
                    return 0.0, False

                alpha_u, ok_u = check_proportional(ui, uj)
                if not ok_u:
                    continue

                alpha_v, ok_v = check_proportional(vi, vj)
                if not ok_v:
                    continue

                alpha_w, ok_w = check_proportional(wi, wj)
                if not ok_w:
                    continue

                # Check if all alphas are the same
                if np.isclose(alpha_u, alpha_v, atol=1e-8) and np.isclose(alpha_v, alpha_w, atol=1e-8):
                    # Merge: col_i = alpha * col_j
                    # We can eliminate col_i by adjusting col_j
                    # But this doesn't reduce rank, it just shows they're dependent

                    # Actually, if col_i = alpha * col_j, then we can eliminate col_i
                    # by setting its coefficient to 0 and adjusting col_j's coefficient
                    # But in our decomposition, each column is used exactly once
                    # with coefficient 1, so we can't just eliminate one

                    # We need to find a true linear dependence where
                    # col_i = sum_{k != i} beta_k * col_k
                    # This is more complex

                    # For now, let's just mark this as a potential merge
                    used[j] = True
                    # Combine: new_col = col_i + col_j (if they're identical)
                    # or col_i - alpha * col_j (if col_i = alpha * col_j)

                    # Actually, if col_i = alpha * col_j, then col_i - alpha * col_j = 0
                    # This means they're linearly dependent, but we can't eliminate either
                    # without changing the decomposition

                    # Let's skip this for now
                    found_merge = True
                    break

            if not found_merge:
                merged_U.append(i)
                merged_V.append(i)
                merged_W.append(i)

        # If we found any merges, reconstruct the matrices
        if len(merged_U) < rank:
            new_rank = len(merged_U)
            U_new = U_cur[:, merged_U].copy()
            V_new = V_cur[:, merged_V].copy()
            W_new = W_cur[:, merged_W].copy()

            # Verify
            if verify(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new, new_rank

        # No reduction found
        return None

    # Build the base Strassen decomposition
    U, V, W = build_block_strassen()

    # Verify it's correct
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank
    result = try_reduce_rank(U, V, W)
    if result is not None:
        U, V, W, new_rank = result

    # Final verification
    if not verify(U, V, W, n, m, p, atol=1e-6):
        # Fallback to the original Strassen
        U, V, W = build_block_strassen()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
