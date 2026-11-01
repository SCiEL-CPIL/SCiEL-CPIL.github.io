"""Matrix multiplication 2x4x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0

    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices

        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1

        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1

        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1

        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1

        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1

        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1

        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1

    # Block 1: A_L B_11
    strassen_block([(0, 0), (0, 1), (1, 0), (1, 1)],
                   [(0, 0), (0, 1), (1, 0), (1, 1)],
                   [(0, 0), (1, 0), (0, 1), (1, 1)])
    # Block 2: A_R B_21
    strassen_block([(0, 2), (0, 3), (1, 2), (1, 3)],
                   [(2, 0), (2, 1), (3, 0), (3, 1)],
                   [(0, 0), (1, 0), (0, 1), (1, 1)])
    # Block 3: A_L B_12
    strassen_block([(0, 0), (0, 1), (1, 0), (1, 1)],
                   [(0, 2), (0, 3), (1, 2), (1, 3)],
                   [(2, 0), (3, 0), (2, 1), (3, 1)])
    # Block 4: A_R B_22
    strassen_block([(0, 2), (0, 3), (1, 2), (1, 3)],
                   [(2, 2), (2, 3), (3, 2), (3, 3)],
                   [(2, 0), (3, 0), (2, 1), (3, 1)])

    # Block 5: A_L B_15 (2x2 * 2x1)
    for (ii, jj) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        U[u_idx(ii, jj), col] = 1
        V[v_idx(ii, 4), col] = 1
        W[w_idx(4, jj), col] += 1
        col += 1

    # Block 6: A_R B_25 (2x2 * 2x1)
    for (ii, jj) in [(0, 2), (0, 3), (1, 2), (1, 3)]:
        U[u_idx(ii, jj), col] = 1
        V[v_idx(ii + 2, 4), col] = 1
        W[w_idx(4, jj - 2), col] += 1
        col += 1

    return U, V, W


def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _build_flat_tensors(U, V, W):
    """Build the (rank, dim) matrix where each row is a flattened rank-1 tensor."""
    rank = U.shape[1]
    dim = U.shape[0] * V.shape[0] * W.shape[0]
    T = np.zeros((rank, dim), dtype=np.float64)
    for c in range(rank):
        T[c, :] = np.kron(np.kron(U[:, c], V[:, c]), W[:, c])
    return T


def _find_removable(U, V, W, T, deadline, max_check=8):
    """
    Find columns whose rank-1 tensor is in the span of the others.
    Uses rank comparison: rank([others; t_c]) == rank(others) means t_c is removable.
    Returns a list of removable column indices.
    """
    rank = U.shape[1]
    removable = []

    for c in range(rank):
        if time.time() > deadline:
            break
        others = [T[r, :] for r in range(rank) if r != c]
        if not others:
            continue
        A = np.array(others)
        try:
            r1 = np.linalg.matrix_rank(A, tol=1e-8)
            combined = np.vstack([A, T[c:c + 1, :]])
            r2 = np.linalg.matrix_rank(combined, tol=1e-8)
            if r1 == r2:
                removable.append(c)
        except np.linalg.LinAlgError:
            pass

    return removable


def _try_remove_columns(U, V, W, cols_to_remove, tol=1e-9):
    """Remove specified columns and verify the decomposition still holds."""
    rank = U.shape[1]
    mask = [i for i in range(rank) if i not in cols_to_remove]
    if len(mask) == 0:
        return None
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    if _verify_decomposition(U_new, V_new, W_new, tol):
        return U_new, V_new, W_new
    return None


def _greedy_removal(U, V, W, deadline):
    """
    Iteratively try to remove columns one at a time.
    After each removal, rebuild the flat tensor matrix and search for new redundancies.
    Continue until no more removable columns are found or deadline is reached.
    """
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    
    while time.time() < deadline:
        T = _build_flat_tensors(current_U, current_V, current_W)
        removable = _find_removable(current_U, current_V, current_W, T, deadline)
        
        if not removable:
            break
        
        # Try removing all identified removable columns at once
        result = _try_remove_columns(current_U, current_V, current_W, set(removable))
        if result is not None:
            current_U, current_V, current_W = result
        else:
            # Try removing them one at a time
            for c in removable:
                if time.time() > deadline:
                    break
                result = _try_remove_columns(current_U, current_V, current_W, {c})
                if result is not None:
                    current_U, current_V, current_W = result
                    # Rebuild and continue outer loop to find more
                    break
    
    return current_U, current_V, current_W


def _pairwise_removal(U, V, W, deadline):
    """Try removing pairs of columns simultaneously."""
    rank = U.shape[1]
    from itertools import combinations
    
    for c1, c2 in combinations(range(rank), 2):
        if time.time() > deadline:
            break
        result = _try_remove_columns(U, V, W, {c1, c2})
        if result is not None:
            return result
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin before 150s limit

    # Start with the known rank-36 decomposition
    U, V, W = _build_strassen_36()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    if not _verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Phase 1: Algebraic dependency detection and greedy removal
    if time.time() < deadline:
        U, V, W = _greedy_removal(U, V, W, deadline)
        if U.shape[1] < best_U.shape[1]:
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Phase 2: Pairwise removal on the best result
    if time.time() < deadline:
        result = _pairwise_removal(best_U, best_V, best_W, deadline)
        if result is not None:
            U2, V2, W2 = result
            if U2.shape[1] < best_U.shape[1]:
                best_U, best_V, best_W = U2.copy(), V2.copy(), W2.copy()

    # Phase 3: Another round of greedy removal on the best result
    if time.time() < deadline:
        U3, V3, W3 = _greedy_removal(best_U, best_V, best_W, deadline)
        if U3.shape[1] < best_U.shape[1]:
            best_U, best_V, best_W = U3.copy(), V3.copy(), W3.copy()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
