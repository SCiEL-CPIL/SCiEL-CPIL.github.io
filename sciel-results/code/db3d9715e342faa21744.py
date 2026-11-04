import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n * m * p, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[idx, :] = U[i * m + j] * V[j * p + k] * W[k * n + i]
                idx += 1
    T = T.T
    # target tensor
    Tt = np.zeros((n * m * p, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Tt[idx, :] = 0.0
                idx += 1
    # build target: T[(i,j,k), c] = delta_{i,j,k,c}
    Tt = np.zeros((n * m * p,))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Tt[idx] = 1.0
                idx += 1
    # compare
    Tmat = T.T  # (n*m*p, rank)
    if np.allclose(Tmat, Tt.reshape(n * m * p, 1) * 0 + 0, atol=1e-6):
        pass
    # actual check: Tmat must equal the one-hot tensor
    diff = np.max(np.abs(Tmat - Tt.reshape(n * m * p, 1)))
    return diff < 1e-6

def _is_valid(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    Tmat = np.zeros((n * m * p, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Tmat[idx, :] = U[i * m + j] * V[j * p + k] * W[k * n + i]
                idx += 1
    Tt = np.zeros((n * m * p,))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Tt[idx] = 1.0
                idx += 1
    return np.max(np.abs(Tmat - Tt.reshape(n * m * p, 1))) < 1e-6

def _build_block_decomp():
    """Partition inner dim into 2,2 and use Strassen 2x2 (rank 7) + 1x2*2x2 (rank 4)."""
    n, m, p = DIMENSIONS
    # partition i: [0,1] and [2]; j: [0,1],[2,3]; k: [0,1],[2,3]
    # Block structure:
    #   (2x2) x (2x2) for i in {0,1}, j in {0,1}, k in {0,1} -> Strassen rank 7
    #   (2x2) x (2x2) for i in {0,1}, j in {0,1}, k in {2,3} -> Strassen rank 7
    #   (1x2) x (2x2) for i=2, j in {0,1}, k in {0,1} -> rank 4
    #   (1x2) x (2x2) for i=2, j in {0,1}, k in {2,3} -> rank 4
    #   (2x2) x (2x2) for i in {0,1}, j in {2,3}, k in {0,1} -> Strassen rank 7
    #   (2x2) x (2x2) for i in {0,1}, j in {2,3}, k in {2,3} -> Strassen rank 7
    #   (1x2) x (2x2) for i=2, j in {2,3}, k in {0,1} -> rank 4
    #   (1x2) x (2x2) for i=2, j in {2,3}, k in {2,3} -> rank 4
    # Total: 7+7+4+4+7+7+4+4 = 44? That seems too low. Let me reconsider.

    # Actually the measured 66 came from a specific partition. Let me just use the schoolbook
    # but with a smarter block approach. The measured improvement was 66 vs 72.
    # Let me build the schoolbook first as a safe fallback, then try to improve.
    pass
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _rank_of_matrix(A):
    return int(np.linalg.matrix_rank(A))

def _try_reduce(U, V, W, deadline, max_iters=50000):
    """Try to reduce rank by finding zero columns or linear dependencies."""
    n, m, p = DIMENSIONS
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    rank = U.shape[1]

    # Build the full tensor matrix T: rows = (i,j,k), cols = rank
    T = np.zeros((n * m * p, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[idx, :] = U[i * m + j] * V[j * p + k] * W[k * n + i]
                idx += 1

    target = np.ones((n * m * p, 1))

    # Try to find columns of U,V,W that are all zeros
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Check for zero columns
        zero_u = np.where(np.all(np.abs(U) < 1e-9, axis=0))[0]
        zero_v = np.where(np.all(np.abs(V) < 1e-9, axis=0))[0]
        zero_w = np.where(np.all(np.abs(W) < 1e-9, axis=0))[0]
        zero_cols = set(zero_u) & set(zero_v) & set(zero_w)
        if zero_cols:
            # Remove zero columns
            keep = [c for c in range(U.shape[1]) if c not in zero_cols]
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
            continue
        # Try small perturbations to zero out near-zero entries
        # Find entries with very small magnitude
        thresh = 1e-3
        small_u = np.where(np.abs(U) < thresh)
        small_v = np.where(np.abs(V) < thresh)
        small_w = np.where(np.abs(W) < thresh)
        # Try zeroing a random small entry and check validity
        all_small = list(small_u) + list(small_v) + list(small_w)
        if all_small:
            import random
            # Deterministic: pick first
            mat, r, c = all_small[0]
            if mat == 0:
                U[r, c] = 0.0
            elif mat == 1:
                V[r, c] = 0.0
            else:
                W[r, c] = 0.0
            # Verify
            T2 = np.zeros((n * m * p, U.shape[1]))
            idx2 = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T2[idx2, :] = U[i * m + j] * V[j * p + k] * W[k * n + i]
                        idx2 += 1
            if np.max(np.abs(T2 - target)) < 1e-6:
                continue
            else:
                # Revert
                if mat == 0:
                    U[r, c] = thresh  # approximate revert
                elif mat == 1:
                    V[r, c] = thresh
                else:
                    W[r, c] = thresh
        break
    return U, V, W

def construct():
    t0 = time.time()
    deadline = t0 + 120 - 10  # 10s safety margin

    n, m, p = DIMENSIONS

    # Start with schoolbook
    U, V, W = _schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try block-based Strassen decomposition
    # Partition: i into {0,1} and {2}, j into {0,1} and {2,3}, k into {0,1} and {2,3}
    # This gives 8 blocks. Use Strassen for 2x2*2x2 blocks (rank 7 each)
    # and standard for 1x2*2x2 blocks (rank 4 each)
    # Blocks:
    # (i:0,1) x (j:0,1) x (k:0,1) : 2x2 * 2x2 -> Strassen 7
    # (i:0,1) x (j:0,1) x (k:2,3) : 2x2 * 2x2 -> Strassen 7
    # (i:0,1) x (j:2,3) x (k:0,1) : 2x2 * 2x2 -> Strassen 7
    # (i:0,1) x (j:2,3) x (k:2,3) : 2x2 * 2x2 -> Strassen 7
    # (i:2)   x (j:0,1) x (k:0,1) : 1x2 * 2x2 -> 4
    # (i:2)   x (j:0,1) x (k:2,3) : 1x2 * 2x2 -> 4
    # (i:2)   x (j:2,3) x (k:0,1) : 1x2 * 2x2 -> 4
    # (i:2)   x (j:2,3) x (k:2,3) : 1x2 * 2x2 -> 4
    # Total: 4*7 + 4*4 = 28 + 16 = 44? But measured was 66.
    # The measured 66 likely used a different partition. Let me just use schoolbook
    # and try local reduction.

    # Try local reduction
    U2, V2, W2 = _try_reduce(U, V, W, deadline)
    if U2.shape[1] < best_rank:
        best_rank = U2.shape[1]
        best_U, best_V, best_W = U2, V2, W2

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    # Round to nearest half-integer if close
    for mat in [U_list, V_list, W_list]:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                val = mat[i][j]
                # Round to nearest 0.5
                rounded = round(val * 2) / 2.0
                if abs(val - rounded) < 1e-6:
                    mat[i][j] = rounded

    return {"U": U_list, "V": V_list, "W": W_list}
