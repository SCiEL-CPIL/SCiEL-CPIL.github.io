import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # Strassen rank-7 terms
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # Build rank-35 Strassen solution
    rank = 7 * (p // 2)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            for idx in range(n * m):
                U[idx, col] = a_vec[idx]
            V[0 * p + 2 * k, col] = b_vec[0]
            V[0 * p + 2 * k + 1, col] = b_vec[1]
            V[1 * p + 2 * k, col] = b_vec[2]
            V[1 * p + 2 * k + 1, col] = b_vec[3]
            W[2 * k * n + 0, col] = c_vec[0]
            W[2 * k * n + 1, col] = c_vec[1]
            W[(2 * k + 1) * n + 0, col] = c_vec[2]
            W[(2 * k + 1) * n + 1, col] = c_vec[3]
            col += 1

    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    def verify(U_mat, V_mat, W_mat):
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T[i * m + k, k * p + j, j * n + i] = 1.0
        result = np.zeros((n * m, m * p, p * n))
        for k in range(U_mat.shape[1]):
            result += np.einsum('a,b,c->abc', U_mat[:, k], V_mat[:, k], W_mat[:, k])
        return np.allclose(result, T, atol=1e-6)

    # Try to reduce rank by finding linear dependencies among the rank-35 columns
    # This is a heuristic: check if any column is a linear combination of others
    if time.time() < deadline - 10:
        # Try random linear combinations to see if we can drop a column
        rng = np.random.default_rng(42)
        n_cols = rank

        # Check for exact linear dependencies using QR with column pivoting
        # Combine all three matrices into one tall matrix
        combined = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        combined = combined.reshape(-1, rank)
        
        # Find rank of the combined matrix
        q, r = np.linalg.qr(combined.T)
        diag = np.abs(np.diag(r))
        tol = 1e-10
        effective_rank = np.sum(diag > tol)
        
        if effective_rank < rank:
            # There's a dependency, but finding the exact decomposition is complex
            # For now, just note it
            pass

    # Verify the rank-35 solution
    if verify(U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    # Fallback to schoolbook if needed
    def schoolbook():
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_sb[i * m + k, col] = 1
                    V_sb[k * p + j, col] = 1
                    W_sb[j * n + i, col] = 1
                    col += 1
        return U_sb, V_sb, W_sb, rank_sb

    if not verify(best_U, best_V, best_W):
        best_U, best_V, best_W, best_rank = schoolbook()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
