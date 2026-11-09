import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = 4, 4, 6

    # Strassen 2x2 x 2x2 coefficients
    alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    # Build the 77-rank decomposition
    R = 77
    U = np.zeros((16, R), dtype=int)
    V = np.zeros((24, R), dtype=int)
    W = np.zeros((24, R), dtype=int)

    col = 0
    for s in range(7):
        if time.time() > deadline:
            break
        # Inner Strassen for columns 0,1 of the 4x6 result
        for si in range(7):
            for a in range(4):
                if alpha[si][a] == 0:
                    continue
                x_row, x_col = a // 2, a % 2
                for r in range(4):
                    if alpha[s][r] == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                    U[u_idx, col] += alpha[si][a] * alpha[s][r]

            for b in range(4):
                if beta[si][b] == 0:
                    continue
                y_row, y_inner_col = b // 2, b % 2
                for c in range(4):
                    if beta[s][c] == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                    V[v_idx, col] += beta[si][b] * beta[s][c]

            for (a, b, gamma) in recon[s]:
                for (za, zb, zgamma) in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma

            col += 1

        # Schoolbook for column 2 of the 4x6 result
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                    U[u_idx, col] += alpha[s][r]

                for c in range(4):
                    if beta[s][c] == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                    V[v_idx, col] += beta[s][c]

                for (a, b, gamma) in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma

                col += 1

    # Try to reduce rank by removing zero columns and merging
    # Check for zero columns
    nonzero_mask = np.any(U != 0, axis=0) & np.any(V != 0, axis=0) & np.any(W != 0, axis=0)
    
    if not np.all(nonzero_mask):
        keep = np.where(nonzero_mask)[0]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]

    # Also check for duplicate columns that can be merged
    # Build a hash of each column
    def col_hash(mat, col):
        return tuple(mat[:, col].tolist())
    
    R_final = U.shape[1]
    col_keys = {}
    for j in range(R_final):
        key = (col_hash(U, j), col_hash(V, j), col_hash(W, j))
        if key not in col_keys:
            col_keys[key] = []
        col_keys[key].append(j)
    
    # Keep only one copy of each unique column
    new_cols = []
    for key, indices in col_keys.items():
        new_cols.append(indices[0])
    
    if len(new_cols) < R_final:
        U = U[:, new_cols]
        V = V[:, new_cols]
        W = W[:, new_cols]

    # Attempt to find a lower-rank decomposition via random search
    # We'll try to find a decomposition with rank less than the current one
    current_rank = U.shape[1]
    best_rank = current_rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()

    # Try to reduce rank by finding linear dependencies
    # We can try to express some columns as linear combinations of others
    # This is a hard problem, so we'll use a simple heuristic

    # Check if we can find a lower rank by looking at the tensor structure
    # For matrix multiplication, the rank is at least max(n*m, m*p, p*n) / something
    # For 4x4x6, the trivial bound is 4*4*6 = 96, but we have 77

    # Let's try a different approach: use the fact that we can sometimes
    # combine terms if they share the same output position

    # For now, return the current decomposition
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
