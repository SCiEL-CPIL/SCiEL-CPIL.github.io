import numpy as np
import time

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # M2
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        # M3
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # M4
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        # M5
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        # M6
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # M7
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Part 1: 2x4x4 (k=0..3) - 4 blocks of 2x2x2
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)

    # Part 2: 2x4x2 (k=4..5) - 2 blocks of 2x2x2
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)

    # Part 3: 2x4x1 (k=6) - 8 rank-1 terms
    for i in range(n):
        for j in range(m):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1

    assert col == rank, f"Rank mismatch: used {col}, expected {rank}"

    # Verify correctness using numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    recon = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    if not np.allclose(recon, T, atol=1e-9):
        # Fallback: return schoolbook rank 56
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j][c] = 1
                    V_sb[j*p + k][c] = 1
                    W_sb[k*n + i][c] = 1
                    c += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Try to reduce rank by checking linear dependence
    # Matricization rank lower bound
    T_mat1 = T.reshape(n * m, -1)
    T_mat2 = T.reshape(m * p, -1)
    T_mat3 = T.reshape(p * n, -1)
    lb1 = np.linalg.matrix_rank(T_mat1)
    lb2 = np.linalg.matrix_rank(T_mat2)
    lb3 = np.linalg.matrix_rank(T_mat3)
    rank_lower = max(lb1, lb2, lb3)

    if time.time() > deadline - 5:
        return {"U": U, "V": V, "W": W}

    # Try removing columns one at a time (greedy)
    # For each column r, check if the tensor can be reconstructed without it
    # by seeing if the "residual" is in the span of remaining terms
    best_rank = rank
    best_U, best_V, best_W = U[:], V[:], W[:]

    # Convert to numpy for efficient operations
    U_cur = np.array(U, dtype=float)
    V_cur = np.array(V, dtype=float)
    W_cur = np.array(W, dtype=float)

    # Try removing each column
    for r_remove in range(rank):
        if time.time() > deadline - 10:
            break

        # Create version without column r_remove
        mask = [i for i in range(rank) if i != r_remove]
        U_try = U_cur[:, mask]
        V_try = V_cur[:, mask]
        W_try = W_cur[:, mask]

        # Check if this still reconstructs T
        recon_try = np.einsum('ar,br,cr->abc', U_try, V_try, W_try)
        if np.allclose(recon_try, T, atol=1e-9):
            # Column r_remove is redundant!
            U_cur = U_try
            V_cur = V_try
            W_cur = W_try
            best_rank = len(mask)
            best_U = U_cur.tolist()
            best_V = V_cur.tolist()
            best_W = W_cur.tolist()
            # Update mask for next iteration
            rank = best_rank
            # Re-index
            continue

    # Try pairwise removals if single removals didn't help
    if best_rank == rank and time.time() < deadline - 20:
        # Try removing pairs of columns
        for r1 in range(rank):
            if time.time() > deadline - 15:
                break
            for r2 in range(r1 + 1, rank):
                if time.time() > deadline - 15:
                    break
                mask = [i for i in range(rank) if i != r1 and i != r2]
                U_try = U_cur[:, mask]
                V_try = V_cur[:, mask]
                W_try = W_cur[:, mask]
                recon_try = np.einsum('ar,br,cr->abc', U_try, V_try, W_try)
                if np.allclose(recon_try, T, atol=1e-9):
                    U_cur = U_try
                    V_cur = V_try
                    W_cur = W_try
                    best_rank = len(mask)
                    best_U = U_cur.tolist()
                    best_V = V_cur.tolist()
                    best_W = W_cur.tolist()
                    rank = best_rank
                    break
            if best_rank < rank:
                break

    if time.time() > deadline - 2:
        if best_rank < rank:
            return {"U": best_U, "V": best_V, "W": best_W}
        return {"U": U, "V": V, "W": W}

    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    return {"U": U, "V": V, "W": W}
