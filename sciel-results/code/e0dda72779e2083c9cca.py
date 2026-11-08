import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Strassen 2x2x2
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W

    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W

    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W

    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W

    U_parts = []
    V_parts = []
    W_parts = []

    def add_merged(u_idx_a, u_idx_b, v_idx_a, v_idx_b, w_idx, U_blk_a, V_blk_a, W_blk_a, U_blk_b, V_blk_b, W_blk_b):
        """Merge two blocks that share the same W pattern."""
        for r in range(U_blk_a.shape[1]):
            U = np.zeros(n * m)
            V = np.zeros(m * p)
            W = np.zeros(p * n)
            for i, idx in enumerate(u_idx_a):
                U[idx] += U_blk_a[i, r]
            for i, idx in enumerate(u_idx_b):
                U[idx] += U_blk_b[i, r]
            for i, idx in enumerate(v_idx_a):
                V[idx] += V_blk_a[i, r]
            for i, idx in enumerate(v_idx_b):
                V[idx] += V_blk_b[i, r]
            for i, idx in enumerate(w_idx):
                W[idx] = W_blk_a[i, r]  # W is the same pattern
            U_parts.append(U)
            V_parts.append(V)
            W_parts.append(W)

    def add_single(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            U = np.zeros(n * m)
            V = np.zeros(m * p)
            W = np.zeros(p * n)
            for i, idx in enumerate(u_idx):
                U[idx] = U_blk[i, r]
            for i, idx in enumerate(v_idx):
                V[idx] = V_blk[i, r]
            for i, idx in enumerate(w_idx):
                W[idx] = W_blk[i, r]
            U_parts.append(U)
            V_parts.append(V)
            W_parts.append(W)

    # Pair 1: W indices [0,3,1,4] - Strassen blocks 1 & 4
    add_merged(
        [0, 1, 4, 5], [2, 3, 6, 7],
        [0, 1, 5, 6], [10, 11, 15, 16],
        [0, 3, 1, 4],
        *strassen_2x2x2(), *strassen_2x2x2()
    )

    # Pair 2: W indices [6,9,7,10] - Strassen blocks 2 & 5
    add_merged(
        [0, 1, 4, 5], [2, 3, 6, 7],
        [2, 3, 7, 8], [12, 13, 17, 18],
        [6, 9, 7, 10],
        *strassen_2x2x2(), *strassen_2x2x2()
    )

    # Pair 3: W indices [12,13] - Schoolbook 2x2x1 blocks 3 & 6
    add_merged(
        [0, 1, 4, 5], [2, 3, 6, 7],
        [4, 9], [14, 19],
        [12, 13],
        *schoolbook_2x2x1(), *schoolbook_2x2x1()
    )

    # Pair 4: W indices [2,5] - Schoolbook 1x2x2 blocks 7 & 10
    add_merged(
        [8, 9], [10, 11],
        [0, 1, 5, 6], [10, 11, 15, 16],
        [2, 5],
        *schoolbook_1x2x2(), *schoolbook_1x2x2()
    )

    # Pair 5: W indices [8,11] - Schoolbook 1x2x2 blocks 8 & 11
    add_merged(
        [8, 9], [10, 11],
        [2, 3, 7, 8], [12, 13, 17, 18],
        [8, 11],
        *schoolbook_1x2x2(), *schoolbook_1x2x2()
    )

    # Pair 6: W indices [14] - Schoolbook 1x2x1 blocks 9 & 12
    add_merged(
        [8, 9], [10, 11],
        [4, 9], [14, 19],
        [14],
        *schoolbook_1x2x1(), *schoolbook_1x2x1()
    )

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Remove columns where any factor is entirely zero
    rank = U.shape[1]
    keep = []
    for r in range(rank):
        if np.any(U[:, r] != 0) and np.any(V[:, r] != 0) and np.any(W[:, r] != 0):
            keep.append(r)
    if len(keep) < rank:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
