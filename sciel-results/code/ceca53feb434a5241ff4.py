"""Matrix multiplication 5x5x5 - Strassen hybrid decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _get_strassen_2x2():
    """Return U, V, W for 2x2 matrix multiplication using Strassen (rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    n = 2
    rank = 7
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))

    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_strassen_4x4():
    """Build Strassen 4x4 decomposition (rank 49)."""
    n = 4
    sU, sV, sW = _get_strassen_2x2()
    rank = 49
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))

    a_blocks = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    b_blocks = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]
    c_blocks = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)],
    ]

    def block_flat(bi, bj, local):
        row = bi * 2 + local // 2
        col = bj * 2 + local % 2
        return row * 4 + col

    for r in range(7):
        for s in range(7):
            idx = r * 7 + s
            for block_idx, a_coeff in a_blocks[r]:
                bi = block_idx // 2
                bj = block_idx % 2
                for local in range(4):
                    val = sU[local, s] * a_coeff
                    if val != 0:
                        U[block_flat(bi, bj, local), idx] += val
            for block_idx, b_coeff in b_blocks[r]:
                bi = block_idx // 2
                bj = block_idx % 2
                for local in range(4):
                    val = sV[local, s] * b_coeff
                    if val != 0:
                        V[block_flat(bi, bj, local), idx] += val
            for block_idx, c_coeff in c_blocks[r]:
                bi = block_idx // 2
                bj = block_idx % 2
                for local in range(4):
                    val = sW[local, s] * c_coeff
                    if val != 0:
                        W[block_flat(bi, bj, local), idx] += val

    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS

    U4, V4, W4 = _build_strassen_4x4()

    # Schoolbook terms not covered by 4x4 Strassen
    core_rank = 49
    border_terms = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i < 4 and j < 4 and k < 4:
                    continue
                border_terms.append((i, j, k))
    border_rank = len(border_terms)
    total_rank = core_rank + border_rank

    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))

    # Core: 4x4 Strassen
    for r in range(core_rank):
        for i in range(4):
            for j in range(4):
                val = U4[i * 4 + j, r]
                if val != 0:
                    U[i * 5 + j, r] = val
        for j in range(4):
            for k in range(4):
                val = V4[j * 4 + k, r]
                if val != 0:
                    V[j * 5 + k, r] = val
        for k in range(4):
            for i in range(4):
                val = W4[k * 4 + i, r]
                if val != 0:
                    W[k * 5 + i, r] = val

    # Border: schoolbook
    col = core_rank
    for (i, j, k) in border_terms:
        U[i * 5 + j, col] = 1
        V[j * 5 + k, col] = 1
        W[k * 5 + i, col] = 1
        col += 1

    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Verify with a random test
        try:
            rng = np.random.default_rng(42)
            A_t = rng.random((5, 5))
            B_t = rng.random((5, 5))
            C_exp = A_t @ B_t
            C_act = np.zeros((5, 5))
            for r in range(total_rank):
                for i in range(5):
                    for j in range(5):
                        for k in range(5):
                            C_act[i, k] += U[i * 5 + j, r] * V[j * 5 + k, r] * W[k * 5 + i, r]
            if not np.allclose(C_act, C_exp, atol=1e-10):
                # Fallback to pure schoolbook if verification fails
                rank = 125
                U2 = np.zeros((25, rank))
                V2 = np.zeros((25, rank))
                W2 = np.zeros((25, rank))
                c = 0
                for i in range(5):
                    for j in range(5):
                        for k in range(5):
                            U2[i * 5 + j, c] = 1
                            V2[j * 5 + k, c] = 1
                            W2[k * 5 + i, c] = 1
                            c += 1
                return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
        except Exception:
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
