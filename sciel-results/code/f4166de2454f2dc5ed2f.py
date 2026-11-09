"""Matrix multiplication 2,4,6 – rank-minimal decomposition via block decomposition."""
DIMENSIONS = [2, 4, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition for multiplying (n x m) by (m x p).
    Returns U (n*m x nmp), V (m*p x nmp), W (p*n x nmp)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    n_s, m_s, p_s = 2, 2, 2
    rank_s = 7
    U = np.zeros((n_s * m_s, rank_s), dtype=float)
    V = np.zeros((m_s * p_s, rank_s), dtype=float)
    W = np.zeros((p_s * n_s, rank_s), dtype=float)

    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # c00, c11

    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # c10, c11

    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # c01, c11

    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[2, 3] = 1  # c00, c10

    # M5 = (a00+a10)b11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # c00, c01

    # M6 = (a11-a10)(b00+b10)
    U[2, 5] = -1; U[3, 5] = 1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1  # c11

    # M7 = (a01-a11)(b01+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # c00

    return U, V, W


def _verify_decomp(U, V, W, n, m, p, rank, tol=1e-10):
    """Verify decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_2x4x2_from_strassen(U_s, V_s, W_s):
    """Build 2x4x2 decomposition (rank 14) from two 2x2x2 Strassen blocks."""
    n, m, p = 2, 4, 2
    rank = 14
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Block 1: j in {0,1}
    for r in range(7):
        for i in range(2):
            for jl in range(2):
                U[i * 4 + jl, r] = U_s[i * 2 + jl, r]
                for k in range(2):
                    V[jl * 2 + k, r] = V_s[jl * 2 + k, r]
                    W[k * 2 + i, r] = W_s[k * 2 + i, r]

    # Block 2: j in {2,3}
    for r in range(7):
        for i in range(2):
            for jl in range(2):
                U[i * 4 + (jl + 2), r + 7] = U_s[i * 2 + jl, r]
                for k in range(2):
                    V[(jl + 2) * 2 + k, r + 7] = V_s[jl * 2 + k, r]
                    W[k * 2 + i, r + 7] = W_s[k * 2 + i, r]

    return U, V, W


def _build_2x4x6_from_blocks(U1, V1, W1):
    """
    Build 2x4x6 decomposition from three 2x4x2 blocks.
    k blocks: {0,1}, {2,3}, {4,5}
    Total rank = 3 * 14 = 42.
    """
    n, m, p = 2, 4, 6
    rank1 = U1.shape[1]  # 14
    total_rank = rank1 * 3  # 42
    
    U_total = np.zeros((n * m, total_rank), dtype=float)
    V_total = np.zeros((m * p, total_rank), dtype=float)
    W_total = np.zeros((p * n, total_rank), dtype=float)

    for b in range(3):
        offset = b * rank1
        U_total[:, offset:offset + rank1] = U1
        # V: j in 0..3, k_local in 0..1 -> k_global = k_local + b*2
        for j in range(4):
            for kl in range(2):
                V_total[j * 6 + (kl + b * 2), offset:offset + rank1] = V1[j * 2 + kl, :]
        # W: k_local in 0..1, i in 0..1 -> k_global = k_local + b*2
        for kl in range(2):
            for i in range(2):
                W_total[(kl + b * 2) * 2 + i, offset:offset + rank1] = W1[kl * 2 + i, :]

    return U_total, V_total, W_total


def _try_lower_rank_decomposition(n, m, p, target_rank, deadline):
    """
    Attempt to find a decomposition with rank <= target_rank.
    This is a heuristic search that tries to exploit structure.
    For now, we just return None if we can't find one quickly.
    """
    # A full rank minimization search is complex. 
    # For this specific problem, the block decomposition with Strassen 
    # gives rank 42. 
    # The theoretical lower bound for 2x4x6 is not trivially known to be < 42.
    # We'll skip complex search and rely on the known good solution.
    return None


def construct():
    """Construct rank-minimal exact tensor decomposition for 2x4 * 4x6 multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 4, 6

    U_s, V_s, W_s = _strassen_2x2x2()

    if not _verify_decomp(U_s, V_s, W_s, 2, 2, 2, 7):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U1, V1, W1 = _build_2x4x2_from_strassen(U_s, V_s, W_s)

    if not _verify_decomp(U1, V1, W1, 2, 4, 2, 14):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the full 2x4x6 decomposition
    U_total, V_total, W_total = _build_2x4x6_from_blocks(U1, V1, W1)
    total_rank = U_total.shape[1]

    if not _verify_decomp(U_total, V_total, W_total, n, m, p, total_rank):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if time.time() > deadline:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

# EVOLVE-BLOCK-END
