"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication.

Uses the Kronecker product of the Strassen 2x2 algorithm (rank 7) to get
a rank-49 decomposition for 4x4 matrix multiplication. The decomposition
is verified exactly before returning; on any failure it falls back to the
rank-64 schoolbook algorithm.
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _build_perm_u():
    """Permutation mapping target index S=i*4+j to kron row R."""
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_out, i_in = i // 2, i % 2
        j_out, j_in = j // 2, j % 2
        a_idx = i_out * 2 + j_out
        b_idx = i_in * 2 + j_in
        perm[S] = a_idx * 4 + b_idx
    return perm


def _build_perm_v():
    """Permutation mapping target index S=j*4+k to kron row R."""
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        a_idx = j_out * 2 + k_out
        b_idx = j_in * 2 + k_in
        perm[S] = a_idx * 4 + b_idx
    return perm


def _build_perm_w():
    """Permutation mapping target index S=k*4+i to kron row R."""
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        i = S % 4
        k_out, k_in = k // 2, k % 2
        i_out, i_in = i // 2, i % 2
        a_idx = k_out * 2 + i_out
        b_idx = k_in * 2 + i_in
        perm[S] = a_idx * 4 + b_idx
    return perm


# Precompute permutations at module load (negligible cost)
_PERM_U = _build_perm_u()
_PERM_V = _build_perm_v()
_PERM_W = _build_perm_w()


def _verify_4x4(U4, V4, W4):
    """
    Vectorized exact verification: for all (i,j,k), check
    sum_c U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c] == 1.
    Uses einsum to compute all 64 products in one shot.
    """
    # Reshape to 3D tensors:
    # U4 shape (16, 49) -> (4, 4, 49)  indexed by [i, j, c]
    # V4 shape (16, 49) -> (4, 4, 49)  indexed by [j, k, c]
    # W4 shape (16, 49) -> (4, 4, 49)  indexed by [k, i, c]
    U3 = U4.reshape(4, 4, 49)  # [i, j, c]
    V3 = V4.reshape(4, 4, 49)  # [j, k, c]
    W3 = W4.reshape(4, 4, 49)  # [k, i, c]

    # Compute T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)

    # All entries should be exactly 1
    return np.all(T == 1)


def _strassen_kron_4x4(n, m, p):
    """Rank-49 decomposition via Kronecker product of Strassen 2x2."""
    # Strassen 2x2 decomposition (rank 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Kronecker product to get 4x4 decomposition (rank 49)
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)

    # Permute rows to match linear index convention
    U4 = U4[_PERM_U, :]
    V4 = V4[_PERM_V, :]
    W4 = W4[_PERM_W, :]

    # Verify the full 4x4 decomposition (vectorized)
    if not _verify_4x4(U4, V4, W4):
        return _schoolbook(n, m, p)

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}


def construct():
    """
    Constructs a minimal-rank exact tensor decomposition for 4x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer entries.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Use rank-49 Kronecker-Strassen decomposition
    if time.time() < deadline:
        r49 = _strassen_kron_4x4(n, m, p)
        return r49

    # Final fallback: schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
