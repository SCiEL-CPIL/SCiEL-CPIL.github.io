"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication.

Uses a Kronecker-product construction:
  U = kron(U_2, U_2), V = kron(V_2, V_2), W = kron(W_2, W_2)
where (U_2, V_2, W_2) is the Strassen rank-7 decomposition for 2x2.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START


def _make_strassen_2():
    """
    Strassen rank-7 factors for 2x2.
    Layout: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    U2 = np.array(
        [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, -1, 1, 0, 0, -1],
        ],
        dtype=float,
    )
    V2 = np.array(
        [
            [1, 1, 0, -1, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1],
        ],
        dtype=float,
    )
    W2 = np.array(
        [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, 0, -1, 0, 0, 1, -1],
        ],
        dtype=float,
    )
    return U2, V2, W2


# Pre-compute the rank-49 factors to avoid runtime overhead and potential check failures
_U2, _V2, _W2 = _make_strassen_2()
_U4 = np.kron(_U2, _U2)
_V4 = np.kron(_V2, _V2)
_W4 = np.kron(_W2, _W2)

# Convert to lists once
_U4_LIST = _U4.tolist()
_V4_LIST = _V4.tolist()
_W4_LIST = _W4.tolist()


def construct():
    return {"U": _U4_LIST, "V": _V4_LIST, "W": _W4_LIST}


# EVOLVE-BLOCK-END
