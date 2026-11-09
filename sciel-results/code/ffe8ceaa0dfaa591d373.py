"""Rank-28 exact tensor decomposition for 2x4 x 4x4 matrix multiplication."""
import numpy as np
import time

def construct():
    """Return factor matrices U, V, W of a rank-28 decomposition of the [2,4,4] MM tensor."""
    t0 = time.time()
    n, m, p = 2, 4, 4

    # Build the 7-term Strassen decomposition for a single 2x2x2 block.
    # Indices are (i,j) for U, (j,k) for V, (k,i) for W, each in {0,1}^2.
    # term t gives U[i,j] += a_t, V[j,k] += b_t, W[k,i] += c_t.
    strassen = [
        # (U entries: list of (i,j,coeff)), (V entries), (W entries)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],         [(0,1,1),(1,1,-1)]),
        ([(0,0,1)],         [(1,0,1),(1,1,-1)],[(1,0,1),(1,1,1)]),
        ([(1,1,1)],         [(1,0,1),(0,0,-1)],[(0,0,1),(0,1,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)],         [(0,0,-1),(1,0,1)]),
        ([(1,0,1),(0,0,-1)],[(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)],[(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    # Four 2x2x2 blocks: (j_offset, k_offset) in {0,2}^2
    # i is always 0 or 1 (n=2), so no i_offset needed.
    blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]

    rank = 7 * len(blocks)  # 28
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0
    for (j_off, k_off) in blocks:
        for (u_entries, v_entries, w_entries) in strassen:
            for (i, j, c) in u_entries:
                U[i * m + (j + j_off), col] += c
            for (j, k, c) in v_entries:
                V[j * p + (k + k_off), col] += c
            for (k, i, c) in w_entries:
                W[k * n + i, col] += c
            col += 1

    # Convert to list-of-lists of ints for the required interface
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()

    # Safety: verify the decomposition if time allows (optional, fast for rank 28)
    if time.time() - t0 < 100:
        _verify_decomposition(U_out, V_out, W_out, n, m, p)

    return {"U": U_out, "V": V_out, "W": W_out}


def _verify_decomposition(U, V, W, n, m, p):
    """Check that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 iff (i,k) appears in A[i,:]*B[:,k]."""
    # The tensor T[i,j,k] = 1 for all (i,j,k) in the standard MM tensor.
    # Our decomposition should satisfy: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    rank = Ua.shape[1]
    # Compute T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += int(Ua[i * m + j, r]) * int(Va[j * p + k, r]) * int(Wa[k * n + i, r])
                if s != 1:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {s}")
