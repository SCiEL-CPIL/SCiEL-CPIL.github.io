import numpy as np
import time

def _verify_block(U, V, W):
    # U: (6, r), V: (9, r), W: (6, r) for M_{2,3,3}
    # T[i*3+j, j*3+k, k*2+i] = 1
    n, m, p = 2, 3, 3
    r = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=1e-9)

def _verify_full(U, V, W):
    # U: (6, R), V: (27, R), W: (18, R) for M_{2,3,9}
    n, m, p = 2, 3, 9
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=1e-9)

def construct():
    deadline = time.time() + 135  # safety margin before 150s kill

    n, m, p = 2, 3, 9
    # Schoolbook fallback (rank 54)
    R_sb = n * m * p
    U_sb = np.zeros((n*m, R_sb))
    V_sb = np.zeros((m*p, R_sb))
    W_sb = np.zeros((p*n, R_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best = (U_sb.copy(), V_sb.copy(), W_sb.copy())

    # Rank-5 M_{2,3,3} decomposition (candidate). Indices: U rows 0..5 (i*3+j),
    # V rows 0..8 (j*3+k), W rows 0..5 (k*2+i).
    U5 = np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1],
    ], dtype=float)
    V5 = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=float)
    W5 = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1],
    ], dtype=float)

    # Verify the block decomposition
    if _verify_block(U5, V5, W5):
        # Build the rank-15 M_{2,3,9} decomposition as direct sum of 3 blocks.
        # Block ell uses V columns 3*ell..3*ell+2 and W rows 3*ell..3*ell+2.
        R15 = 15
        U15 = np.zeros((n*m, R15))
        V15 = np.zeros((m*p, R15))
        W15 = np.zeros((p*n, R15))
        for ell in range(3):
            s = ell * 5
            U15[:, s:s+5] = U5
            V15[3*ell:3*ell+3, s:s+5] = V5
            W15[3*ell:3*ell+3, s:s+5] = W5
        if _verify_full(U15, V15, W15):
            best = (U15, V15, W15)

    # If the hardcoded candidate failed, try a quick local search to reduce rank.
    # (Kept short to stay within the deadline.)
    if best[0].shape[1] == R_sb and time.time() < deadline - 20:
        # Attempt a small randomised low-rank fit is unreliable for exactness;
        # instead we keep the verified schoolbook. No risky changes.
        pass

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    U, V, W = best
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
