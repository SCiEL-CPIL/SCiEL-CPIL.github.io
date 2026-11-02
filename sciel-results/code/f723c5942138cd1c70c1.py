import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct a low-rank exact tensor decomposition for 4x4 * 4x7 matrix multiplication.

    Strategy: split the 7-dimensional output (columns of B) into a 4-dim subspace
    and a 3-dim subspace.  Apply recursive Strassen (rank 49) to the 4x4x4 block
    and schoolbook (rank 48) to the 4x4x3 block.  Total rank = 97.
    """
    n, m, p = 4, 4, 7
    deadline = time.time() + 130  # safety margin before 150s kill

    def verify_full(U, V, W, n, m, p):
        """Verify sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2x2():
        """Rank-7 decomposition of the 2x2*2x2 multiplication tensor.

        Returns (U, V, W) each of shape (4, 7).  Index convention:
        U[i*2+j, c] (A element), V[j*2+k, c] (B element), W[k*2+i, c] (C element).
        Verified: sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] == 1 for all i,j,k.
        """
        # Strassen: A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B
        # M1=(a+d)(e+h)  M2=(c+d)e  M3=a(f-h)  M4=d(g-e)
        # M5=(a+b)h      M6=(c-a)(e+f)  M7=(b-d)(g+h)
        # C00=M1-M4-M5+M7  C01=M3+M5  C10=M2+M4  C11=M1-M2+M3-M6
        A = [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
             [1, 1, 0, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
        B = [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1],
             [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
        C = [[1, 0, 0, -1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0],
             [0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, -1, 0]]
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j, c] = A[c][i * 2 + j]
                    V[j * 2 + i, c] = B[c][j * 2 + i]
                    W[i * 2 + j, c] = C[c][i * 2 + j]
        # verify
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for c in range(7):
                        s += U[i * 2 + j, c] * V[j * 2 + k, c] * W[k * 2 + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return None
        return U, V, W

    def strassen_4x4x4():
        """Recursive Strassen: rank-49 decomposition of 4x4*4x4.

        Level 1: split into 2x2 blocks, 7 Strassen products of 2x2*2x2.
        Level 2: each 2x2*2x2 uses the rank-7 base.
        Returns (U, V, W) each of shape (16, 49).
        """
        base = strassen_2x2x2()
        if base is None:
            return None
        bU, bV, bW = base

        A_coeff = [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
                   [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]]
        B_coeff = [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1],
                   [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
        C_out = [[1, 0, 0, -1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0],
                 [0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, -1, 0]]

        def block_indices(br, bc):
            r0 = br * 2; c0 = bc * 2
            return [(r0) * 4 + c0, (r0) * 4 + c0 + 1,
                    (r0 + 1) * 4 + c0, (r0 + 1) * 4 + c0 + 1]

        U = np.zeros((16, 49)); V = np.zeros((16, 49)); W = np.zeros((16, 49))
        for r in range(7):
            for s in range(7):
                col = r * 7 + s
                # U: A-part
                for i in range(4):
                    for j in range(4):
                        val = 0.0
                        for ab in range(4):
                            if A_coeff[r][ab] == 0:
                                continue
                            br, bc = divmod(ab, 2)
                            for li, l in enumerate(block_indices(br, bc)):
                                ii, jj = divmod(l, 4)
                                if ii == i and jj == j:
                                    val += A_coeff[r][ab] * bU[li, s]
                        U[i * 4 + j, col] = val
                # V: B-part
                for j in range(4):
                    for k in range(4):
                        val = 0.0
                        for bb in range(4):
                            if B_coeff[r][bb] == 0:
                                continue
                            br, bc = divmod(bb, 2)
                            for li, l in enumerate(block_indices(br, bc)):
                                jj, kk = divmod(l, 4)
                                if jj == j and kk == k:
                                    val += B_coeff[r][bb] * bV[li, s]
                        V[j * 4 + k, col] = val
                # W: C-part (output block (br, bc) from A-block br, B-block bc)
                for i in range(4):
                    for k in range(4):
                        val = 0.0
                        for ab in range(4):
                            if A_coeff[r][ab] == 0:
                                continue
                            br, bc = divmod(ab, 2)
                            if br != i // 2:
                                continue
                            for bb in range(4):
                                if B_coeff[r][bb] == 0:
                                    continue
                                _, bcc = divmod(bb, 2)
                                if bcc != k // 2:
                                    continue
                                for li, l in enumerate(block_indices(br, bcc)):
                                    ii, kk = divmod(l, 4)
                                    if ii == i and kk == k:
                                        val += C_out[li][s]
                        W[k * 4 + i, col] = val
        # verify
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0.0
                    for c in range(49):
                        s += U[i * 4 + j, c] * V[j * 4 + k, c] * W[k * 4 + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return None
        return U, V, W

    def schoolbook_4x4x3():
        """Rank-48 decomposition of 4x4 * 4x3.  U (16,48), V (12,48), W (12,48)."""
        rank = 4 * 4 * 3
        U = np.zeros((16, rank)); V = np.zeros((12, rank)); W = np.zeros((12, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i * 4 + j, col] = 1
                    V[j * 3 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1
        return U, V, W

    # ---- Assemble the 4x4 * 4x7 decomposition (rank 97) ----
    s44 = strassen_4x4x4()
    if s44 is not None and time.time() < deadline:
        U44, V44, W44 = s44
        U43, V43, W43 = schoolbook_4x4x3()
        rank44, rank43 = 49, 48
        total_rank = rank44 + rank43
        U = np.zeros((16, total_rank))
        V = np.zeros((28, total_rank))
        W = np.zeros((28, total_rank))
        # 4x4x4 part (columns 0..48): B cols 0..3, C cols 0..3
        U[:, :rank44] = U44
        V[:16, :rank44] = V44
        W[:16, :rank44] = W44
        # 4x4x3 part (columns 49..96): B cols 4..6, C cols 4..6
        U[:, rank44:] = U43
        V[16:, rank44:] = V43
        W[16:, rank44:] = W43
        if verify_full(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Fallback: schoolbook (rank 112) ----
    U, V, W = schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
