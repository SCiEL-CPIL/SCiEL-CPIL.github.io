import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = 2, 4, 8

    # Build the standard Strassen 2x2x2 decomposition (rank 7)
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)

    # r=0 (M1) = (A00+A11)(B00+B11)
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1; W_local[3, 0] = 1

    # r=1 (M2) = (A10+A11)B00
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1; W_local[3, 1] = -1

    # r=2 (M3) = A00(B10-B11)
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1; W_local[3, 2] = 1

    # r=3 (M4) = A11(B01-B00)
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1; W_local[2, 3] = 1

    # r=4 (M5) = (A00+A01)B11
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1; W_local[1, 4] = 1

    # r=5 (M6) = (A10-A00)(B00+B01)
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1

    # r=6 (M7) = (A01-A11)(B10+B11)
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1

    # Build the 56-rank decomposition by tiling the Strassen block
    total_rank = 56
    U = np.zeros((8, total_rank), dtype=int)
    V = np.zeros((32, total_rank), dtype=int)
    W = np.zeros((16, total_rank), dtype=int)

    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7

            # a_map: maps local 2x2 index (la) to global U index
            if s == 0:
                a_map = [0, 1, 4, 5]   # A[0,0], A[0,1], A[1,0], A[1,1]
            else:
                a_map = [2, 3, 6, 7]   # A[0,2], A[0,3], A[1,2], A[1,3]

            # b_map: maps local 2x2 index (lb) to global V index
            if s == 0:
                b_map = [0*8 + 2*j, 0*8 + 2*j+1, 1*8 + 2*j, 1*8 + 2*j+1]
            else:
                b_map = [2*8 + 2*j, 2*8 + 2*j+1, 3*8 + 2*j, 3*8 + 2*j+1]

            # c_map: maps local 2x2 index (lc) to global W index
            # W[k*n+i, r] => W[k*2+i, r]
            c_map = [2*j*2+0, (2*j+1)*2+0, 2*j*2+1, (2*j+1)*2+1]

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val

    def verify(U, V, W):
        R = U.shape[1]
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    s = 0
                    for r in range(R):
                        s += U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
                    if s != 1:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to naive 64
        U = np.zeros((8, 64), dtype=int)
        V = np.zeros((32, 64), dtype=int)
        W = np.zeros((16, 64), dtype=int)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j, r] = 1
                    V[j*8+k, r] = 1
                    W[k*2+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now attempt to find a lower-rank decomposition by exploiting linear
    # dependencies among the 56 rank-1 tensor vectors.
    #
    # The 56 terms t_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r] span a subspace of
    # R^{8*32*16}. If this subspace has dimension < 56, we can find a
    # linear combination of the 56 terms that yields a valid decomposition
    # with fewer non-zero coefficients (after Gaussian elimination).
    #
    # However, a more practical approach: try to find a rank-48 decomposition
    # by searching for 8 linear relations among the tensor vectors and using
    # them to eliminate 8 terms.
    #
    # Strategy: Build the 56x4096 tensor matrix, compute its row space basis,
    # and check if we can express each row as a combination of fewer rows.
    #
    # Actually, a cleaner approach: the 56 terms correspond to 8 independent
    # Strassen blocks (one for each (j, s) pair). If we can find linear
    # combinations of blocks that cancel out, we can reduce the rank.
    #
    # Let's try a different tactic: use the fact that the problem has
    # symmetries. The 8 Strassen blocks are indexed by (j, s) where j∈{0,1,2,3}
    # and s∈{0,1}. The blocks with the same j but different s act on different
    # rows of B. The blocks with the same s but different j act on different
    # column pairs of B.
    #
    # Key insight: For fixed s, the 4 blocks (j=0,1,2,3) all use the same
    # A sub-block (either columns 0-1 or 2-3) but different B row pairs.
    # We might be able to combine them.
    #
    # Let's try to construct a rank-48 decomposition by using a 4x4
    # matrix multiplication structure on the B side.
    #
    # Alternative approach: Use the "laser method" or direct search for
    # a rank-48 decomposition.
    #
    # For now, let's try a simpler idea: check if the 56 tensor vectors
    # are linearly dependent. If so, we can reduce the rank.

    if time.time() < deadline - 5:
        # Build the tensor matrix T (56 x 4096)
        tensor_size = 8 * 32 * 16  # 4096
        T = np.zeros((total_rank, tensor_size), dtype=float)

        for r in range(total_rank):
            for i in range(2):
                for j in range(4):
                    for k in range(8):
                        idx = i * 32 * 16 + j * 16 + k
                        T[r, idx] = U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]

        # Compute the rank of T
        try:
            rank_T = np.linalg.matrix_rank(T, tol=1e-10)
        except:
            rank_T = total_rank

        if rank_T < total_rank:
            # There are dependencies. Try to find a lower-rank decomposition.
            # Use SVD to get a basis for the row space.
            try:
                U_svd, svals, Vt_svd = np.linalg.svd(T, full_matrices=False)
                # The number of non-zero singular values gives the rank.
                # We can use the top rank_T left singular vectors as a basis.
                # But this doesn't directly give us a lower-rank decomposition.
                # We need to find coefficients such that the linear combination
                # of the 56 terms still equals the tensor.
                #
                # This is complex. For now, just note the dependency and
                # return the 56-rank solution.
                pass
            except:
                pass

    # Return the verified 56-rank solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
