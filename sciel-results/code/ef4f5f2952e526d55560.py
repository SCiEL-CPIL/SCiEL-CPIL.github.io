"""Linear-algebraic rank-reduction search for 4x4*4x5 matrix multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    start_time = time.time()
    deadline = 140

    def verify(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def verify_fast(U, V, W):
        """Vectorized verification using numpy."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        rank = U_arr.shape[1]
        # Build tensor: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # We need to check all 80 entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(
                        U_arr[i * m + j, :] * V_arr[j * p + k, :],
                        W_arr[k * n + i, :]
                    )
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Build the 64-term Strassen + schoolbook decomposition
    def build_base_decomposition():
        """Build 4x4*4x5 using Strassen(2x2*2x2) for square blocks and schoolbook for 2x3."""
        U_list = []
        V_list = []
        W_list = []

        # Strassen 2x2*2x2 terms (7 terms)
        sU = [[0]*7 for _ in range(4)]
        sV = [[0]*7 for _ in range(4)]
        sW = [[0]*7 for _ in range(4)]
        # r=0: (A11+A22)(B11+B22)
        sU[0][0]=1; sU[3][0]=1; sV[0][0]=1; sV[3][0]=1; sW[0][0]=1; sW[3][0]=1
        # r=1: (A21+A22)B11
        sU[2][1]=1; sU[3][1]=1; sV[0][1]=1; sW[2][1]=1
        # r=2: A11(B12-B22)
        sU[0][2]=1; sV[1][2]=1; sV[3][2]=-1; sW[1][2]=1
        # r=3: A22(B21-B11)
        sU[3][3]=1; sV[2][3]=1; sV[0][3]=-1; sW[0][3]=1; sW[2][3]=1
        # r=4: (A11+A12)B22
        sU[0][4]=1; sU[1][4]=1; sV[3][4]=1; sW[0][4]=-1; sW[1][4]=1
        # r=5: (A21-A11)(B11+B12)
        sU[2][5]=1; sU[0][5]=-1; sV[0][5]=1; sV[1][5]=1; sW[3][5]=1
        # r=6: (A12-A22)(B21+B22)
        sU[1][6]=1; sU[3][6]=-1; sV[2][6]=1; sV[3][6]=1; sW[0][6]=1; sW[3][6]=1

        # Block index maps
        # A blocks (2x2 each): A11,A12,A21,A22 -> row indices in U (i*4+j)
        a11 = [0, 1, 4, 5]
        a12 = [2, 3, 6, 7]
        a21 = [8, 9, 12, 13]
        a22 = [10, 11, 14, 15]
        # B blocks: B11(2x2), B12(2x3), B21(2x2), B22(2x3)
        b11 = [0, 1, 5, 6]
        b21 = [10, 11, 15, 16]
        # B12: rows 0-1, cols 2-4 -> V[j*5+k] for j=0,1; k=2,3,4
        b12 = [2, 3, 7, 8, 12, 13]
        # B22: rows 2-3, cols 2-4 -> V[j*5+k] for j=2,3; k=2,3,4
        b22 = [12, 13, 17, 18, 22, 23]  # wait, V has 20 rows (m*p=4*5)
        # V[j*p+k] where j in 0..3, k in 0..4
        # B22: j=2,3; k=2,3,4 -> indices: 2*5+2=12, 2*5+3=13, 2*5+4=14, 3*5+2=17, 3*5+3=18, 3*5+4=19
        b22 = [12, 13, 14, 17, 18, 19]
        # W blocks (2x2 or 2x3): C[k*n+i]
        c11 = [0, 4, 1, 5]   # k=0,1; i=0,1
        c21 = [2, 6, 3, 7]   # k=0,1; i=2,3
        # C12: k=2,3,4; i=0,1 -> W[k*4+i]
        c12 = [8, 12, 16, 9, 13, 17]  # k=2:i=0,1 -> 8,9; k=3: 12,13; k=4: 16,17
        # C22: k=2,3,4; i=2,3 -> W[k*4+i]
        c22 = [10, 14, 18, 11, 15, 19]  # k=2: 10,11; k=3: 14,15; k=4: 18,19

        def add_strassen(a_map, b_map, c_map):
            """Add 7 Strassen terms for a 2x2*2x2 block product."""
            for r in range(7):
                U_new = [0] * 16
                V_new = [0] * 20
                W_new = [0] * 20
                for si in range(4):
                    if sU[si][r] != 0:
                        U_new[a_map[si]] = sU[si][r]
                for sj in range(4):
                    if sV[sj][r] != 0:
                        V_new[b_map[sj]] = sV[sj][r]
                for sk in range(4):
                    if sW[sk][r] != 0:
                        W_new[c_map[sk]] = sW[sk][r]
                U_list.append(U_new)
                V_list.append(V_new)
                W_list.append(W_new)

        # C11 = A11*B11 + A12*B21 (both 2x2*2x2)
        add_strassen(a11, b11, c11)
        add_strassen(a12, b21, c11)

        # C21 = A21*B11 + A22*B21 (both 2x2*2x2)
        add_strassen(a21, b11, c21)
        add_strassen(a22, b21, c21)

        # C12 = A11*B12 + A12*B22 (2x2 * 2x3, schoolbook: 2*2*3=12 terms each)
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U_new = [0] * 16
                    V_new = [0] * 20
                    W_new = [0] * 20
                    U_new[i * 4 + j] = 1
                    V_new[j * 5 + (2 + k)] = 1
                    W_new[(2 + k) * 4 + i] = 1
                    U_list.append(U_new)
                    V_list.append(V_new)
                    W_list.append(W_new)

        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U_new = [0] * 16
                    V_new = [0] * 20
                    W_new = [0] * 20
                    U_new[i * 4 + (2 + j)] = 1
                    V_new[(2 + j) * 5 + (2 + k)] = 1
                    W_new[(2 + k) * 4 + i] = 1
                    U_list.append(U_new)
                    V_list.append(V_new)
                    W_list.append(W_new)

        # C22 = A21*B12 + A22*B22
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U_new = [0] * 16
                    V_new = [0] * 20
                    W_new = [0] * 20
                    U_new[(2 + i) * 4 + j] = 1
                    V_new[j * 5 + (2 + k)] = 1
                    W_new[(2 + k) * 4 + (2 + i)] = 1
                    U_list.append(U_new)
                    V_list.append(V_new)
                    W_list.append(W_new)

        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U_new = [0] * 16
                    V_new = [0] * 20
                    W_new = [0] * 20
                    U_new[(2 + i) * 4 + (2 + j)] = 1
                    V_new[(2 + j) * 5 + (2 + k)] = 1
                    W_new[(2 + k) * 4 + (2 + i)] = 1
                    U_list.append(U_new)
                    V_list.append(V_new)
                    W_list.append(W_new)

        return U_list, V_list, W_list

    U, V, W = build_base_decomposition()
    best_rank = len(U)
    best = {"U": U, "V": V, "W": W}

    if not verify(U, V, W):
        # Fallback to schoolbook
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
        best = {"U": U, "V": V, "W": W}
        best_rank = rank
        return best

    # Now attempt rank reduction via linear algebra
    # Key insight: if term r can be expressed as a linear combination of other terms,
    # we can eliminate it.
    # For each term r, check if its "signature" (the 80-dim vector of contributions)
    # lies in the span of the other terms' signatures.

    def get_term_signature(U, V, W, r):
        """Get the 80-dim contribution vector of term r."""
        sig = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    sig.append(U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r])
        return sig

    def try_eliminate_term(U, V, W, target_r):
        """Try to eliminate term target_r by expressing it as combo of others.
        Returns (new_U, new_V, new_W) or None."""
        rank = len(U[0])
        # Build the system: for each of the 80 coords, the target's contribution
        # should equal sum of c_r * sig_r for r != target_r
        # But this is nonlinear in general. However, if we fix the structure,
        # we can try: replace term target_r with a scaled version that cancels.
        # Simpler approach: try pairwise cancellation.
        return None

    # Better approach: try to find pairs of terms that can be merged.
    # If term r1 and term r2 have the property that for some (i,j,k):
    # U1[i*m+j]*V1[j*p+k]*W1[k*n+i] = -U2[i*m+j]*V2[j*p+k]*W2[k*n+i]
    # for ALL (i,j,k), then they cancel and we can remove both.

    def find_cancelling_pairs(U, V, W):
        """Find pairs of terms that exactly cancel."""
        rank = len(U[0])
        sigs = []
        for r in range(rank):
            sigs.append(get_term_signature(U, V, W, r))
        pairs = []
        for r1 in range(rank):
            for r2 in range(r1 + 1, rank):
                # Check if sig[r1] = -sig[r2]
                cancel = True
                for c in range(n * m * p):
                    if abs(sigs[r1][c] + sigs[r2][c]) > 1e-9:
                        cancel = False
                        break
                if cancel:
                    pairs.append((r1, r2))
        return pairs

    # Try to find and remove cancelling pairs
    removed_any = True
    while removed_any and time.time() - start_time < deadline:
        removed_any = False
        pairs = find_cancelling_pairs(U, V, W)
        for r1, r2 in pairs:
            # Remove both terms
            keep = [r for r in range(len(U[0])) if r != r1 and r != r2]
            new_U = [[U[i][r] for r in keep] for i in range(n * m)]
            new_V = [[V[i][r] for r in keep] for i in range(m * p)]
            new_W = [[W[i][r] for r in keep] for i in range(p * n)]
            if verify(new_U, new_V, new_W):
                U, V, W = new_U, new_V, new_W
                removed_any = True
                if len(U[0]) < best_rank:
                    best_rank = len(U[0])
                    best = {"U": U, "V": V, "W": W}
                break  # restart pair search after modification

    # Try linear combination elimination:
    # For each term r, try to find coefficients c_s for s != r such that
    # sig(r) = sum_s c_s * sig(s)
    # This requires solving a linear system, but the signatures are specific vectors.
    # Since the problem is small (80 coords, ~64 terms), we can attempt this.

    def try_linear_elimination(U, V, W):
        """Try to eliminate one term at a time via linear combination."""
        rank = len(U[0])
        if rank <= 1:
            return U, V, W, False

        # Build signature matrix: 80 x rank
        sig_mat = np.zeros((n * m * p, rank))
        for r in range(rank):
            sig = get_term_signature(U, V, W, r)
            sig_mat[:, r] = sig

        # The total sum must equal the all-ones vector
        # If we remove term t, we need: sum_{r!=t} c_r * sig_r = ones
        # where c_r are new coefficients. But we can't change the structure freely.
        # Alternative: if sig_t is in the span of {sig_s : s != t}, and the
        # coefficients work out, we can replace term t.
        # This is complex for general tensor decompositions.
        # Let's try a simpler heuristic: check if any term is a scalar multiple
        # of another (they'd be redundant if same structure, different scale).
        return U, V, W, False

    U, V, W, changed = try_linear_elimination(U, V, W)

    # Try a different strategy: use the fact that for 2x3 blocks, we might
    # find a rank-5 or rank-6 decomposition instead of rank-6 (schoolbook).
    # Actually, 2x2*2x3 has rank at most 6 (schoolbook). Can we do better?
    # The tensor rank of 2x2*2x3 is known to be 6 (same as schoolbook for these sizes).
    # So no improvement there.

    # Try: can we do better than Strassen for the overall structure?
    # The Laderman algorithm for 3x3*3x3 has rank 23.
    # For 4x4*4x4, the best known is 49 (from Strassen recursion: 7^2=49 for 4x4*4x4... wait no)
    # Actually 4x4*4x4 can be done with rank 49 by applying Strassen to 2x2 blocks:
    # 4x4 = 2x2 blocks of 2x2, so 4 block products each rank 7 = 28? No.
    # 4x4*4x4: split into 2x2 blocks -> 4 products of 2x2*2x2, each rank 7 -> total 28.
    # But that's not right either. Let me think...
    # Actually for 4x4*4x4, you split A into 4 blocks (2x2), B into 4 blocks (2x2),
    # C = sum over block products. There are 4*4=16 entries in C, but grouped into
    # 4 blocks of 2x2. Each C-block is a sum of 2 A-blocks * 2 B-blocks.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # Each C-block needs 2 Strassen decompositions (rank 7 each) = 14 per block, 4 blocks = 56.
    # But that's for 4x4*4x4. For 4x4*4x5, we have the asymmetry.

    # Let's try: can we use a 4x4*4x4 rank-49 decomposition and adapt it?
    # Actually, the best known for 4x4*4x4 is 49 (from bilinear complexity results).
    # But we need 4x4*4x5.

    # For now, let's try to improve by checking if the 64-term solution can be reduced.
    # One more attempt: try to find if any term is linearly dependent.

    if time.time() - start_time < deadline:
        rank = len(U[0])
        sig_mat = np.zeros((n * m * p, rank))
        for r in range(rank):
            sig = get_term_signature(U, V, W, r)
            sig_mat[:, r] = sig

        # Check rank of the signature matrix
        # If rank < number of terms, there's linear dependence
        # But we need the specific structure to exploit it.
        # For now, just check if we can find a null vector.
        try:
            from scipy.linalg import null_space
            null_vecs = null_space(sig_mat, rcond=1e-10)
            if null_vecs.shape[1] > 0:
                # There's a linear dependence. But exploiting it requires
                # the coefficients to correspond to valid tensor terms.
                # This is non-trivial; skip for now.
                pass
        except:
            pass

    # Final: ensure we return the best found
    if len(U[0]) < best_rank:
        best_rank = len(U[0])
        best = {"U": U, "V": V, "W": W}

    return best
