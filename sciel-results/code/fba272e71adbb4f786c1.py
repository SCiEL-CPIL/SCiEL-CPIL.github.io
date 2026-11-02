"""Linear-algebraic rank reduction for 4x4*4x5 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start = time.time()
    deadline = 130  # safety margin before 150s kill

    def elapsed():
        return time.time() - start

    # ---------- helpers ----------
    def verify(U, V, W):
        """U: (nm x rank), V: (mp x rank), W: (pn x rank) as lists of lists."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def schoolbook():
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

    best = schoolbook()
    best_rank = n * m * p

    # ---------- Strassen 2x2*2x2 ----------
    def strassen_2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # M1=(A11+A22)(B11+B22)
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # M2=(A21+A22)B11
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1
        # M3=A11(B12-B22)
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        # M4=A22(B21-B11)
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        # M5=(A11+A12)B22
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        # M6=(A21-A11)(B11+B12)
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # M7=(A12-A22)(B21+B22)
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return np.array(U), np.array(V), np.array(W)

    # ---------- Build over-determined structured decomposition ----------
    def build_structured():
        """
        Build a decomposition using Strassen 2x2*2x2 for each 2x2*2x2 sub-block
        of the 4x4*4x4 part, plus schoolbook for the 4x4*4x1 part.
        This gives 8*7 + 16 = 72 terms (vs 80 schoolbook).
        We return U, V, W as numpy arrays of shape (nm, rank), (mp, rank), (pn, rank).
        """
        sU, sV, sW = strassen_2x2()
        terms_U = []
        terms_V = []
        terms_W = []

        # Block index maps for 4x4 matrices (row-major: index = i*4+j for A, j*5+k for B, k*4+i for C)
        # A blocks (2x2 each):
        a11 = [0, 1, 4, 5]   # rows 0-1, cols 0-1
        a12 = [2, 3, 6, 7]   # rows 0-1, cols 2-3
        a21 = [8, 9, 12, 13] # rows 2-3, cols 0-1
        a22 = [10, 11, 14, 15] # rows 2-3, cols 2-3

        # B blocks for 4x4 part (cols 0-3):
        b11 = [0, 1, 5, 6]   # rows 0-1, cols 0-1
        b12 = [2, 3, 7, 8]   # rows 0-1, cols 2-3
        b21 = [10, 11, 15, 16] # rows 2-3, cols 0-1
        b22 = [12, 13, 17, 18] # rows 2-3, cols 2-3

        # C blocks (4x4 part, cols 0-3):
        c11 = [0, 4, 1, 5]   # rows 0-1, cols 0-1
        c12 = [2, 6, 3, 7]   # rows 0-1, cols 2-3
        c21 = [8, 12, 9, 13] # rows 2-3, cols 0-1
        c22 = [10, 14, 11, 15] # rows 2-3, cols 2-3

        def add_strassen(a_map, b_map, c_map):
            for r in range(7):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                for si in range(4):
                    if sU[si, r] != 0:
                        u[a_map[si]] = sU[si, r]
                for sj in range(4):
                    if sV[sj, r] != 0:
                        v[b_map[sj]] = sV[sj, r]
                for sk in range(4):
                    if sW[sk, r] != 0:
                        w[c_map[sk]] = sW[sk, r]
                terms_U.append(u)
                terms_V.append(v)
                terms_W.append(w)

        # 8 Strassen 2x2*2x2 blocks for 4x4*4x4
        add_strassen(a11, b11, c11)
        add_strassen(a12, b21, c11)
        add_strassen(a11, b12, c12)
        add_strassen(a12, b22, c12)
        add_strassen(a21, b11, c21)
        add_strassen(a22, b21, c21)
        add_strassen(a21, b12, c22)
        add_strassen(a22, b22, c22)

        # Schoolbook for 4x4*4x1 (last column of B, col index 4)
        for i in range(4):
            for j in range(4):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[i * 4 + j] = 1
                v[j * 5 + 4] = 1
                w[4 * 4 + i] = 1  # (p-1)*n + i = 4*4 + i = 16+i
                terms_U.append(u)
                terms_V.append(v)
                terms_W.append(w)

        U = np.column_stack(terms_U)
        V = np.column_stack(terms_V)
        W = np.column_stack(terms_W)
        return U, V, W

    # ---------- Linear algebraic rank reduction ----------
    def reduce_rank(U, V, W, max_iters=50):
        """
        Given U (nm x R), V (mp x R), W (pn x R) that form a valid decomposition,
        try to find linear dependencies among columns to reduce rank.
        
        Key insight: if there exists a non-zero vector c in R^R such that
        sum_r c_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0 (as a tensor),
        then we can eliminate one term.
        
        We use a randomized approach: pick random subsets of terms, check if
        the remaining terms still span the multiplication tensor.
        
        Alternative: use the fact that if two terms have the same "pattern"
        (same support in U, V, W), they can be merged.
        
        More powerful: use linear algebra on the coefficient matrix.
        For each output entry (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1.
        This is a system of n*m*p equations in the "flattened" coefficients.
        
        We'll use a greedy elimination: for each term r, check if removing it
        still allows the system to be satisfied (i.e., the tensor is still in the
        span of the remaining terms).
        """
        R = U.shape[1]
        if R <= 1:
            return U, V, W

        # Precompute the "tensor" for each term as a 3D array, then flatten
        # term_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want to check if term_r is in the span of {term_s : s != r}
        
        # Build the matrix M where M[:, r] is the flattened term_r
        # Shape: (n*m*p, R)
        M = np.zeros((n * m * p, R))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    M[idx, :] = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
        
        # The target is the all-ones vector (since we need sum = 1 for each (i,j,k))
        # Actually, M @ 1_R should equal ones(n*m*p) for a valid decomposition.
        
        # To check if column r is redundant: check if M[:, r] is in the span of M[:, :r] and M[:, r+1:]
        # Use QR or rank-based test.
        
        new_U_cols = []
        new_V_cols = []
        new_W_cols = []
        kept = [True] * R
        
        for r in range(R):
            if elapsed() > deadline:
                break
            if not kept[r]:
                continue
            # Check if column r is in span of other kept columns
            other_indices = [s for s in range(R) if s != r and kept[s]]
            if len(other_indices) == 0:
                continue
            M_other = M[:, other_indices]
            M_r = M[:, r:r+1]
            
            # Check rank: rank(M_other) == rank(M_other | M_r)
            # Use SVD for numerical stability
            try:
                rank_other = np.linalg.matrix_rank(M_other, tol=1e-9)
                rank_combined = np.linalg.matrix_rank(np.hstack([M_other, M_r]), tol=1e-9)
                if rank_combined == rank_other:
                    # Column r is redundant, remove it
                    kept[r] = False
            except Exception:
                pass
        
        if not any(kept):
            return U, V, W
        
        new_U = U[:, kept]
        new_V = V[:, kept]
        new_W = W[:, kept]
        
        # Verify
        if verify(new_U.tolist(), new_V.tolist(), new_W.tolist()):
            return new_U, new_V, new_W
        else:
            return U, V, W

    # ---------- Iterative reduction with randomization ----------
    def iterative_reduce(U, V, W, max_rounds=10):
        """Apply reduce_rank multiple times, with random column perturbations to break symmetry."""
        for round_num in range(max_rounds):
            if elapsed() > deadline:
                break
            U, V, W = reduce_rank(U, V, W)
            if U.shape[1] <= 1:
                break
            # Check if further reduction is possible by trying random linear combinations
            # This is a heuristic: if we can find a linear combination of columns that
            # is also a valid "term", we might be able to swap it in.
            # For now, just do one more pass.
            U2, V2, W2 = reduce_rank(U, V, W)
            if U2.shape[1] < U.shape[1]:
                U, V, W = U2, V2, W2
            else:
                # No more progress
                break
        return U, V, W

    # ---------- Main: build and reduce ----------
    if elapsed() < deadline:
        U_struct, V_struct, W_struct = build_structured()
        struct_rank = U_struct.shape[1]
        
        # Verify structured decomposition
        if verify(U_struct.tolist(), V_struct.tolist(), W_struct.tolist()):
            if struct_rank < best_rank:
                best_rank = struct_rank
                best = {"U": U_struct.tolist(), "V": V_struct.tolist(), "W": W_struct.tolist()}
        
        # Apply linear algebraic reduction
        if elapsed() < deadline:
            U_red, V_red, W_red = iterative_reduce(U_struct, V_struct, W_struct, max_rounds=5)
            red_rank = U_red.shape[1]
            if red_rank < best_rank:
                if verify(U_red.tolist(), V_red.tolist(), W_red.tolist()):
                    best_rank = red_rank
                    best = {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}

    # ---------- Try a different structured decomposition: 3x1 split ----------
    if elapsed() < deadline:
        def build_3x1_split():
            """
            Split B into 4x3 and 4x2.
            4x4*4x3: use Strassen 2x2*2x2 for the 2x2*2x2 parts and handle the 3rd col.
            This is complex; instead use a 2x2 block split on A and B.
            
            Actually, let's try: split A into 2x4 row blocks, B into 4x5.
            C_top = A_top (2x4) * B (4x5) = 2x5 output
            C_bot = A_bot (2x4) * B (4x5) = 2x5 output
            
            For C_top: split B into 4x2 and 4x3.
            C_top_left = A_top * B_left (2x4 * 4x2): schoolbook = 16
            C_top_right = A_top * B_right (2x4 * 4x3): schoolbook = 24
            Total for top: 40
            Same for bottom: 40
            Total: 80 (no improvement)
            
            Let's try Strassen on the 2x2*2x2 sub-blocks within.
            """
            sU, sV, sW = strassen_2x2()
            terms_U = []
            terms_V = []
            terms_W = []

            # A row blocks: top = rows 0-1, bottom = rows 2-3
            # Within each, split into 2x2:
            # A_top: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3)
            # A_bot: A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)

            # B col blocks: left = cols 0-1, right = cols 2-4
            # Within left: B11 (rows 0-1, cols 0-1), B21 (rows 2-3, cols 0-1)
            # Within right: B12 (rows 0-1, cols 2-4), B22 (rows 2-3, cols 2-4)

            # C blocks:
            # C_top_left: rows 0-1, cols 0-1
            # C_top_right: rows 0-1, cols 2-4
            # C_bot_left: rows 2-3, cols 0-1
            # C_bot_right: rows 2-3, cols 2-4

            a11 = [0, 1, 4, 5]
            a12 = [2, 3, 6, 7]
            a21 = [8, 9, 12, 13]
            a22 = [10, 11, 14, 15]

            b11 = [0, 1, 5, 6]
            b21 = [10, 11, 15, 16]
            # B12: rows 0-1, cols 2-4 -> 2x3 block
            # For Strassen 2x2*2x2, we need 2x2. So split cols 2-4 into col 2 and cols 3-4?
            # This gets messy. Let's just use schoolbook for the 2x3 parts.

            # For the 2x2*2x2 parts (cols 0-1):
            c11 = [0, 4, 1, 5]   # C_top_left
            c21 = [8, 12, 9, 13] # C_bot_left

            def add_strassen(a_map, b_map, c_map):
                for r in range(7):
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    for si in range(4):
                        if sU[si, r] != 0:
                            u[a_map[si]] = sU[si, r]
                    for sj in range(4):
                        if sV[sj, r] != 0:
                            v[b_map[sj]] = sV[sj, r]
                    for sk in range(4):
                        if sW[sk, r] != 0:
                            w[c_map[sk]] = sW[sk, r]
                    terms_U.append(u)
                    terms_V.append(v)
                    terms_W.append(w)

            # 4 Strassen blocks for the 2x2*2x2 parts (cols 0-1)
            add_strassen(a11, b11, c11)
            add_strassen(a12, b21, c11)
            add_strassen(a21, b11, c21)
            add_strassen(a22, b21, c21)

            # Schoolbook for the 2x3 parts (cols 2-4)
            # C_top_right: rows 0-1, cols 2-4
            for i in range(2):
                for j in range(4):
                    for k in range(3):
                        u = np.zeros(n * m)
                        v = np.zeros(m * p)
                        w = np.zeros(p * n)
                        u[i * 4 + j] = 1
                        v[j * 5 + (2 + k)] = 1
                        w[(2 + k) * 4 + i] = 1
                        terms_U.append(u)
                        terms_V.append(v)
                        terms_W.append(w)

            # C_bot_right: rows 2-3, cols 2-4
            for i in range(2):
                for j in range(4):
                    for k in range(3):
                        u = np.zeros(n * m)
                        v = np.zeros(m * p)
                        w = np.zeros(p * n)
                        u[(2 + i) * 4 + j] = 1
                        v[j * 5 + (2 + k)] = 1
                        w[(2 + k) * 4 + (2 + i)] = 1
                        terms_U.append(u)
                        terms_V.append(v)
                        terms_W.append(w)

            U = np.column_stack(terms_U)
            V = np.column_stack(terms_V)
            W = np.column_stack(terms_W)
            return U, V, W

        U_31, V_31, W_31 = build_3x1_split()
        rank_31 = U_31.shape[1]
        if verify(U_31.tolist(), V_31.tolist(), W_31.tolist()):
            if rank_31 < best_rank:
                best_rank = rank_31
                best = {"U": U_31.tolist(), "V": V_31.tolist(), "W": W_31.tolist()}

            if elapsed() < deadline:
                U_31r, V_31r, W_31r = iterative_reduce(U_31, V_31, W_31, max_rounds=5)
                if U_31r.shape[1] < best_rank:
                    if verify(U_31r.tolist(), V_31r.tolist(), W_31r.tolist()):
                        best_rank = U_31r.shape[1]
                        best = {"U": U_31r.tolist(), "V": V_31r.tolist(), "W": W_31r.tolist()}

    # ---------- Final: try to reduce the best found ----------
    if elapsed() < deadline:
        U_best = np.array(best["U"])
        V_best = np.array(best["V"])
        W_best = np.array(best["W"])
        U_final, V_final, W_final = iterative_reduce(U_best, V_best, W_best, max_rounds=3)
        if U_final.shape[1] < best_rank:
            if verify(U_final.tolist(), V_final.tolist(), W_final.tolist()):
                best_rank = U_final.shape[1]
                best = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    return best
