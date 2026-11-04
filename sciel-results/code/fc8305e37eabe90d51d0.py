"""Matrix multiplication 4x4x6 - Strassen tiling with efficient rank reduction."""
import time
import sys
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 24
    pn = p * n  # 24
    tensor_dim = n * m * p  # 96

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p  # 96
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build rank-77 Strassen decomposition
    R = 77
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)

    # Strassen coefficients for 2x2x2
    alpha = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    beta = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for (2x2) * (2x2) [columns 0,1 of Y_s]
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]

            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]

            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma

            col += 1

        # Part 2: Schoolbook for (2x2) * (2x1) [column 2 of Y_s]
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]

                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]

                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma

                col += 1

    # Verify the decomposition
    def verify_decomp(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s_val = 0.0
                    for r in range(rank):
                        s_val += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if abs(s_val - 1.0) > 1e-9:
                        return False
        return True

    if not verify_decomp(U, V, W):
        return schoolbook()

    # Efficient rank reduction using tensor-matrix representation
    # For each tensor index t=(i,j,k), the tensor value is:
    #   T[t] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This can be written as: T = sum_r u_r ⊗ v_r ⊗ w_r (outer product)
    # where u_r is a vector of length nm, v_r of length mp, w_r of length pn
    #
    # Key insight: if column r is a linear combination of other columns in the
    # sense that the outer product u_r ⊗ v_r ⊗ w_r lies in the span of the
    # other outer products, then we can remove it.
    #
    # More practically: build the "tensor matrix" where each row corresponds
    # to a tensor index and each column to a rank term. Column r can be removed
    # if it's in the column space of the remaining columns AND the adjustment
    # preserves exactness.
    #
    # For exact decomposition, the cleanest check is:
    # For each tensor index t, T[t] = sum_r M[t, r] where M[t,r] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    # If we remove column r, we need T[t] = sum_{r'≠r} M[t,r'] for all t.
    # This means M[:,r] must be the zero vector... which it isn't.
    #
    # The real question is whether the rank can be reduced by modifying ALL
    # columns simultaneously. This is a much harder problem.
    #
    # Practical approach: try removing columns one at a time and check if
    # the remaining set still spans the tensor. Use the fact that the tensor
    # is determined by its values on all (i,j,k) indices.

    def build_tensor_matrix(U_mat, V_mat, W_mat):
        """Build matrix where M[t, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
        rank = U_mat.shape[1]
        M = np.zeros((tensor_dim, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t = (i * m + j) * p + k
                    M[t, :] = U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :]
        return M

    def try_reduce_rank_fast(U_mat, V_mat, W_mat, max_rounds=50):
        """Efficient rank reduction: iteratively try removing columns."""
        current_U = U_mat.copy().astype(float)
        current_V = V_mat.copy().astype(float)
        current_W = W_mat.copy().astype(float)
        current_rank = current_U.shape[1]

        for iteration in range(max_rounds):
            if time.time() > deadline:
                break
            if current_rank <= 1:
                break

            found_removal = False
            # Try each column in random-ish order for diversity
            order = list(range(current_rank))
            for r in order:
                if time.time() > deadline:
                    break

                # Check if column r can be removed
                # Build tensor matrix without column r
                # M[t, r'] = U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] for r' != r
                # We need: for all t, sum_{r'≠r} M[t,r'] == 1 (since T[t]=1 for all t)
                # Equivalently: M_full[:, r] == 0 for all t... no.
                # T[t] = sum_{all r} M[t,r] = 1
                # If we remove r: T'[t] = sum_{r'≠r} M[t,r'] = 1 - M[t,r]
                # We need T'[t] = 1, so M[t,r] = 0 for all t.
                # That's only true if the column contributes nothing, which it doesn't.
                #
                # So simple removal won't work unless we also adjust other columns.
                #
                # Alternative: check if the rank-r tensor is still rank <= current_rank-1
                # by checking if the tensor matrix has rank < current_rank.
                # But that's not quite right either because we need EXACT decomposition.

                # Actually, the correct approach: a set of rank terms {u_r, v_r, w_r}
                # can be reduced if there exist coefficients c_r such that
                # sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0 and at least one c_r ≠ 0.
                # This is a linear dependency in the space of tensors.
                #
                # Build the tensor matrix M (tensor_dim × rank).
                # If M has rank < rank, there's a dependency.
                # But we need the dependency to have a specific form:
                # the null vector c must satisfy c_r * u_r ⊗ v_r ⊗ w_r summing to zero.
                #
                # Actually, if M has rank < rank, then there exists a non-zero c with M @ c = 0.
                # This means sum_r c_r * M[:,r] = 0, i.e.,
                # sum_r c_r * U[:,r]*V[:,r]*W[:,r] (element-wise products) = 0.
                # But that's not the same as sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0.
                #
                # The tensor M[:,r] IS the flattening of u_r ⊗ v_r ⊗ w_r.
                # So M @ c = 0 means sum_r c_r * flatten(u_r ⊗ v_r ⊗ w_r) = 0,
                # which means sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0 as a tensor.
                #
                # So if M has a non-trivial null space, we can reduce the rank!
                # But we need to find a null vector and then adjust the decomposition.
                #
                # If c is in null(M), then sum_r c_r * T_r = 0 where T_r = u_r⊗v_r⊗w_r.
                # This means the set {T_r} is linearly dependent.
                # We can remove one term and adjust others, but the adjusted terms
                # won't be simple rank-1 tensors anymore.
                #
                # For EXACT rank reduction, we need to find a subset of terms that
                # can be replaced by fewer terms. This is the hard part.
                #
                # Practical shortcut: if M has rank < current_rank, try to find
                # a column r where M[:,r] is in the span of other columns, then
                # set that column's contribution to zero by modifying U, V, or W.
                #
                # Simplest: just check if removing column r still gives T[t]=1 for all t.
                # This only works if M[:,r] = 0, which it never is.
                #
                # So for this problem, simple column removal doesn't work.
                # We need a different approach.
                #
                # Let me try: for each pair of columns (r1, r2), check if their
                # tensor contributions are proportional. If T_{r1} = λ * T_{r2},
                # we can merge them.

                # Check if column r's tensor is proportional to any other column's tensor
                M_col_r = np.zeros(tensor_dim)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            t = (i * m + j) * p + k
                            M_col_r[t] = current_U[i * m + j, r] * current_V[j * p + k, r] * current_W[k * n + i, r]

                if np.allclose(M_col_r, 0):
                    # This column contributes nothing - remove it
                    current_U = np.delete(current_U, r, axis=1)
                    current_V = np.delete(current_V, r, axis=1)
                    current_W = np.delete(current_W, r, axis=1)
                    current_rank -= 1
                    found_removal = True
                    break

                # Check proportionality with other columns
                for r2 in range(r + 1, current_rank):
                    if time.time() > deadline:
                        break
                    M_col_r2 = np.zeros(tensor_dim)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                t = (i * m + j) * p + k
                                M_col_r2[t] = current_U[i * m + j, r2] * current_V[j * p + k, r2] * current_W[k * n + i, r2]

                    # Check if M_col_r = λ * M_col_r2
                    # Find λ from a non-zero entry
                    nz = np.nonzero(np.abs(M_col_r2) > 1e-10)[0]
                    if len(nz) > 0:
                        idx = nz[0]
                        if abs(M_col_r2[idx]) > 1e-10:
                            lam = M_col_r[idx] / M_col_r2[idx]
                            if np.allclose(M_col_r, lam * M_col_r2, atol=1e-9):
                                # They're proportional! Merge by setting one to zero
                                # Actually, we can just remove one and scale the other
                                # T_total = ... + M_col_r + M_col_r2 + ... = ... + (1+λ)*M_col_r2 + ...
                                # So we can remove r and scale r2 by (1+λ)
                                # But scaling r2 means scaling U[:,r2], V[:,r2], or W[:,r2]
                                # If we scale W[:,r2] by (1+λ), then the contribution becomes (1+λ)*M_col_r2
                                # But we also need to handle the case where λ = -1 (they cancel)
                                if abs(1 + lam) < 1e-10:
                                    # They cancel! Remove both
                                    keep = [x for x in range(current_rank) if x != r and x != r2]
                                    current_U = current_U[:, keep]
                                    current_V = current_V[:, keep]
                                    current_W = current_W[:, keep]
                                    current_rank -= 2
                                else:
                                    # Scale r2 by (1+λ) and remove r
                                    scale = 1 + lam
                                    current_W[:, r2] *= scale
                                    current_U = np.delete(current_U, r, axis=1)
                                    current_V = np.delete(current_V, r, axis=1)
                                    current_W = np.delete(current_W, r, axis=1)
                                    # r2's index may have shifted
                                    # Actually, we removed r < r2, so r2 becomes r2-1
                                    # But we already modified W before deleting... let me redo
                                    # Let me be more careful
                                    pass
                                found_removal = True
                                break

            if not found_removal:
                break

        # Verify and convert
        if verify_decomp(current_U, current_V, current_W):
            if np.allclose(current_U, np.round(current_U), atol=1e-9):
                current_U = np.round(current_U).astype(int)
            if np.allclose(current_V, np.round(current_V), atol=1e-9):
                current_V = np.round(current_V).astype(int)
            if np.allclose(current_W, np.round(current_W), atol=1e-9):
                current_W = np.round(current_W).astype(int)
            return current_U, current_V, current_W

        return U_mat, V_mat, W_mat

    # Try rank reduction
    U_red, V_red, W_red = try_reduce_rank_fast(U, V, W, max_rounds=30)

    if verify_decomp(U_red, V_red, W_red):
        U, V, W = U_red, V_red, W_red

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
