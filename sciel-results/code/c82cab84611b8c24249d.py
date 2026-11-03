"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    dim_u = n * m   # 6
    dim_v = m * p   # 30
    dim_w = p * n   # 20

    # Build the true tensor
    T_true = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        if U_mat.shape[1] == 0:
            return np.allclose(T_true, 0, atol=tol)
        T_rec = np.einsum('ic,jc,kc->ijk', U_mat, V_mat, W_mat)
        return np.allclose(T_rec, T_true, atol=tol)

    # Build the rank-55 Strassen-based decomposition
    rank = 55
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))

    col = 0
    # Part A: j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1

    # Part B: j=1,2 Strassen on 2x2 blocks
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11, col] += 1; U[u_22, col] += 1
        V[v_11, col] += 1; V[v_22, col] += 1
        W[w_00, col] += 1; W[w_11, col] += 1
        col += 1
        # M2 = (a21 + a22)b11
        U[u_21, col] += 1; U[u_22, col] += 1
        V[v_11, col] += 1
        W[w_10, col] += 1; W[w_11, col] += -1
        col += 1
        # M3 = a11(b12 - b22)
        U[u_11, col] += 1
        V[v_12, col] += 1; V[v_22, col] += -1
        W[w_01, col] += 1; W[w_11, col] += 1
        col += 1
        # M4 = a22(b21 - b11)
        U[u_22, col] += 1
        V[v_21, col] += 1; V[v_11, col] += -1
        W[w_00, col] += 1; W[w_10, col] += 1
        col += 1
        # M5 = (a11 + a12)b22
        U[u_11, col] += 1; U[u_12, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += -1; W[w_01, col] += 1
        col += 1
        # M6 = (a21 - a11)(b11 + b12)
        U[u_21, col] += 1; U[u_11, col] += -1
        V[v_11, col] += 1; V[v_12, col] += 1
        W[w_11, col] += 1
        col += 1
        # M7 = (a12 - a22)(b21 + b22)
        U[u_12, col] += 1; U[u_22, col] += -1
        V[v_21, col] += 1; V[v_22, col] += 1
        W[w_00, col] += 1
        col += 1

    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank

    # Vectorized peeling: check if column c is in the span of others
    def peel_round(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        if r < 2:
            return U_mat, V_mat, W_mat, False

        # Build M where M[:,c] = vec(u_c ⊗ v_c ⊗ w_c)
        M = np.einsum('ic,jc,kc->ijkc', U_mat, V_mat, W_mat)
        M = M.reshape(-1, r)

        # Compute Gram matrix G = M^T M (r x r)
        G = M.T @ M
        # For each c, check if M[:,c] is in span of M[:,others]
        # This is equivalent to: the Schur complement of G[c,c] in G is singular
        # i.e., det(G_others) * G[c,c] - det(G) = 0 approximately
        # More robust: compute rank of G and check if removing row/col c reduces rank
        
        # Use the fact that c is redundant iff G_others has same rank as G
        # and M[:,c] is in colspace of M[:,others]
        # Efficient: for each c, compute the projection residual using the 
        # precomputed inverse of G (if G is invertible)
        
        try:
            # Compute pseudo-inverse of G
            G_inv = np.linalg.pinv(G, rcond=1e-10)
        except Exception:
            return U_mat, V_mat, W_mat, False

        for c in range(r):
            if time.time() > deadline - 20:
                break
            # The coefficient vector for expressing M[:,c] in terms of others
            # If G is invertible, the full solution to G*x = M[:,c] gives
            # x[c] = G_inv[c,c] and x[others] = G_inv[others,c]
            # But we need x[c]=0 for it to be in span of others
            # Equivalent: the c-th component of the least squares solution 
            # with constraint x[c]=0 should match
            # Simpler: residual = M[:,c] - M[:,others] @ (M_others^T M_others)^{-1} M_others^T M[:,c]
            # Using Woodbury: if G is invertible, the residual norm squared = 
            # 1 / G_inv[c,c]
            g_inv_cc = G_inv[c, c]
            if g_inv_cc < 1e-12:
                # g_inv_cc near zero means M[:,c] is in span of others
                # Verify
                others = [i for i in range(r) if i != c]
                M_others = M[:, others]
                b = M[:, c]
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(M_others, b, rcond=None)
                    proj = M_others @ coeffs
                    residual = b - proj
                    if np.linalg.norm(residual) < 1e-8 * max(1.0, np.linalg.norm(b)):
                        U_mat = np.delete(U_mat, c, axis=1)
                        V_mat = np.delete(V_mat, c, axis=1)
                        W_mat = np.delete(W_mat, c, axis=1)
                        return U_mat, V_mat, W_mat, True
                except Exception:
                    continue

        return U_mat, V_mat, W_mat, False

    # Iterative peeling
    peeled = True
    while peeled and time.time() < deadline - 30:
        current_U, current_V, current_W, peeled = peel_round(current_U, current_V, current_W)

    if verify(current_U, current_V, current_W):
        if current_rank < rank:
            rank = current_rank
        # Try to go even lower with a targeted search

        best_U = current_U.copy()
        best_V = current_V.copy()
        best_W = current_W.copy()
        best_rank = current_rank

        # Try lower ranks with vectorized ALS
        for target_rank in range(best_rank - 1, max(best_rank - 12, 38), -1):
            if time.time() > deadline - 25:
                break

            # Multiple initializations: random + structured
            inits = []
            
            # Random initialization
            np.random.seed(42)
            U_init = np.random.randn(dim_u, target_rank) * 0.1
            V_init = np.random.randn(dim_v, target_rank) * 0.1
            W_init = np.random.randn(dim_w, target_rank) * 0.1
            inits.append((U_init.copy(), V_init.copy(), W_init.copy()))
            
            # Initialize from current decomposition's first target_rank columns
            if current_U.shape[1] >= target_rank:
                U_init2 = current_U[:, :target_rank].copy()
                V_init2 = current_V[:, :target_rank].copy()
                W_init2 = current_W[:, :target_rank].copy()
                inits.append((U_init2, V_init2, W_init2))
            
            # Structured initialization: perturb the Strassen structure
            if target_rank >= 50:
                # Take the Strassen columns and add small random perturbation
                U_init3 = current_U[:, :target_rank].copy() + np.random.randn(dim_u, target_rank) * 0.01
                V_init3 = current_V[:, :target_rank].copy() + np.random.randn(dim_v, target_rank) * 0.01
                W_init3 = current_W[:, :target_rank].copy() + np.random.randn(dim_w, target_rank) * 0.01
                inits.append((U_init3, V_init3, W_init3))

            for U_i, V_i, W_i in inits:
                if time.time() > deadline - 20:
                    break

                # Vectorized ALS with more iterations and momentum
                for iteration in range(50):
                    if time.time() > deadline - 20:
                        break

                    # Update U
                    VW = np.einsum('jc,kc->cjk', V_i, W_i).reshape(target_rank, -1)
                    for i in range(dim_u):
                        t_i = T_true[i, :, :].reshape(-1)
                        try:
                            U_i[i, :], _, _, _ = np.linalg.lstsq(VW.T, t_i, rcond=None)
                        except Exception:
                            pass

                    # Update V
                    UW = np.einsum('ic,kc->cik', U_i, W_i).reshape(target_rank, -1)
                    for j in range(dim_v):
                        t_j = T_true[:, j, :].reshape(-1)
                        try:
                            V_i[j, :], _, _, _ = np.linalg.lstsq(UW.T, t_j, rcond=None)
                        except Exception:
                            pass

                    # Update W
                    UV = np.einsum('ic,jc->cij', U_i, V_i).reshape(target_rank, -1)
                    for k in range(dim_w):
                        t_k = T_true[:, :, k].reshape(-1)
                        try:
                            W_i[k, :], _, _, _ = np.linalg.lstsq(UV.T, t_k, rcond=None)
                        except Exception:
                            pass

                    # Check convergence every 5 iterations
                    if iteration % 5 == 4:
                        T_rec = np.einsum('ic,jc,kc->ijk', U_i, V_i, W_i)
                        residual = np.linalg.norm(T_rec - T_true) / max(1e-12, np.linalg.norm(T_true))
                        if residual < 1e-9:
                            break

                # Final check
                if time.time() > deadline - 15:
                    break
                T_rec = np.einsum('ic,jc,kc->ijk', U_i, V_i, W_i)
                if np.allclose(T_rec, T_true, atol=1e-8):
                    best_U = U_i
                    best_V = V_i
                    best_W = W_i
                    best_rank = target_rank
                    break

            if best_rank == target_rank:
                break

        if best_rank < current_rank:
            current_U = best_U
            current_V = best_V
            current_W = best_W
            current_rank = best_rank

        if verify(current_U, current_V, current_W):
            return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}

    # Final verification of the Strassen decomposition
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col_s = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col_s] = 1
                V_school[j * p + k, col_s] = 1
                W_school[k * n + i, col_s] = 1
                col_s += 1
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
