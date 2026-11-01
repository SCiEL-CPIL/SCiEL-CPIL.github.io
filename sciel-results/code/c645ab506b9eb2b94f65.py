import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS  # n=2, m=2, p=9

    # Trivial decomposition fallback (Rank 36)
    def trivial_decomposition():
        rank_t = n * m * p
        U_t = [[0] * rank_t for _ in range(n * m)]
        V_t = [[0] * rank_t for _ in range(m * p)]
        W_t = [[0] * rank_t for _ in range(p * n)]
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_t[ii*m+jj][c] = 1
                    V_t[jj*p+kk][c] = 1
                    W_t[kk*n+ii][c] = 1
                    c += 1
        return {"U": U_t, "V": V_t, "W": W_t}

    # Build rank-32 decomposition
    def build_rank32():
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            U[0][col] = 1; U[3][col] = 1; V[k0][col] = 1; V[p+k1][col] = 1; W[2*k0][col] = 1; W[2*k1+1][col] = 1; col += 1
            U[2][col] = 1; U[3][col] = 1; V[k0][col] = 1; W[2*k0+1][col] = 1; W[2*k1+1][col] = -1; col += 1
            U[0][col] = 1; V[k1][col] = 1; V[p+k1][col] = -1; W[2*k1][col] = 1; W[2*k1+1][col] = 1; col += 1
            U[3][col] = 1; V[p+k0][col] = 1; V[k0][col] = -1; W[2*k0][col] = 1; W[2*k0+1][col] = 1; col += 1
            U[0][col] = 1; U[1][col] = 1; V[p+k1][col] = 1; W[2*k0][col] = -1; W[2*k1][col] = 1; col += 1
            U[2][col] = 1; U[0][col] = -1; V[k0][col] = 1; V[k1][col] = 1; W[2*k1+1][col] = 1; col += 1
            U[1][col] = 1; U[3][col] = -1; V[p+k0][col] = 1; V[p+k1][col] = 1; W[2*k0][col] = 1; col += 1

        def add_schoolbook(k):
            nonlocal col
            U[0][col] = 1; V[k][col] = 1; W[2*k][col] = 1; col += 1
            U[1][col] = 1; V[p+k][col] = 1; W[2*k][col] = 1; col += 1
            U[2][col] = 1; V[k][col] = 1; W[2*k+1][col] = 1; col += 1
            U[3][col] = 1; V[p+k][col] = 1; W[2*k+1][col] = 1; col += 1

        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_schoolbook(8)
        return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

    U32, V32, W32 = build_rank32()

    # Fast vectorized verification
    def verify_fast(U, V, W):
        """Vectorized check: for all (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1"""
        # Build tensors: U shape (n,m,rank), V shape (m,p,rank), W shape (p,n,rank)
        U_t = U.reshape(n, m, -1)
        V_t = V.reshape(m, p, -1)
        W_t = W.reshape(p, n, -1)
        # For each (i,j,k): sum_c U_t[i,j,c] * V_t[j,k,c] * W_t[k,i,c]
        # Use einsum: result[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        return np.allclose(result, 1.0, atol=1e-9)

    if not verify_fast(U32, V32, W32):
        return trivial_decomposition()

    # Optimizer for rank-r decomposition
    def optimize_rank(r, init_U, init_V, init_W, max_iter=300):
        """Optimize factor matrices to satisfy the tensor decomposition."""
        def objective(params):
            if time.time() > deadline - 3:
                return 1e10
            U_f = params[:n*m*r].reshape(n*m, r)
            V_f = params[n*m*r:2*n*m*r].reshape(m*p, r)
            W_f = params[2*n*m*r:].reshape(p*n, r)
            U_t = U_f.reshape(n, m, r)
            V_t = V_f.reshape(m, p, r)
            W_t = W_f.reshape(p, n, r)
            result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            return np.sum((result - 1.0)**2)

        def gradient(params):
            if time.time() > deadline - 3:
                return np.zeros_like(params)
            U_f = params[:n*m*r].reshape(n*m, r)
            V_f = params[n*m*r:2*n*m*r].reshape(m*p, r)
            W_f = params[2*n*m*r:].reshape(p*n, r)
            U_t = U_f.reshape(n, m, r)
            V_t = V_f.reshape(m, p, r)
            W_t = W_f.reshape(p, n, r)
            result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            err = 2.0 * (result - 1.0)  # shape (n,m,p)
            
            # dU[i,j,c] = sum_k err[i,j,k] * V[j,k,c] * W[k,i,c]
            dU_t = np.einsum('ijk,jkc,kic->ijc', err, V_t, W_t)
            # dV[j,k,c] = sum_i err[i,j,k] * U[i,j,c] * W[k,i,c]
            dV_t = np.einsum('ijk,ijc,kic->jkc', err, U_t, W_t)
            # dW[k,i,c] = sum_j err[i,j,k] * U[i,j,c] * V[j,k,c]
            dW_t = np.einsum('ijk,ijc,jkc->kic', err, U_t, V_t)
            
            return np.concatenate([dU_t.flatten(), dV_t.flatten(), dW_t.flatten()])

        x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < 1e-10:
                U_opt = res.x[:n*m*r].reshape(n*m, r)
                V_opt = res.x[n*m*r:2*n*m*r].reshape(m*p, r)
                W_opt = res.x[2*n*m*r:].reshape(p*n, r)
                if verify_fast(U_opt, V_opt, W_opt):
                    return U_opt, V_opt, W_opt
        except Exception:
            pass
        return None

    def try_single_removal():
        """Try removing each of the 32 terms and re-optimizing to rank 31."""
        if time.time() > deadline - 10:
            return None
        # Prioritize removing schoolbook terms (last 4) and then strassen terms
        # Try all 32 removals, but limit total time
        time_per_attempt = min(8.0, (deadline - time.time() - 5) / 32)
        for remove_idx in range(32):
            if time.time() > deadline - 5:
                break
            rank = 31
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            # Copy all columns except remove_idx
            cols_keep = [c for c in range(32) if c != remove_idx]
            for new_c, old_c in enumerate(cols_keep):
                U[:, new_c] = U32[:, old_c]
                V[:, new_c] = V32[:, old_c]
                W[:, new_c] = W32[:, old_c]
            result = optimize_rank(rank, U, V, W, max_iter=200)
            if result is not None:
                return result
        return None

    def try_pair_removal():
        """Try removing pairs of terms and re-optimizing to rank 30."""
        if time.time() > deadline - 15:
            return None
        # Focus on pairs that are structurally similar
        # Schoolbook terms: indices 28,29,30,31
        # Strassen blocks: 0-6, 7-13, 14-20, 21-27
        # Try pairs within same strassen block or schoolbook
        candidate_pairs = [
            (28, 29), (28, 30), (28, 31), (29, 30), (29, 31), (30, 31),
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
            (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
            (2, 3), (2, 4), (2, 5), (2, 6),
            (3, 4), (3, 5), (3, 6),
            (4, 5), (4, 6),
            (5, 6),
        ]
        # Also try cross-block pairs that share structure
        candidate_pairs += [
            (0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12), (6, 13),
            (7, 14), (8, 15), (9, 16), (10, 17), (11, 18), (12, 19), (13, 20),
            (14, 21), (15, 22), (16, 23), (17, 24), (18, 25), (19, 26), (20, 27),
            (21, 28), (22, 29), (23, 30), (24, 31),
        ]
        
        time_per_attempt = min(6.0, (deadline - time.time() - 5) / len(candidate_pairs))
        for r1, r2 in candidate_pairs:
            if time.time() > deadline - 5:
                break
            rank = 30
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            cols_keep = [c for c in range(32) if c != r1 and c != r2]
            for new_c, old_c in enumerate(cols_keep):
                U[:, new_c] = U32[:, old_c]
                V[:, new_c] = V32[:, old_c]
                W[:, new_c] = W32[:, old_c]
            result = optimize_rank(rank, U, V, W, max_iter=150)
            if result is not None:
                return result
        return None

    # Try rank-31 first (single removal)
    result31 = try_single_removal()
    if result31 is not None:
        U_opt, V_opt, W_opt = result31
        U_list = [[float(U_opt[i][j]) for j in range(31)] for i in range(n*m)]
        V_list = [[float(V_opt[i][j]) for j in range(31)] for i in range(m*p)]
        W_list = [[float(W_opt[i][j]) for j in range(31)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    # Try rank-30 (pair removal) if time permits
    if time.time() < deadline - 10:
        result30 = try_pair_removal()
        if result30 is not None:
            U_opt, V_opt, W_opt = result30
            U_list = [[float(U_opt[i][j]) for j in range(30)] for i in range(n*m)]
            V_list = [[float(V_opt[i][j]) for j in range(30)] for i in range(m*p)]
            W_list = [[float(W_opt[i][j]) for j in range(30)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Return rank-32
    U_list = [[float(U32[i][j]) for j in range(32)] for i in range(n*m)]
    V_list = [[float(V32[i][j]) for j in range(32)] for i in range(m*p)]
    W_list = [[float(W32[i][j]) for j in range(32)] for i in range(p*n)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
