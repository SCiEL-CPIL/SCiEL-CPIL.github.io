"""Suite task: rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 12
    M = m * p  # 20
    P = p * n  # 15

    start = time.time()
    deadline = start + 110.0  # safety margin before 120s limit

    # Build the flattening tensor T: T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((N, M, P), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check_rank(r, U, V, W):
        """Verify U @ (V * W) reproduces T exactly."""
        # U: (N, r), V: (M, r), W: (P, r)
        # T[i, j, k] should equal sum_r U[i,r]*V[j,r]*W[k,r]
        # Build: for each r, outer product U[:,r] * V[:,r] * W[:,r]
        # Efficient: S = U @ V.T  (N, M), then T_hat[i,j,k] = sum_r S[i,j] * W[k,r]
        # Actually T_hat = einsum('ir,jr,kr->ijk', U, V, W)
        T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_hat, T, rtol=0, atol=1e-9)

    def rank_via_random(r, seed=0):
        """Try to find rank-r decomposition via randomized sketching + SVD."""
        rng = np.random.RandomState(seed)
        # Random matrix to project: R (M x r)
        R = rng.randn(M, r)
        # Sketch: S = T_{(2)} (M, N*P) @ R -> (N*P, r)
        # T reshaped: T2 = T.transpose(1,0,2).reshape(M, N*P)
        T2 = T.transpose(1, 0, 2).reshape(M, N * P)
        S = T2 @ R  # (N*P, r)
        # SVD of S
        try:
            U_s, s_vals, Vt = np.linalg.svd(S, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        # U_s: (N*P, min(N*P,r)), keep top r
        U_s = U_s[:, :r]
        # V: from R, adjust: V = R @ U_s.T? 
        # Actually: T2 ≈ V @ U_s.T where V = R, U_s from SVD
        # T2[i, j*p+k] ≈ sum_s V[i,s] * U_s[j*p+k, s]
        # V: (M, r), U_s: (N*P, r)
        # Reshape U_s to (N, P) for W: U_s[i*P+k, s] -> W[k, i*P?] 
        # Wait: T[i,j,k] = sum_s V[j,s] * U_s[i*P+k, s]
        # So U[i, s] and W[k, s]: T[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s]
        # We need U_s[i*P+k, s] = U[i,s] * W[k,s] -- can't separate directly.
        # Better approach: use 3-way SVD / ALS
        return None

    def als_decomposition(r, iters=50, seed=0, lr=0.1):
        """Alternating least squares for tensor rank-r decomposition."""
        rng = np.random.RandomState(seed)
        # Initialize
        U = rng.randn(N, r) * 0.1
        V = rng.randn(M, r) * 0.1
        W = rng.randn(P, r) * 0.1

        for it in range(iters):
            # Update U: for each i, solve for U[i,:]
            # T[i,j,k] = sum_s U[i,s] * V[j,s] * W[k,s]
            # T[i] = (V * W) @ U[i].T  where (V*W)[j,k] = sum_s V[j,s]*W[k,s] = (V @ W.T)[j,k]
            VW = V @ W.T  # (M, P)
            for i in range(N):
                # T[i, :, :] = VW @ U[i, :].T
                # Flatten: T[i].reshape(M*P) = (VW.reshape(M*P)) * U[i] via Khatri-Rao
                # Actually: T[i,j,k] = sum_s U[i,s] * VW[j,k]
                # So T[i].reshape(M*P,1) = VW.reshape(M*P,1) * U[i]  -> no, that's element-wise
                # T[i,j,k] = sum_s U[i,s] * VW[j,k]
                # Vectorize: T_i_flat = (VW_flat) .* ? No.
                # T_i = VW @ diag(U[i]) ... no
                # T[i,j,k] = (VW * U[i].T) summed over s? No.
                # T[i,j,k] = sum_s U[i,s] * VW[j,k] -- but VW[j,k] doesn't depend on s!
                # That's wrong. VW[j,k] = sum_s V[j,s]*W[k,s].
                # T[i,j,k] = sum_s U[i,s] * V[j,s] * W[k,s]
                # For fixed i: T[i] = sum_s U[i,s] * outer(V[:,s], W[:,s])
                # T[i].reshape(M*P) = sum_s U[i,s] * (V[:,s] * W[:,s]).reshape(M*P)
                # = KR(V.T, W.T).T @ U[i]  where KR gives (M*P, r)
                KR = (V * W.T).reshape(M * P, r)  # wait, this is element-wise V[j,s]*W[k,s] indexed by (j,k,s)
                # V is (M, r), W is (P, r). V[:,s]*W[:,s] for fixed s gives (M, P).
                # Stack over s: shape (M, P, r) -> reshape to (M*P, r)
                KR = (V[:, :, np.newaxis] * W[np.newaxis, :, :]).reshape(M * P, r)
                # T[i].reshape(M*P) = KR.T @ U[i]
                rhs = T[i].reshape(M * P)
                # Solve: KR.T @ U[i] = rhs  -> U[i] = (KR.T @ KR)^-1 KR.T @ rhs
                A = KR.T @ KR  # (r, r)
                b = KR.T @ rhs  # (r,)
                try:
                    U[i] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    U[i] = np.linalg.lstsq(KR.T, rhs, rcond=None)[0]

            # Update V similarly
            UW = U @ W.T  # not directly useful; do per-column
            for j in range(M):
                # T[:,j,k] = sum_s U[i,s] * V[j,s] * W[k,s]
                # For fixed j: T[:,j,k] = sum_s V[j,s] * (U[:,s] * W[k,s])
                # T[:,j].reshape(N*P) = KR2.T @ V[j] where KR2 from U and W
                KR2 = (U[:, :, np.newaxis] * W[np.newaxis, :, :]).reshape(N * P, r)
                rhs = T[:, j, :].reshape(N * P)
                A = KR2.T @ KR2
                b = KR2.T @ rhs
                try:
                    V[j] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    V[j] = np.linalg.lstsq(KR2.T, rhs, rcond=None)[0]

            # Update W similarly
            for k in range(P):
                KR3 = (U[:, :, np.newaxis] * V[np.newaxis, :, :]).reshape(N * M, r)
                rhs = T[:, :, k].reshape(N * M)
                A = KR3.T @ KR3
                b = KR3.T @ rhs
                try:
                    W[k] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    W[k] = np.linalg.lstsq(KR3.T, rhs, rcond=None)[0]

            # Check convergence
            if it % 10 == 0 or it == iters - 1:
                T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.max(np.abs(T_hat - T))
                if err < 1e-8:
                    return U, V, W, err

        T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.max(np.abs(T_hat - T))
        if err < 1e-7:
            return U, V, W, err
        return None

    # Schoolbook as fallback
    def schoolbook():
        rank = N * M * P  # 60
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Try to find lower rank
    best_result = None
    # Try ranks from a lower bound upward
    # Lower bound: max(N, M, P) = 20, but realistically higher
    # Known: rank of matrix mult (n x m) * (m x p) is between max(n*p, m^2) and n*m*p
    # For 3x4*4x5: lower bound ~ max(15, 16) = 16, upper 60
    # Try a range of ranks with multiple seeds
    
    # First, quickly try some promising ranks
    for r in [20, 24, 28, 30, 32, 35, 40, 45, 50]:
        if time.time() > deadline:
            break
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            res = als_decomposition(r, iters=80, seed=seed)
            if res is not None:
                U, V, W, err = res
                if check_rank(r, U, V, W):
                    best_result = (r, U, V, W)
                    found = True
                    break
        if found:
            break  # Got a valid one at this rank, try lower? No, we're going up.
    
    if best_result is None:
        # Try a few more ranks
        for r in [55, 58, 59]:
            if time.time() > deadline:
                break
            for seed in range(3):
                if time.time() > deadline:
                    break
                res = als_decomposition(r, iters=60, seed=seed)
                if res is not None:
                    U, V, W, err = res
                    if check_rank(r, U, V, W):
                        best_result = (r, U, V, W)
                        break
            if best_result is not None:
                break

    if best_result is None:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        r = 60
    else:
        r, U, V, W = best_result

    # Convert to lists with rounded values for cleanliness
    U_list = [[float(round(U[i][j], 10)) for j in range(r)] for i in range(N)]
    V_list = [[float(round(V[i][j], 10)) for j in range(r)] for i in range(M)]
    W_list = [[float(round(W[i][j], 10)) for j in range(r)] for i in range(P)]

    # Final validation
    U_np = np.array(U_list)
    V_np = np.array(V_list)
    W_np = np.array(W_list)
    if not check_rank(r, U_np, V_np, W_np):
        # Revert to schoolbook if rounding broke it
        U, V, W = schoolbook()
        r = 60
        U_list = [[float(U[i][j]) for j in range(r)] for i in range(N)]
        V_list = [[float(V[i][j]) for j in range(r)] for i in range(M)]
        W_list = [[float(W[i][j]) for j in range(r)] for i in range(P)]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
