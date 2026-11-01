import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n, m, p = 2, 2, 11

    # Build the target tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # T shape: (n*m, m*p, p*n) = (4, 22, 22)
    # T[a_idx, b_idx, c_idx] = 1 if c_idx corresponds to the product
    # a_idx: i*m+j (A row i, col j)
    # b_idx: j*p+k (B row j, col k)
    # c_idx: k*n+i (C row i, col k) -- trace convention
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    def build_decomposition(U, V, W):
        """Build the result dict from factor matrices."""
        rank = U.shape[1]
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    def verify_exact(U, V, W, tol=1e-9):
        """Check if the decomposition exactly reconstructs T."""
        rank = U.shape[1]
        # Reconstruct tensor
        recon = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for r in range(rank):
            recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        return np.allclose(recon, T, atol=tol)

    def round_to_half(x):
        """Round to nearest half-integer."""
        return np.round(2 * x) / 2.0

    def als_cp(T, rank, n_iter=200, tol=1e-10, seed=42):
        """ALS for CP decomposition."""
        rng = np.random.RandomState(seed)
        n1, n2, n3 = T.shape
        U = rng.randn(n1, rank)
        V = rng.randn(n2, rank)
        W = rng.randn(n3, rank)
        prev_err = np.inf
        for it in range(n_iter):
            if time.time() > deadline - 30:
                break
            # Update U
            for r in range(rank):
                # T_{1..} contracted with V_r and W_r
                # T contracted over axes 1,2 with V[:,r] and W[:,r]
                # = sum_{j,k} T[i,j,k] * V[j,r] * W[k,r] for each i
                TW = np.einsum('jkr,k->jr', T, W[:, r])
                U[:, r] = np.einsum('ijr,j->i', TW, V[:, r])
                # Actually let's do it properly with matmuls
            # More efficient: use matmul
            # U_new[:,r] = T_{1..} * (V[:,r] ⊗ W[:,r])
            # T reshaped: T_{1..} is (n1, n2*n3)
            T_123 = T.reshape(n1, n2 * n3)
            VW = np.outer(V[:, r], W[:, r]).reshape(-1)
            U[:, r] = T_123 @ VW

            # Update V
            T_213 = T.reshape(n2, n1 * n3)
            UW = np.outer(U[:, r], W[:, r]).reshape(-1)
            V[:, r] = T_213 @ UW

            # Update W
            T_312 = T.reshape(n3, n1 * n2)
            UV = np.outer(U[:, r], V[:, r]).reshape(-1)
            W[:, r] = T_312 @ UV

            # Compute error
            recon = np.zeros_like(T)
            for r in range(rank):
                recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(recon - T)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        return U, V, W

    def als_cp_batch(T, rank, n_iter=300, tol=1e-10, seed=42):
        """ALS for CP decomposition - batch version."""
        rng = np.random.RandomState(seed)
        n1, n2, n3 = T.shape
        U = rng.randn(n1, rank)
        V = rng.randn(n2, rank)
        W = rng.randn(n3, rank)
        prev_err = np.inf
        for it in range(n_iter):
            if time.time() > deadline - 20:
                break
            for r in range(rank):
                T_123 = T.reshape(n1, n2 * n3)
                VW = np.outer(V[:, r], W[:, r]).reshape(-1)
                U[:, r] = T_123 @ VW

                T_213 = T.reshape(n2, n1 * n3)
                UW = np.outer(U[:, r], W[:, r]).reshape(-1)
                V[:, r] = T_213 @ UW

                T_312 = T.reshape(n3, n1 * n2)
                UV = np.outer(U[:, r], V[:, r]).reshape(-1)
                W[:, r] = T_312 @ UV

            recon = np.zeros_like(T)
            for r in range(rank):
                recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            err = np.linalg.norm(recon - T)
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        return U, V, W

    # Try to find a lower-rank exact decomposition
    best_result = None
    best_rank = 39  # default: Strassen+schoolbook

    # Try ranks from 32 to 38
    for trial_rank in range(32, 39):
        if time.time() > deadline - 40:
            break
        found = False
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            U, V, W = als_cp_batch(T, trial_rank, n_iter=500, seed=trial_rank * 100 + attempt)
            # Round to half-integers
            U_r = round_to_half(U)
            V_r = round_to_half(V)
            W_r = round_to_half(W)
            if verify_exact(U_r, V_r, W_r, tol=1e-9):
                best_result = build_decomposition(U_r, V_r, W_r)
                best_rank = trial_rank
                found = True
                break
        if found:
            break

    # If no lower rank found, use the Strassen+schoolbook construction (rank 39)
    if best_result is None:
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0

        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1

            v_e = k0
            v_f = k1
            v_g = 11 + k0
            v_h = 11 + k1

            w_c00 = 2 * k0
            w_c10 = 2 * k0 + 1
            w_c01 = 2 * k1
            w_c11 = 2 * k1 + 1

            # m1 = (a+d)(e+h)
            U[0][r] += 1; U[3][r] += 1
            V[v_e][r] += 1; V[v_h][r] += 1
            W[w_c00][r] += 1; W[w_c11][r] += 1
            r += 1

            # m2 = (c+d)e
            U[2][r] += 1; U[3][r] += 1
            V[v_e][r] += 1
            W[w_c10][r] += 1; W[w_c11][r] += -1
            r += 1

            # m3 = a(f-h)
            U[0][r] += 1
            V[v_f][r] += 1; V[v_h][r] += -1
            W[w_c01][r] += 1; W[w_c11][r] += 1
            r += 1

            # m4 = d(g-e)
            U[3][r] += 1
            V[v_g][r] += 1; V[v_e][r] += -1
            W[w_c00][r] += 1; W[w_c10][r] += 1
            r += 1

            # m5 = (a+b)h
            U[0][r] += 1; U[1][r] += 1
            V[v_h][r] += 1
            W[w_c00][r] += -1; W[w_c01][r] += 1
            r += 1

            # m6 = (c-a)(e+f)
            U[2][r] += 1; U[0][r] += -1
            V[v_e][r] += 1; V[v_f][r] += 1
            W[w_c11][r] += 1
            r += 1

            # m7 = (b-d)(g+h)
            U[1][r] += 1; U[3][r] += -1
            V[v_g][r] += 1; V[v_h][r] += 1
            W[w_c00][r] += 1
            r += 1

            if time.time() > deadline:
                break

        # Last column k=10: schoolbook
        if r < rank:
            k2 = 10
            v_e2 = k2
            v_g2 = 11 + k2
            w_c02 = 2 * k2
            w_c12 = 2 * k2 + 1

            U[0][r] += 1
            V[v_e2][r] += 1
            W[w_c02][r] += 1
            r += 1

            U[1][r] += 1
            V[v_g2][r] += 1
            W[w_c02][r] += 1
            r += 1

            U[2][r] += 1
            V[v_e2][r] += 1
            W[w_c12][r] += 1
            r += 1

            U[3][r] += 1
            V[v_g2][r] += 1
            W[w_c12][r] += 1
            r += 1

        best_result = {"U": U, "V": V, "W": W}

    return best_result
