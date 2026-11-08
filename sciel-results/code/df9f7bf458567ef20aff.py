import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = 4, 4, 7
    num_u = n * m  # 16
    num_v = m * p  # 28
    num_w = p * n  # 28

    # Build the full 4x4x7 multiplication tensor T of shape (16, 28, 28)
    # T[u_idx, v_idx, w_idx] = 1 where u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i
    T = np.zeros((num_u, num_v, num_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def verify(U, V, W):
        """Check if U, V, W (as numpy arrays) exactly decompose T."""
        rank = U.shape[1]
        # Compute sum_r U[:,r] * V[:,r] * W[:,r]
        # Reshape for efficient computation
        # T_reconstructed[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        # Check exact match (with small tolerance for floating point)
        return np.allclose(T_rec, T, atol=1e-9, rtol=1e-9)

    def round_to_half_int(x):
        """Round each element to nearest half-integer (multiple of 0.5)."""
        return np.round(x * 2.0) / 2.0

    def als_decompose(T, R, max_iter=200, tol=1e-10, seed=None):
        """
        Alternating Least Squares for CP decomposition of tensor T.
        T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        Returns U (num_u x R), V (num_v x R), W (num_w x R)
        """
        rng = np.random.default_rng(seed)
        # Initialize with small random values
        U = rng.standard_normal((num_u, R)) * 0.1
        V = rng.standard_normal((num_v, R)) * 0.1
        W = rng.standard_normal((num_w, R)) * 0.1

        for it in range(max_iter):
            # Update U: for each r, U[:,r] = (T * (V[:,r] ⊙ W[:,r])) / (||V[:,r]||^2 * ||W[:,r]||^2)
            # T * (V[:,r] ⊙ W[:,r]) means element-wise product of T with outer product of V[:,r] and W[:,r]
            # Reshape T to (num_u, num_v*num_w)
            T_flat = T.reshape(num_u, num_v * num_w)
            for r in range(R):
                # Compute V[:,r] ⊙ W[:,r] as vector of length num_v*num_w
                # Actually, we need: for each a, sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
                # = (T.reshape(num_u, -1)) @ (V[:,r] * W[:,r]).reshape(num_v*num_w)
                vw = (V[:, r] * W[:, r]).reshape(-1)
                rhs = T_flat @ vw
                denom = np.dot(V[:, r], V[:, r]) * np.dot(W[:, r], W[:, r])
                if abs(denom) > 1e-15:
                    U[:, r] = rhs / denom

            # Update V
            T_flat_v = T.transpose(1, 0, 2).reshape(num_v, num_u * num_w)
            for r in range(R):
                uw = (U[:, r] * W[:, r]).reshape(-1)
                rhs = T_flat_v @ uw
                denom = np.dot(U[:, r], U[:, r]) * np.dot(W[:, r], W[:, r])
                if abs(denom) > 1e-15:
                    V[:, r] = rhs / denom

            # Update W
            T_flat_w = T.transpose(2, 0, 1).reshape(num_w, num_u * num_v)
            for r in range(R):
                uv = (U[:, r] * V[:, r]).reshape(-1)
                rhs = T_flat_w @ uv
                denom = np.dot(U[:, r], U[:, r]) * np.dot(V[:, r], V[:, r])
                if abs(denom) > 1e-15:
                    W[:, r] = rhs / denom

            # Check convergence
            T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = np.linalg.norm(T_rec - T) / (np.linalg.norm(T) + 1e-15)
            if err < tol:
                break

        return U, V, W, err

    def try_rank(R, num_trials=50, seed_base=42):
        """Try to find an exact rank-R decomposition via ALS + rounding."""
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None

            seed = seed_base + trial
            U, V, W, err = als_decompose(T, R, max_iter=150, tol=1e-12, seed=seed)

            if err > 0.01:
                continue

            # Try rounding to half-integers
            U_r = round_to_half_int(U)
            V_r = round_to_half_int(V)
            W_r = round_to_half_int(W)

            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r

            # Also try rounding to integers
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            if verify(U_i, V_i, W_i):
                return U_i, V_i, W_i

            # Try mixed: some columns integer, some half-integer
            # Just check if the unrounded solution is already very close to half-integers
            U_close = np.allclose(U, U_r, atol=1e-6)
            V_close = np.allclose(V, V_r, atol=1e-6)
            W_close = np.allclose(W, W_r, atol=1e-6)
            if U_close and V_close and W_close:
                # Very close, try exact verification again
                if verify(U_r, V_r, W_r):
                    return U_r, V_r, W_r

        return None

    # Try ranks from low to high
    # Lower bound for 4x4x7: at least rank(4x4x4) = 28 (but likely higher)
    # Upper bound: schoolbook = 112, current best = 93
    # Try: 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92
    candidate_ranks = [28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 64, 68, 72, 76, 80, 84, 88, 92]

    best_result = None
    for R in candidate_ranks:
        if time.time() > deadline - 20:
            break

        result = try_rank(R, num_trials=30)
        if result is not None:
            U, V, W = result
            # Convert to lists for output
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            best_result = (R, U_list, V_list, W_list)
            break  # Found minimal rank, stop

    if best_result is None:
        # Fallback: use the known 93-rank structural decomposition
        # Build it manually as a safety net
        terms = []

        def add_term(u, v, w):
            terms.append((u, v, w))

        # Schoolbook for the whole thing as ultimate fallback (rank 112)
        # But let's try the Strassen head + schoolbook tail (49+48=97) or just schoolbook
        # Actually, let's just do schoolbook to be safe
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * (n * m)
                    u[i * m + j] = 1
                    v = [0] * (m * p)
                    v[j * p + k] = 1
                    w = [0] * (p * n)
                    w[k * n + i] = 1
                    add_term(u, v, w)

        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        for c, (u, v, w) in enumerate(terms):
            for r in range(n * m):
                if u[r] != 0:
                    U[r][c] = u[r]
            for r in range(m * p):
                if v[r] != 0:
                    V[r][c] = v[r]
            for r in range(p * n):
                if w[r] != 0:
                    W[r][c] = w[r]

        return {"U": U, "V": V, "W": W}

    R, U_list, V_list, W_list = best_result
    return {"U": U_list, "V": V_list, "W": W_list}
