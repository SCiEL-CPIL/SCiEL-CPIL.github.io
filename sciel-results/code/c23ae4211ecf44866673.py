import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 142.0  # 8s safety margin before 150s kill

    n, m, p = 3, 3, 6
    dim_U = n * m  # 9
    dim_V = m * p  # 18
    dim_W = p * n  # 18

    # Build target tensor: T_target[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def compute_residual(U, V, W):
        """Compute sum_r U[:,r] V[:,r] W[:,r] - T_target squared norm."""
        R = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((R - T_target) ** 2)

    def snap_and_verify(U, V, W):
        """Snap to nearest half-integer and verify exactness."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        if np.allclose(R, T_target, atol=1e-9):
            return True, U_s.tolist(), V_s.tolist(), W_s.tolist()
        return False, None, None, None

    def als_optimize(U, V, W, max_iter=200, tol=1e-12):
        """Optimized ALS: precompute R_full once per factor-update block."""
        dim_U, rank = U.shape
        dim_V, _ = V.shape
        dim_W, _ = W.shape

        # Precompute contractions of T_target for each factor update
        # For U update: T_contract_U[i] = sum_{j,k} A[j]*B[k]*T_target[i,j,k]
        # But A and B depend on r, so we can't fully precompute.
        # However, we can precompute T_target reshaped for faster einsum.
        T_U = T_target  # shape (dim_U, dim_V, dim_W)
        T_V = T_target
        T_W = T_target

        for iteration in range(max_iter):
            if time.time() > deadline - 15:
                break

            old_res = compute_residual(U, V, W)

            # === Update U ===
            R_full = np.einsum('ir,jr,kr->ijk', U, V, W)
            for r in range(rank):
                A = V[:, r]
                B = W[:, r]
                A_sq = np.dot(A, A)
                B_sq = np.dot(B, B)
                denom = A_sq * B_sq
                if denom < 1e-15:
                    continue
                R_r = np.einsum('i,j,k->ijk', U[:, r], A, B)
                R_other = R_full - R_r
                rhs = np.einsum('j,k,ijk->i', A, B, T_U) - np.einsum('j,k,ijk->i', A, B, R_other)
                U[:, r] = rhs / denom

            # === Update V ===
            R_full = np.einsum('ir,jr,kr->ijk', U, V, W)
            for r in range(rank):
                A = U[:, r]
                B = W[:, r]
                A_sq = np.dot(A, A)
                B_sq = np.dot(B, B)
                denom = A_sq * B_sq
                if denom < 1e-15:
                    continue
                R_r = np.einsum('i,j,k->ijk', A, V[:, r], B)
                R_other = R_full - R_r
                rhs = np.einsum('i,k,ijk->j', A, B, T_V) - np.einsum('i,k,ijk->j', A, B, R_other)
                V[:, r] = rhs / denom

            # === Update W ===
            R_full = np.einsum('ir,jr,kr->ijk', U, V, W)
            for r in range(rank):
                A = U[:, r]
                B = V[:, r]
                A_sq = np.dot(A, A)
                B_sq = np.dot(B, B)
                denom = A_sq * B_sq
                if denom < 1e-15:
                    continue
                R_r = np.einsum('i,j,k->ijk', A, B, W[:, r])
                R_other = R_full - R_r
                rhs = np.einsum('i,j,ijk->k', A, B, T_W) - np.einsum('i,j,ijk->k', A, B, R_other)
                W[:, r] = rhs / denom

            new_res = compute_residual(U, V, W)
            if abs(old_res - new_res) < tol:
                break

        return U, V, W

    def get_schoolbook(rank):
        """Standard schoolbook decomposition with rank = n*m*p."""
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Try ranks from 42 to 54, using optimized ALS
    for rank in range(42, 55):
        if time.time() > deadline - 25:
            break
        
        # More attempts for lower ranks where success is harder
        num_attempts = 12 if rank <= 45 else 8 if rank <= 50 else 4
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(rank * 1000 + attempt * 37 + 13)
            
            if attempt == 0:
                # Schoolbook-like initialization with noise
                U = np.zeros((dim_U, rank))
                V = np.zeros((dim_V, rank))
                W = np.zeros((dim_W, rank))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < rank:
                                U[i*m+j, col] = 1.0
                                V[j*p+k, col] = 1.0
                                W[k*n+i, col] = 1.0
                            col += 1
                # Add small noise to break symmetry
                U += np.random.randn(dim_U, rank) * 0.05
                V += np.random.randn(dim_V, rank) * 0.05
                W += np.random.randn(dim_W, rank) * 0.05
            elif attempt == 1:
                # Scaled random
                U = np.random.randn(dim_U, rank) * 0.4
                V = np.random.randn(dim_V, rank) * 0.4
                W = np.random.randn(dim_W, rank) * 0.4
            elif attempt == 2:
                # Binary-like init
                U = np.random.choice([-1.0, 0.0, 1.0], size=(dim_U, rank))
                V = np.random.choice([-1.0, 0.0, 1.0], size=(dim_V, rank))
                W = np.random.choice([-1.0, 0.0, 1.0], size=(dim_W, rank))
            else:
                # Standard random
                U = np.random.randn(dim_U, rank) * 0.5
                V = np.random.randn(dim_V, rank) * 0.5
                W = np.random.randn(dim_W, rank) * 0.5
            
            # Run ALS
            U_opt, V_opt, W_opt = als_optimize(U, V, W, max_iter=120, tol=1e-12)
            
            # Check if we got close to zero residual
            res = compute_residual(U_opt, V_opt, W_opt)
            if res < 1e-8:
                success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    return {"U": U_s, "V": V_s, "W": W_s}
            
            # Polish pass if close
            if res < 1e-4:
                U_opt2, V_opt2, W_opt2 = als_optimize(U_opt, V_opt, W_opt, max_iter=80, tol=1e-14)
                res2 = compute_residual(U_opt2, V_opt2, W_opt2)
                if res2 < 1e-8:
                    success, U_s, V_s, W_s = snap_and_verify(U_opt2, V_opt2, W_opt2)
                    if success:
                        return {"U": U_s, "V": V_s, "W": W_s}

    # Fallback to schoolbook (rank 54)
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
