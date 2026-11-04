"""Matrix multiplication 3x4x6 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    rank_schoolbook = n * m * p  # 72
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, idx] = 1.0
                V_sb[j * p + k, idx] = 1.0
                W_sb[k * n + i, idx] = 1.0
                idx += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    # Precompute matricizations for efficient ALS
    T_1 = T.reshape(n * m, m * p * p * n)  # Mode-1 unfolding
    T_2 = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)  # Mode-2 unfolding
    T_3 = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)  # Mode-3 unfolding

    def als_decompose(T, T_1, T_2, T_3, rank, max_iter=200, tol=1e-10, seed=42):
        """
        Perform ALS to decompose T into rank-1 tensors.
        Returns (U, V, W, error) if convergence is good, else (None, None, None, None).
        """
        if time.time() > DEADLINE - 10:
            return None, None, None, None
        
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > DEADLINE - 10:
                return None, None, None, None
            
            # Update U: For each column r, solve least squares
            # U[:, r] = argmin_u || T_1 - u * (V[:, r] ⊗ W[:, r]) ||_F
            # where ⊗ is the Kronecker product
            for r in range(rank):
                # Compute the "other" part: sum_{s != r} u_s v_s w_s
                # More efficiently: compute residual and update
                # Let B_r = (V[:, r] ⊗ W[:, r])
                B_r = np.kron(V[:, r], W[:, r])
                # Solve: u_r = (T_1 @ B_r) / (B_r @ B_r)
                num = T_1 @ B_r
                den = B_r @ B_r
                if den < 1e-12:
                    continue
                U[:, r] = num / den
            
            # Update V: For each column r
            for r in range(rank):
                B_r = np.kron(U[:, r], W[:, r])
                num = T_2 @ B_r
                den = B_r @ B_r
                if den < 1e-12:
                    continue
                V[:, r] = num / den
            
            # Update W: For each column r
            for r in range(rank):
                B_r = np.kron(U[:, r], V[:, r])
                num = T_3 @ B_r
                den = B_r @ B_r
                if den < 1e-12:
                    continue
                W[:, r] = num / den
            
            # Compute error
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T - T_recon, 'fro')
            
            if abs(prev_err - err) < tol * max(1, prev_err):
                break
            prev_err = err
        
        # Verify convergence
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(T - T_recon, 'fro')
        if err < 1e-8:
            return U, V, W, err
        return None, None, None, None

    # Try ranks from 71 down to 60 with multiple seeds
    for rank in range(rank_schoolbook - 1, 59, -1):
        if time.time() > DEADLINE - 20:
            break
        
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > DEADLINE - 15:
                break
            
            U, V, W, err = als_decompose(T, T_1, T_2, T_3, rank, max_iter=150, tol=1e-10, seed=seed)
            
            if U is not None and err < 1e-8:
                # Verify exactness
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.allclose(T_recon, T, atol=1e-9):
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    # Found a better rank, continue searching for even lower
                    break
        
        # If we found a rank, we can stop early if time is running out
        if best_rank < rank_schoolbook and time.time() > DEADLINE - 30:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
