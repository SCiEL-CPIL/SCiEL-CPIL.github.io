"""Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition search.

Strategy: Kronecker product decomposition + ALS optimization
  - Factor p=8 as 4*2, giving two independent 3x4x4 multiplications
  - Use ALS to find low-rank decompositions
  - High-precision polish pass for exactness
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 96
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid i,j,k
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook fallback
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
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    def polish(U, V, W, T, n_iter=20):
        """High-precision polish pass."""
        for it in range(n_iter):
            # Update U
            S = np.einsum('jr,kr->jk', V, W)
            SS = S @ S.T
            for a in range(U.shape[0]):
                rhs = T[a] @ S.T
                try:
                    U[a] = np.linalg.solve(SS, rhs)
                except np.linalg.LinAlgError:
                    U[a] = np.linalg.lstsq(SS, rhs, rcond=None)[0]
            
            S = np.einsum('ir,kr->ik', U, W)
            SS = S @ S.T
            for b in range(V.shape[0]):
                rhs = T[:, b, :] @ S.T
                try:
                    V[b] = np.linalg.solve(SS, rhs)
                except np.linalg.LinAlgError:
                    V[b] = np.linalg.lstsq(SS, rhs, rcond=None)[0]
            
            S = np.einsum('ir,jr->ij', U, V)
            SS = S @ S.T
            for c in range(W.shape[0]):
                rhs = T[:, :, c] @ S.T
                try:
                    W[c] = np.linalg.solve(SS, rhs)
                except np.linalg.LinAlgError:
                    W[c] = np.linalg.lstsq(SS, rhs, rcond=None)[0]
        return U, V, W
    
    def als_decompose(rank, n_iter=100, seed=42):
        """Alternating least squares for tensor decomposition."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        for it in range(n_iter):
            if time.time() > deadline:
                break
            # Update U
            S = np.einsum('jr,kr->jk', V, W)
            SS = S @ S.T
            for a in range(n * m):
                Ta = T[a]
                rhs = Ta @ S.T
                try:
                    u_a = np.linalg.solve(SS, rhs)
                except np.linalg.LinAlgError:
                    u_a = np.linalg.lstsq(SS, rhs, rcond=None)[0]
                U[a] = u_a
            
            # Update V
            S = np.einsum('ir,kr->ik', U, W)
            SS = S @ S.T
            for b in range(m * p):
                Tb = T[:, b, :]
                rhs = Tb @ S.T
                try:
                    v_b = np.linalg.solve(SS, rhs)
                except np.linalg.LinAlgError:
                    v_b = np.linalg.lstsq(SS, rhs, rcond=None)[0]
                V[b] = v_b
            
            # Update W
            S = np.einsum('ir,jr->ij', U, V)
            SS = S @ S.T
            for c in range(p * n):
                Tc = T[:, :, c]
                rhs = Tc @ S.T
                try:
                    w_c = np.linalg.solve(SS, rhs)
                except np.linalg.LinAlgError:
                    w_c = np.linalg.lstsq(SS, rhs, rcond=None)[0]
                W[c] = w_c
        
        # Compute residual
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = np.max(np.abs(T - S))
        return U, V, W, residual
    
    # Try various ranks
    for rank in [96, 80, 64, 48, 40, 32]:
        if time.time() > deadline:
            break
        for seed in range(5):
            if time.time() > deadline:
                break
            U, V, W, residual = als_decompose(rank, n_iter=50, seed=seed)
            if residual < 1e-6:
                # Found a good decomposition!
                # Polish it
                U, V, W = polish(U, V, W, T, n_iter=20)
                S = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.max(np.abs(T - S)) < 1e-10:
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
# EVOLVE-BLOCK-END
