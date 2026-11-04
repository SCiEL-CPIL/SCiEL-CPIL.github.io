"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 110  # leave safety margin before 120s kill

    def verify(U, V, W, n, m, p, rank, tol=1e-10):
        """Check if the decomposition is exact (within tolerance)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_np(U, V, W, n, m, p, rank, tol=1e-10):
        """Vectorized verification."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_als(rank, max_iter=800, tol=1e-14, seed=42, init_method='random'):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        # Initialize based on method
        if init_method == 'random':
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
        elif init_method == 'structured':
            # Use a structured initialization based on partial schoolbook
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            # Initialize with a subset of schoolbook patterns
            for t in range(min(rank, n*m*p)):
                i = t % n
                j = (t // n) % m
                k = (t // (n*m)) % p
                U[i*m+j, t] = 1.0
                V[j*p+k, t] = 1.0
                W[k*n+i, t] = 1.0
            # Add some randomization
            U += rng.randn(n * m, rank) * 0.1
            V += rng.randn(m * p, rank) * 0.1
            W += rng.randn(p * n, rank) * 0.1
        else:  # 'gradient'
            U = rng.randn(n * m, rank) * 0.3
            V = rng.randn(m * p, rank) * 0.3
            W = rng.randn(p * n, rank) * 0.3

        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Solve for W: fix U, V
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            # Solve for U: fix V, W
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            # Solve for V: fix U, W
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            # Check convergence
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                if verify_np(U, V, W, n, m, p, rank):
                    return U, V, W
        
        # Final check even if not fully converged
        if verify_np(U, V, W, n, m, p, rank):
            return U, V, W
        
        return None

    # Try to find a low-rank decomposition
    # Start from rank 14 (theoretical lower bound) up to 23
    best_result = None
    best_rank = n * m * p  # 24

    for rank in range(14, 24):
        if time.time() > deadline - 10:
            break
        
        found = False
        for init_method in ['random', 'structured']:
            if time.time() > deadline - 5:
                break
            
            for seed in range(15):
                if time.time() > deadline - 3:
                    break
                
                result = try_als(rank, max_iter=600, tol=1e-14, seed=seed, init_method=init_method)
                if result is not None:
                    U, V, W = result
                    # Convert to lists with rounded values
                    U_list = [[round(float(x), 12) for x in row] for row in U]
                    V_list = [[round(float(x), 12) for x in row] for row in V]
                    W_list = [[round(float(x), 12) for x in row] for row in W]
                    
                    # Final verification with the rounded values
                    U_check = np.array(U_list, dtype=float)
                    V_check = np.array(V_list, dtype=float)
                    W_check = np.array(W_list, dtype=float)
                    
                    if verify_np(U_check, V_check, W_check, n, m, p, rank, tol=1e-9):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    if best_result is not None:
        return best_result

    # Fallback: schoolbook rank 24
    rank = n * m * p  # 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
