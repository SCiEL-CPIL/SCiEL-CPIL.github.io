"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
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
    return U, V, W

def _try_rank(n, m, p, rank, max_iter=1000, tol=1e-8, seed=0):
    """Try to find a decomposition of given rank using alternating minimization."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    np.random.seed(seed)
    
    # Initialize randomly
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Alternating minimization
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            return None
        
        # Update U: for each (i,j), solve for U[i*m+j, :] given V, W
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                try:
                    x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :] given U, W
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        B[i, r] = U[i * m + j, r] * W[k * n + i, r]
                try:
                    x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :] given U, V
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        C[j, r] = U[i * m + j, r] * V[j * p + k, r]
                try:
                    x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Check convergence
        if it % 5 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err = max(err, abs(s - 1.0))
            if err < tol:
                # Success! Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                # Round to nearest half-integer if close
                for row in U_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                for row in V_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                for row in W_list:
                    for idx in range(len(row)):
                        if abs(row[idx] * 2 - round(row[idx] * 2)) < 1e-6:
                            row[idx] = round(row[idx] * 2) / 2.0
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try ranks from low to high
    for rank in range(15, 30):
        if time.time() > _DEADLINE - 5.0:
            break
        # Try multiple random seeds for each rank
        for seed in range(10):
            if time.time() > _DEADLINE - 5.0:
                break
            result = _try_rank(n, m, p, rank, max_iter=500, seed=seed)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
