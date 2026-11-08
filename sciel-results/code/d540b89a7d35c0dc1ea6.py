"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [5, 5, 5]

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=5, m=5, p=5, tol=1e-9):
    """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,k} for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1.0 if (i == k) else 0.0
                if abs(s - expected) > tol:
                    return False
    return True

def construct():
    deadline = time.time() + 120
    best = _schoolbook()
    best_rank = 125
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition via random search
    # Start with rank 124 and go down
    for target_rank in range(124, 80, -1):
        if time.time() > deadline - 30:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            random.seed(attempt * 1000 + target_rank)
            np.random.seed(attempt * 1000 + target_rank)
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with small random values
            U[:] = np.random.randn(n*m, target_rank) * 0.1
            V[:] = np.random.randn(m*p, target_rank) * 0.1
            W[:] = np.random.randn(p*n, target_rank) * 0.1
            
            # Simple gradient-free search: try to reduce the error
            # This is very crude; likely won't find anything
            # But at least it's structured
            success = False
            for step in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            expected = 1.0 if (i == k) else 0.0
                            err += (s - expected) ** 2
                
                if err < 1e-12:
                    success = True
                    break
                
                # Nudge one random entry
                mat = random.choice([U, V, W])
                r = random.randrange(mat.shape[0])
                c = random.randrange(mat.shape[1])
                mat[r, c] += random.gauss(0, 0.01)
            
            if success:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                break
    
    # Convert best to lists if it's numpy
    if isinstance(best["U"], np.ndarray):
        best["U"] = best["U"].tolist()
        best["V"] = best["V"].tolist()
        best["W"] = best["W"].tolist()
    
    return best
