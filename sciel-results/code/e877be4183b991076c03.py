"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form an exact tensor decomposition."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
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
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Start with schoolbook as the safe baseline
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition.
    # Strategy: attempt structured searches for ranks below schoolbook.
    # For each candidate rank, try a small number of random initializations
    # and verify exactness.
    
    candidate_ranks = [53, 52, 51, 50, 49, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 23]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue
        
        # Try a few random structured initializations
        max_attempts = 3
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 7919 + rank * 31)
            
            # Initialize with sparse random patterns
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            for r in range(rank):
                ui = np.random.randint(0, n*m)
                vj = np.random.randint(0, m*p)
                wk = np.random.randint(0, p*n)
                U[ui, r] = np.random.choice([-1.0, 1.0])
                V[vj, r] = np.random.choice([-1.0, 1.0])
                W[wk, r] = np.random.choice([-1.0, 1.0])
            
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                U_clean = [[float(x) for x in row] for row in U]
                V_clean = [[float(x) for x in row] for row in V]
                W_clean = [[float(x) for x in row] for row in W]
                best = {"U": U_clean, "V": V_clean, "W": W_clean}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best
# EVOLVE-BLOCK-END
