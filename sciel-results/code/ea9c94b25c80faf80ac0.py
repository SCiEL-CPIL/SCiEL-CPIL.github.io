"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form an exact tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all valid triples: T[i*m+j, j*p+k, k*n+i] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that invalid triples give 0
    # An invalid triple is one where the indices don't align properly
    # For a = i*m+j, b = j2*p+k, c = k*n+i' where j2 != j or i' != i
    # We check a sample of invalid triples
    np.random.seed(42)
    for _ in range(500):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        
        # Create an invalid triple by changing j or i
        if np.random.rand() < 0.5:
            # Change j
            j2 = (j + 1) % m
            a = i * m + j
            b = j2 * p + k
            c = k * n + i
        else:
            # Change i
            i2 = (i + 1) % n
            a = i * m + j
            b = j * p + k
            c = k * n + i2
        
        val = np.dot(U[a], V[b] * W[c])
        if abs(val) > tol:
            return False
    
    return True

def _try_random_search(rank, n, m, p, max_attempts=100, deadline=None):
    """Try to find a rank-r decomposition using random search."""
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        np.random.seed(attempt * 7919 + rank * 31 + 12345)
        
        # Random integer matrices with entries in {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(n*m, rank))
        V = np.random.randint(-1, 2, size=(m*p, rank))
        W = np.random.randint(-1, 2, size=(p*n, rank))
        
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_sparse_search(rank, n, m, p, max_attempts=50, deadline=None):
    """Try to find a rank-r decomposition using sparse random patterns."""
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        np.random.seed(attempt * 104729 + rank * 7 + 999)
        
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
        
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook()
    best_rank = n * m * p  # 54
    
    # Try to find a lower rank decomposition.
    # Strategy: attempt structured searches for ranks below schoolbook.
    # For each candidate rank, try random initializations and verify exactness.
    
    # Lower bound for matrix multiplication rank is at least max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But known results suggest higher. Let's try from 20 upward to 53.
    
    candidate_ranks = list(range(20, 54))
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue
        
        # Try random search first
        result = _try_random_search(rank, n, m, p, max_attempts=30, deadline=deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            continue
        
        # Try sparse search
        result = _try_sparse_search(rank, n, m, p, max_attempts=20, deadline=deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            continue
    
    return best_result
# EVOLVE-BLOCK-END
