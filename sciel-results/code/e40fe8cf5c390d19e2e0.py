"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
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

def _verify(U, V, W):
    """
    Verify if the decomposition exactly multiplies matrices.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    Convention: W index is k*n + i.
    T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
    We want T[i, j, k] = 1 for all i, j, k.
    """
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != r or len(V[0]) != r or len(W[0]) != r:
        return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U_np[i * m + j]
                v = V_np[j * p + k]
                w = W_np[k * n + i]
                s = np.dot(u, v * w)
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def _try_random_rank(target_rank, timeout=5):
    """
    Attempt to find a random rank decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    for _ in range(1000):
        if time.time() - start_time > timeout:
            break
        U = np.random.randint(-2, 3, size=(n * m, target_rank))
        V = np.random.randint(-2, 3, size=(m * p, target_rank))
        W = np.random.randint(-2, 3, size=(p * n, target_rank))
        
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return None

def _structured_decomposition():
    """
    Try to find a structured low-rank decomposition by exploiting
    the symmetry between the two dimension-5 indices (m and p).
    
    The tensor T[i,j,k] = 1 for all i,j,k can be decomposed as:
    T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    We'll try a block-structured approach where we group indices.
    """
    n, m, p = DIMENSIONS
    
    # Try rank = m*p = 25 first (theoretical lower bound)
    # Then try ranks 26, 27, ... up to some limit
    
    for rank in range(25, 50):
        # Initialize with a structured pattern
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        
        # Try a simple structured initialization
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Fill in a structured pattern
        for c in range(rank):
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            
            U[i * m + j, c] = 1
            V[j * p + k, c] = 1
            W[k * n + i, c] = 1
        
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try with random perturbations
        for trial in range(100):
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for c in range(rank):
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                
                # Add some structure with random signs
                U[i * m + j, c] = random.choice([-1, 1])
                V[j * p + k, c] = random.choice([-1, 1])
                W[k * n + i, c] = random.choice([-1, 1])
            
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _systematic_search(target_rank, timeout=10):
    """
    More systematic search using a grid of possible values.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try small integer values
    values = [-2, -1, 0, 1, 2]
    
    for _ in range(5000):
        if time.time() - start_time > timeout:
            break
        
        U = np.random.choice(values, size=(n * m, target_rank))
        V = np.random.choice(values, size=(m * p, target_rank))
        W = np.random.choice(values, size=(p * n, target_rank))
        
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    deadline = time.time() + 100  # 20s margin for 120s limit
    
    # 1. Try schoolbook (guaranteed)
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # 2. Try structured decomposition
    if time.time() < deadline - 30:
        res = _structured_decomposition()
        if res:
            return res
    
    # 3. Try systematic search for lower ranks
    for r in range(74, 25, -1):
        if time.time() > deadline - 10:
            break
        
        remaining_time = deadline - time.time() - 10
        timeout = min(5, remaining_time / 5)
        
        res = _systematic_search(r, timeout=timeout)
        if res:
            return res
        
        # Also try random search
        timeout2 = min(3, remaining_time / 5)
        res = _try_random_rank(r, timeout=timeout2)
        if res:
            return res
            
    return best
# EVOLVE-BLOCK-END
