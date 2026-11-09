import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _verify(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - expected) > tol:
                    return False
    return True

def _verify_3x3(U, V, W, tol=1e-10):
    """Verify rank-r decomposition for 3x3 x 3x3 -> 3x3"""
    n = 3
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * n or V.shape[0] != n * n or W.shape[0] != n * n:
        return False
    for i in range(n):
        for j in range(n):
            for k in range(n):
                expected = 1.0
                val = 0.0
                for c in range(rank):
                    val += U[i * n + j, c] * V[j * n + k, c] * W[k * n + i, c]
                if abs(val - expected) > tol:
                    return False
    return True

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

def _find_3x3_decomposition(target_rank, max_attempts=100, deadline=None):
    """Find a rank-r decomposition for 3x3 x 3x3 -> 3x3"""
    n = 3
    nn = n * n  # 9
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try with small integer/half-integer values
        vals = [0, 0.5, 1, -0.5, -1, 1.5, -1.5, 2, -2]
        U = np.random.choice(vals, size=(nn, target_rank))
        V = np.random.choice(vals, size=(nn, target_rank))
        W = np.random.choice(vals, size=(nn, target_rank))
        
        if _verify_3x3(U, V, W):
            return U, V, W
    
    return None

def _combine_3x3_to_3x6(U3, V3, W3):
    """Combine two 3x3 decompositions into a 3x3 x 3x6 decomposition"""
    # U3: (9, r), V3: (9, r), W3: (9, r) for 3x3 x 3x3
    # We need U: (9, 2r), V: (18, 2r), W: (18, 2r)
    r = U3.shape[1]
    
    U = np.zeros((9, 2 * r))
    V = np.zeros((18, 2 * r))
    W = np.zeros((18, 2 * r))
    
    # First block: k = 0,1,2 (columns 0-2 of B)
    U[:, 0:r] = U3
    V[0:9, 0:r] = V3
    W[0:9, 0:r] = W3  # k*n+i for k=0,1,2, i=0,1,2 -> indices 0-8
    
    # Second block: k = 3,4,5 (columns 3-5 of B)
    U[:, r:2*r] = U3
    V[9:18, r:2*r] = V3
    W[9:18, r:2*r] = W3  # k*n+i for k=3,4,5, i=0,1,2 -> indices 9-17
    
    return U, V, W

def construct():
    start_time = time.time()
    
    # First, return schoolbook as baseline
    best = _schoolbook()
    
    n, m, p = DIMENSIONS
    um = n * m  # 9
    vm = m * p  # 18
    wm = p * n  # 18
    
    # Try to find a 3x3 decomposition with low rank, then combine
    for target_rank_3x3 in range(19, 27):  # Try ranks 19 to 26
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        result = _find_3x3_decomposition(target_rank_3x3, max_attempts=200, 
                                           deadline=DEADLINE - SAFETY_MARGIN)
        if result is not None:
            U3, V3, W3 = result
            U, V, W = _combine_3x3_to_3x6(U3, V3, W3)
            
            # Verify the full decomposition
            if _verify(U, V, W):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # If 3x3 approach didn't work, try direct search for the full problem
    for target_rank in range(38, 53):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        for attempt in range(20):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            vals = [0, 0.5, 1, -0.5, -1]
            U = np.random.choice(vals, size=(um, target_rank))
            V = np.random.choice(vals, size=(vm, target_rank))
            W = np.random.choice(vals, size=(wm, target_rank))
            
            if _verify(U, V, W):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    return best
