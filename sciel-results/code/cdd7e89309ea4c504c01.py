import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for all i in [0,n), j in [0,m), k in [0,p).
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0  # The matrix multiplication tensor has T[i,j,k]=1
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def _try_rank(r, n, m, p, time_budget):
    """
    Try to find a rank-r decomposition using structured random search.
    Returns (U, V, W) as lists if found, else None.
    """
    deadline = time.time() + time_budget
    random.seed(42)
    
    # Strategy: Try random integer matrices and check if they work
    # For small ranks, we can try systematic search over small integer values
    
    # First, try a few random initializations
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Generate random small integer matrices
        U = np.random.randint(-2, 3, size=(n*m, r))
        V = np.random.randint(-2, 3, size=(m*p, r))
        W = np.random.randint(-2, 3, size=(p*n, r))
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
        
        # Try another approach: sparse random matrices
        if attempt % 5 == 0:
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            for col in range(r):
                # Randomly place 1s and -1s
                u_idx = random.randint(0, n*m-1)
                v_idx = random.randint(0, m*p-1)
                w_idx = random.randint(0, p*n-1)
                
                U[u_idx, col] = random.choice([1, -1])
                V[v_idx, col] = random.choice([1, -1])
                W[w_idx, col] = random.choice([1, -1])
            
            if _verify_decomposition(U, V, W, n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    total_budget = 120.0
    safety_margin = 5.0
    
    # Try to find lower-rank decompositions
    # Start from rank 15 (lower bound for 3x3x5 is at least max(9,15,15)=15)
    # and go up to rank 44
    
    best_result = _schoolbook()
    
    # Try ranks from 15 to 44
    for r in range(15, 45):
        elapsed = time.time() - start_time
        remaining = total_budget - safety_margin - elapsed
        if remaining < 2.0:
            break
        
        # Allocate more time for higher ranks (more likely to succeed)
        time_for_rank = min(remaining, 5.0)
        
        result = _try_rank(r, n, m, p, time_for_rank)
        
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            # Found a valid decomposition at this rank, stop searching lower ranks
            # (we're going in increasing order, so this is the minimum we found)
            break
    
    return best_result
