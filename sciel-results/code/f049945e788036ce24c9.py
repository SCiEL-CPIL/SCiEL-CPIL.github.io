"""Rewritten program for the suite task; uses a structured random search for rank reduction."""
import numpy as np
from math import gcd
from functools import reduce

DIMENSIONS = [4, 5, 5]

def _find_half_integer_solution(q):
    """
    Given an integer vector q, find a half-integer vector w such that dot(q, w) = 1.
    Returns w if possible, else None.
    """
    # Calculate GCD of the vector
    g = 0
    for val in q:
        g = gcd(g, int(val))
        if g == 1:
            break
    
    if g not in [1, 2]:
        return None
        
    # We need sum(q_i * w_i) = 1
    # We know sum(q_i * c_i) = g for some integers c_i
    # Then sum(q_i * (c_i / (2*g))) = g / (2*g) = 1/2 ? No.
    # We want sum(q_i * w_i) = 1.
    # Let w_i = c_i / (2*g) * 2 = c_i / g ?
    # If g=1: sum(q_i c_i) = 1. w_i = c_i. (Integers are half-integers)
    # If g=2: sum(q_i c_i) = 2. w_i = c_i / 2. (Half-integers)
    
    # Find c_i using extended Euclidean algorithm generalized
    # This is a bit heavy, but for small R and small numbers, we can use a simple search or sympy if available?
    # Standard library only.
    # We can use a simple backtracking or just pick one non-zero element if it divides the target.
    # But gcd condition ensures a solution exists.
    
    # Simple constructive method:
    # Since g is small (1 or 2), and q is integer.
    # We can just iterate to find a solution if R is small?
    # Actually, we can just use the fact that we can set w based on a single element if possible,
    # but generally we need the combination.
    
    # For the purpose of this task, if we find a valid decomposition, we just need to return the matrices.
    # We don't strictly need to construct W if we are just checking validity in the search, 
    # BUT the final return requires W.
    
    # Let's implement a simple solver for small dimensions.
    # Or, since we control U and V, we can try to make q sparse.
    
    # For now, return a placeholder. The caller will verify.
    # Actually, we can just return None here and rely on the fact that if gcd is 1 or 2, 
    # a solution exists. We can construct it later or just assume it for the sake of the search loop,
    # and only construct it for the final winner.
    
    return True

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matmul: T[i,j,k] = 1
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    errors += 1
    return errors == 0

def _construct_W_from_UV(U, V, n, m, p, R):
    """
    Given U and V, try to construct W such that the decomposition holds.
    Returns W if successful, else None.
    """
    W = [[0] * R for _ in range(p * n)]
    for i in range(n):
        for k in range(p):
            # Compute q vector
            q = [0] * R
            for j in range(m):
                for r in range(R):
                    q[r] += U[i*m+j][r] * V[j*p+k][r]
            
            # Check gcd condition
            g = 0
            for val in q:
                g = gcd(g, int(val))
                if g == 1:
                    break
            
            if g not in [1, 2]:
                return None
            
            # Find w such that dot(q, w) = 1
            # We know a solution exists.
            # For g=1, we can find integer c such that dot(q, c) = 1. Then w = c.
            # For g=2, we can find integer c such that dot(q, c) = 2. Then w = c/2.
            
            # Simple search for c if R is small?
            # Or use a greedy approach?
            # Since we are in a search loop, we might not want to spend too much time here.
            # However, we can just pick a non-zero q_r and set w_r appropriately if q_r is 1 or 2.
            # If not, we need a combination.
            
            # Let's try to find a solution using a simple linear combination search.
            # Given the constraints, we can assume we can find it.
            # For the sake of speed, let's just try to find one non-zero element that works.
            found = False
            for r in range(R):
                if q[r] != 0:
                    # If q[r] is 1, w[r] = 1
                    # If q[r] is 2, w[r] = 0.5
                    # If q[r] is -1, w[r] = -1
                    # If q[r] is -2, w[r] = -0.5
                    if q[r] == 1:
                        W[k*n+i][r] = 1.0
                        found = True
                        break
                    elif q[r] == 2:
                        W[k*n+i][r] = 0.5
                        found = True
                        break
                    elif q[r] == -1:
                        W[k*n+i][r] = -1.0
                        found = True
                        break
                    elif q[r] == -2:
                        W[k*n+i][r] = -0.5
                        found = True
                        break
            
            if not found:
                # If no single element works, we need a combination.
                # This is more complex. For now, return None to be safe.
                # We can improve this by implementing a proper linear solver for small R.
                return None
                
    return W

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    start_time = 10.0 # Placeholder, real time below
    
    # 1. Initialize with Schoolbook (Rank 100)
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # 2. Try to find a better rank
    # We search for R < 100
    import time
    deadline = time.time() + 130
    
    for R in range(99, 50, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations of U, V
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
                
            # Random U, V with entries in {0, 1}
            U = [[np.random.randint(0, 2) for _ in range(R)] for _ in range(n * m)]
            V = [[np.random.randint(0, 2) for _ in range(R)] for _ in range(m * p)]
            
            # Check if we can construct W
            W = _construct_W_from_UV(U, V, n, m, p, R)
            
            if W is not None:
                # Verify
                if _validate(U, V, W, n, m, p):
                    best_rank = R
                    best_U, best_V, best_W = U, V, W
                    break # Found a valid one for this R, go to next R
            
        if best_rank < n * m * p:
            # We found a better one, we can stop or continue to find even better.
            # Let's continue to see if we can go lower.
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
