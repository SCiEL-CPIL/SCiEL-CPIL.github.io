import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 42
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Trivial decomposition
    U_triv = [[0] * trivial_rank for _ in range(n * m)]
    V_triv = [[0] * trivial_rank for _ in range(m * p)]
    W_triv = [[0] * trivial_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j][col] = 1
                V_triv[j * p + k][col] = 1
                W_triv[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}
    best_rank = trivial_rank
    
    # Strategy: Try to find lower rank decompositions
    # Key insight: For matrix multiplication, we can use the structure to build
    # Winograd-like schemes. For 2x3x7, let's try various approaches.
    
    # Approach 1: Try to use the fact that 7 = 3+3+1 and handle each block
    # Approach 2: Random search with structured initialization
    # Approach 3: Use known results - for small matrices, the minimum rank is often
    # close to the trivial rank but can be reduced
    
    # Let's try a more systematic approach:
    # For each rank from 41 down to 30, try to find a valid decomposition
    # using local search from a good initialization
    
    def try_rank(target_rank, max_attempts=50):
        """Try to find a decomposition of the given rank."""
        nonlocal best_rank, best_result
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Initialize with a pattern that might work
            # Use the trivial solution as a starting point and merge terms
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Fill with random values from a small set
            # Use values that are likely to produce exact results
            # Try integers and half-integers
            val_set = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
            
            # Random initialization
            for i in range(n * m):
                for t in range(target_rank):
                    U[i][t] = random.choice(val_set)
            for j in range(m * p):
                for t in range(target_rank):
                    V[j][t] = random.choice(val_set)
            for k in range(p * n):
                for t in range(target_rank):
                    W[k][t] = random.choice(val_set)
            
            # Check if valid
            if check_decomposition(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
            
            # If not valid, try local search
            # Try to fix one term at a time
            improved = False
            for _ in range(100):  # Local search iterations
                if time.time() > deadline - 3:
                    break
                    
                # Pick a random term to modify
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row = random.randint(0, n*m-1)
                    col_t = random.randint(0, target_rank-1)
                    old_val = U[row][col_t]
                elif mat == 'V':
                    row = random.randint(0, m*p-1)
                    col_t = random.randint(0, target_rank-1)
                    old_val = V[row][col_t]
                else:
                    row = random.randint(0, p*n-1)
                    col_t = random.randint(0, target_rank-1)
                    old_val = W[row][col_t]
                
                # Try a new value
                new_val = random.choice(val_set)
                if new_val == old_val:
                    continue
                
                # Apply change
                if mat == 'U':
                    U[row][col_t] = new_val
                elif mat == 'V':
                    V[row][col_t] = new_val
                else:
                    W[row][col_t] = new_val
                
                # Check if valid
                if check_decomposition(U, V, W, target_rank):
                    return {"U": U, "V": V, "W": W}
                
                # Revert if not better (we're just checking validity, not optimizing)
                if mat == 'U':
                    U[row][col_t] = old_val
                elif mat == 'V':
                    V[row][col_t] = old_val
                else:
                    W[row][col_t] = old_val
        
        return None
    
    # Try ranks from 41 down to 30
    for target_rank in range(41, 29, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(target_rank, max_attempts=30)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break  # Found a valid lower rank, stop
    
    return best_result
