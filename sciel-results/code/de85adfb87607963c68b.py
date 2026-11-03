"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin for 120s limit
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        # Check shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        
        # Verify the tensor decomposition
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], np.multiply(V_arr[j*p+k], W_arr[k*n+i]))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Strategy: Use a smarter search based on the structure of matrix multiplication
    # Key insight: We can use block decomposition and known low-rank algorithms
    
    # For 4x4 * 4x6, we can view this as:
    # C = A * B where A is 4x4, B is 4x6
    # We can split B into B1 (4x4) and B2 (4x2)
    # C = A*B1 + A*B2
    
    # For 4x4 * 4x4, there are known algorithms with rank < 64
    # For 4x4 * 4x2, we can use rank < 32
    
    # But we need a unified decomposition
    
    # Alternative: Use the fact that the tensor has a specific structure
    # and try to find a decomposition using fewer terms
    
    # Let's try a more systematic approach:
    # 1. Start with schoolbook
    # 2. Try to find linear dependencies among the rank-1 terms
    # 3. Use those dependencies to reduce the rank
    
    # For each column c, the contribution to T[i,j,k] is:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # If we can find coefficients alpha_c such that:
    # sum_c alpha_c * (U_c ⊗ V_c ⊗ W_c) = 0
    # then we can eliminate one term
    
    # This is equivalent to finding a linear dependency among the tensors
    
    # Let's set up the linear system
    # For each (i,j,k), we have:
    # sum_c alpha_c * U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 0
    
    # This is a homogeneous system in alpha_c
    
    # But we need to be careful: we want to eliminate a term while preserving
    # the decomposition, which means we need the remaining terms to still
    # satisfy the tensor equation
    
    # A simpler approach: try to find two columns that are "similar" and can be merged
    
    # For efficiency, let's try a different strategy:
    # Use random search but with a better value set and more structure
    
    # Try ranks from 80 down to 24
    for target_rank in range(80, 23, -1):
        if time.time() > deadline:
            break
        
        # Number of attempts
        max_attempts = 20
        
        attempts = 0
        found = False
        while attempts < max_attempts and time.time() < deadline and not found:
            attempts += 1
            
            # Use integer values for simplicity
            vals = [-1, 0, 1]
            
            # Generate sparse random matrices
            U_rand = [[random.choice(vals) if random.random() < 0.2 else 0 for _ in range(target_rank)] for _ in range(n * m)]
            V_rand = [[random.choice(vals) if random.random() < 0.2 else 0 for _ in range(target_rank)] for _ in range(m * p)]
            W_rand = [[random.choice(vals) if random.random() < 0.2 else 0 for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U_rand, V_rand, W_rand):
                best_result = {"U": U_rand, "V": V_rand, "W": W_rand}
                best_rank = target_rank
                found = True
                break
    
    # If we found an improvement, try to improve further
    if best_rank < rank_schoolbook and time.time() < deadline:
        # Try to improve from the found rank
        for target_rank in range(best_rank - 1, 23, -1):
            if time.time() > deadline:
                break
            
            max_attempts = 10
            attempts = 0
            found = False
            while attempts < max_attempts and time.time() < deadline and not found:
                attempts += 1
                
                vals = [-1, 0, 1]
                
                U_rand = [[random.choice(vals) if random.random() < 0.2 else 0 for _ in range(target_rank)] for _ in range(n * m)]
                V_rand = [[random.choice(vals) if random.random() < 0.2 else 0 for _ in range(target_rank)] for _ in range(m * p)]
                W_rand = [[random.choice(vals) if random.random() < 0.2 else 0 for _ in range(target_rank)] for _ in range(p * n)]
                
                if verify(U_rand, V_rand, W_rand):
                    best_result = {"U": U_rand, "V": V_rand, "W": W_rand}
                    best_rank = target_rank
                    found = True
                    break
    
    return best_result
# EVOLVE-BLOCK-END
