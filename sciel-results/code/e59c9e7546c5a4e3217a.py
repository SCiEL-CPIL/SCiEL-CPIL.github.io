"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import random
import itertools

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 84
    
    # Start with the trivial decomposition as baseline
    best_U = [[0] * trivial_rank for _ in range(n * m)]
    best_V = [[0] * trivial_rank for _ in range(m * p)]
    best_W = [[0] * trivial_rank for _ in range(p * n)]
    best_rank = trivial_rank
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify if U, V, W form a valid decomposition of M_{n,m,p}"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries: should sum to 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_arr[i*m + j] * V_arr[j*p + k] * W_arr[k*n + i])
                    if abs(s - 1) > 1e-10:
                        return False
        
        # Check invalid entries: should sum to 0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    
                    is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                    
                    if not is_valid:
                        s = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if abs(s) > 1e-10:
                            return False
        return True
    
    def construct_m344_rank18():
        """Construct rank-18 decomposition for M_{3,4,4}"""
        n1, m1, p1 = 3, 4, 4
        rank1 = 18
        
        U1 = [[0]*rank1 for _ in range(n1*m1)]
        V1 = [[0]*rank1 for _ in range(m1*p1)]
        W1 = [[0]*rank1 for _ in range(p1*n1)]
        
        # Use a structured construction based on tensor rank properties
        # For M_{3,4,4}, we use a pattern that exploits the structure
        
        # Strategy: Use 12 "diagonal" terms plus 6 correction terms
        # The 12 diagonal terms handle the main contributions
        # The 6 correction terms handle the off-diagonal cancellations
        
        # Define the pattern for rank-18 M_{3,4,4}
        # This is based on known constructions
        
        # First 12 terms: one for each (i,j) pair, handling k=0
        for i in range(n1):
            for j in range(m1):
                r = i * m1 + j
                U1[i*m1 + j][r] = 1
                V1[j*p1 + 0][r] = 1
                W1[0*n1 + i][r] = 1
        
        # Next 6 terms: handle additional k values with corrections
        # These terms cancel the unwanted contributions from the first 12 terms
        
        # For k=1,2,3, we need to add terms that contribute to (i,j,k) but cancel for other k
        # Use a systematic pattern
        
        correction_patterns = [
            # (i, j, k, u_val, v_val, w_val)
            (0, 0, 1, 1, 1, 1),
            (1, 1, 1, 1, 1, 1),
            (2, 2, 1, 1, 1, 1),
            (0, 0, 2, 1, 1, 1),
            (1, 1, 2, 1, 1, 1),
            (2, 2, 2, 1, 1, 1),
        ]
        
        for idx, (i, j, k, u_val, v_val, w_val) in enumerate(correction_patterns):
            r = 12 + idx
            U1[i*m1 + j][r] = u_val
            V1[j*p1 + k][r] = v_val
            W1[k*n1 + i][r] = w_val
        
        # Verify this works for M_{3,4,4}
        if verify_decomposition(U1, V1, W1, rank1, n1, m1, p1):
            return U1, V1, W1, rank1
        
        # If the simple pattern doesn't work, try a different approach
        # Use a more sophisticated construction
        
        # Alternative: Use the fact that we can express M_{3,4,4} using
        # bilinear forms with specific coefficients
        
        # Let's try a different pattern: use terms that span the needed space
        # Reset
        U1 = [[0]*rank1 for _ in range(n1*m1)]
        V1 = [[0]*rank1 for _ in range(m1*p1)]
        W1 = [[0]*rank1 for _ in range(p1*n1)]
        
        # Use a construction based on partitioning
        # Split the 4 output columns into two groups of 2
        # Use rank-8 for M_{3,4,2} (trivial is 24, but we can do better)
        
        # Actually, let's use a direct search for M_{3,4,4} with rank 18
        # using structured random search with small integer values
        
        search_values = [0, 1, -1, 2, -2, Fraction(1,2), Fraction(-1,2)]
        
        for attempt in range(500):
            if time.time() > deadline - 30:
                break
            
            U1 = [[random.choice(search_values) for _ in range(rank1)] for _ in range(n1*m1)]
            V1 = [[random.choice(search_values) for _ in range(rank1)] for _ in range(m1*p1)]
            W1 = [[random.choice(search_values) for _ in range(rank1)] for _ in range(p1*n1)]
            
            if verify_decomposition(U1, V1, W1, rank1, n1, m1, p1):
                return U1, V1, W1, rank1
        
        return None
    
    def construct_m343_decomposition():
        """Construct decomposition for M_{3,4,3}"""
        n1, m1, p1 = 3, 4, 3
        # Try rank 12 first (trivial for the diagonal)
        # Then try rank 15, 18, etc.
        
        for target_rank in [12, 15, 18, 21, 24]:
            if time.time() > deadline - 25:
                break
            
            U1 = [[0]*target_rank for _ in range(n1*m1)]
            V1 = [[0]*target_rank for _ in range(m1*p1)]
            W1 = [[0]*target_rank for _ in range(p1*n1)]
            
            # Initialize with diagonal terms
            for i in range(n1):
                for j in range(m1):
                    for k in range(p1):
                        if i*m1 + j + k < target_rank:
                            U1[i*m1 + j][i*m1 + j + k] = 1
                            V1[j*p1 + k][i*m1 + j + k] = 1
                            W1[k*n1 + i][i*m1 + j + k] = 1
            
            if verify_decomposition(U1, V1, W1, target_rank, n1, m1, p1):
                return U1, V1, W1, target_rank
            
            # Try random search
            search_values = [0, 1, -1]
            for attempt in range(200):
                if time.time() > deadline - 20:
                    break
                
                U1 = [[random.choice(search_values) for _ in range(target_rank)] for _ in range(n1*m1)]
                V1 = [[random.choice(search_values) for _ in range(target_rank)] for _ in range(m1*p1)]
                W1 = [[random.choice(search_values) for _ in range(target_rank)] for _ in range(p1*n1)]
                
                if verify_decomposition(U1, V1, W1, target_rank, n1, m1, p1):
                    return U1, V1, W1, target_rank
        
        # Return trivial decomposition
        rank1 = n1 * m1 * p1
        U1 = [[0]*rank1 for _ in range(n1*m1)]
        V1 = [[0]*rank1 for _ in range(m1*p1)]
        W1 = [[0]*rank1 for _ in range(p1*n1)]
        
        col = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U1[i*m1 + j][col] = 1
                    V1[j*p1 + k][col] = 1
                    W1[k*n1 + i][col] = 1
                    col += 1
        
        return U1, V1, W1, rank1
    
    # Strategy: Split p=7 into p1=4 and p2=3
    # Use rank-18 for M_{3,4,4} and find best for M_{3,4,3}
    
    # First, get M_{3,4,4} decomposition
    m344_result = construct_m344_rank18()
    
    if m344_result is not None:
        U1, V1, W1, rank1 = m344_result
    else:
        # Fallback: use trivial for M_{3,4,4}
        n1, m1, p1 = 3, 4, 4
        rank1 = n1 * m1 * p1
        U1 = [[0]*rank1 for _ in range(n1*m1)]
        V1 = [[0]*rank1 for _ in range(m1*p1)]
        W1 = [[0]*rank1 for _ in range(p1*n1)]
        
        col = 0
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U1[i*m1 + j][col] = 1
                    V1[j*p1 + k][col] = 1
                    W1[k*n1 + i][col] = 1
                    col += 1
    
    # Get M_{3,4,3} decomposition
    U2, V2, W2, rank2 = construct_m343_decomposition()
    
    # Combine the two decompositions
    total_rank = rank1 + rank2
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Copy first part (for k=0,1,2,3)
    for i in range(n):
        for j in range(m):
            for r in range(rank1):
                U[i*m + j][r] = U1[i*m + j][r]
    
    for j in range(m):
        for k in range(4):  # Only first 4 k values
            for r in range(rank1):
                V[j*p + k][r] = V1[j*4 + k][r]
    
    for k in range(4):  # Only first 4 k values
        for i in range(n):
            for r in range(rank1):
                W[k*n + i][r] = W1[k*n + i][r]
    
    # Copy second part (for k=4,5,6)
    offset = rank1
    for i in range(n):
        for j in range(m):
            for r in range(rank2):
                U[i*m + j][offset + r] = U2[i*m + j][r]
    
    for j in range(m):
        for k in range(3):  # k=4,5,6 correspond to k=0,1,2 in the sub-problem
            for r in range(rank2):
                V[j*p + (4 + k)][offset + r] = V2[j*3 + k][r]
    
    for k in range(3):  # k=4,5,6 correspond to k=0,1,2 in the sub-problem
        for i in range(n):
            for r in range(rank2):
                W[(4 + k)*n + i][offset + r] = W2[k*n + i][r]
    
    # Verify the combined decomposition
    if verify_decomposition(U, V, W, total_rank, n, m, p):
        best_U = U
        best_V = V
        best_W = W
        best_rank = total_rank
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If the split approach doesn't work, try direct search for the full problem
    # with smaller ranks
    
    # Try rank 30, 33, 36, etc. with structured search
    for target_rank in [30, 33, 36, 40, 45, 50, 60, 70, 84]:
        if time.time() > deadline - 10:
            break
        
        if target_rank >= best_rank:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U = [[0]*target_rank for _ in range(n*m)]
            V = [[0]*target_rank for _ in range(m*p)]
            W = [[0]*target_rank for _ in range(p*n)]
            
            # Initialize with a structured pattern
            # Use the first 12 terms for the "main" contributions
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        if idx < target_rank:
                            U[i*m + j][idx] = 1
                            V[j*p + k][idx] = 1
                            W[k*n + i][idx] = 1
            
            # Try to optimize by adjusting some entries
            search_values = [0, 1, -1]
            for _ in range(100):
                # Randomly perturb some entries
                row_idx = random.randint(0, n*m-1)
                col_idx = random.randint(0, target_rank-1)
                U[row_idx][col_idx] = random.choice(search_values)
                
                row_idx = random.randint(0, m*p-1)
                col_idx = random.randint(0, target_rank-1)
                V[row_idx][col_idx] = random.choice(search_values)
                
                row_idx = random.randint(0, p*n-1)
                col_idx = random.randint(0, target_rank-1)
                W[row_idx][col_idx] = random.choice(search_values)
            
            if verify_decomposition(U, V, W, target_rank, n, m, p):
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break
        
        if best_rank < trivial_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
