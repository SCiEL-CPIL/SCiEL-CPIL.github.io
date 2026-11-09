"""Matrix multiplication 6x6x6 rank-minimal decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Try to find a decomposition with rank < 216
    # Start with known good ranks and work down
    
    # The best known rank for 6x6 is 210
    # Let's try a structured construction
    
    # Strategy: Use a combination of Strassen-like patterns
    # For 6x6, partition into blocks and use mixed strategies
    
    # First, try a direct construction using known algorithms
    # We'll use a randomized search with structured initialization
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook
    
    # Generate schoolbook as fallback
    def schoolbook():
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
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, _ = schoolbook()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Verify function
    def verify(U, V, W, tol=1e-6):
        n, m, p = DIMENSIONS
        # Compute tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try structured constructions with various ranks
    # Use a pattern-based approach
    
    # Approach: Try ranks from 210 down to 200
    # For each rank, try random initializations and local search
    
    def try_rank(target_rank, max_iters=5000):
        n, m, p = DIMENSIONS
        if time.time() > deadline:
            return None
        
        # Random sparse initialization
        for attempt in range(20):
            if time.time() > deadline:
                return None
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with small random values
            for c in range(target_rank):
                # Choose random non-zero pattern
                u_idx = random.randint(0, n * m - 1)
                v_idx = random.randint(0, m * p - 1)
                w_idx = random.randint(0, p * n - 1)
                U[u_idx, c] = random.choice([1, -1])
                V[v_idx, c] = random.choice([1, -1])
                W[w_idx, c] = random.choice([1, -1])
            
            # Local search: try to fix errors
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Find an error
                error_found = False
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(target_rank):
                                s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            if abs(s - 1.0) > 1e-6:
                                # Try to fix by modifying one entry
                                error_found = True
                                break
                        if error_found:
                            break
                    if error_found:
                        break
                
                if not error_found:
                    return U, V, W
                
                # Random perturbation
                c = random.randint(0, target_rank - 1)
                if random.random() < 0.33:
                    idx = random.randint(0, n * m - 1)
                    U[idx, c] = random.choice([0, 1, -1])
                elif random.random() < 0.5:
                    idx = random.randint(0, m * p - 1)
                    V[idx, c] = random.choice([0, 1, -1])
                else:
                    idx = random.randint(0, p * n - 1)
                    W[idx, c] = random.choice([0, 1, -1])
        
        return None
    
    # Try decreasing ranks
    for target_rank in range(215, 200, -1):
        if time.time() > deadline:
            break
        result = try_rank(target_rank, max_iters=2000)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_rank = target_rank
                best_result = (U, V, W)
                break
    
    # Convert to lists for return
    U_list = best_result[0].tolist()
    V_list = best_result[1].tolist()
    W_list = best_result[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
