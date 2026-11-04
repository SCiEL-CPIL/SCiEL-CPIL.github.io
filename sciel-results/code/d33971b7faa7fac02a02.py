"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    schoolbook_rank = n * m * p  # 42
    
    # Deadline: 120s limit, keep 10s margin for safety
    deadline = time.time() + 110
    
    def verify_decomposition(U_list, V_list, W_list, rank):
        """
        Verifies if U, V, W correctly decompose the 2x3x7 multiplication tensor.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        
        Condition: sum_{c} U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] == 1 for all i,j,k
        """
        if len(U_list) != n*m or len(V_list) != m*p or len(W_list) != p*n:
            return False
        if len(U_list[0]) != rank or len(V_list[0]) != rank or len(W_list[0]) != rank:
            return False

        U_arr = np.array(U_list, dtype=np.float64)
        V_arr = np.array(V_list, dtype=np.float64)
        W_arr = np.array(W_list, dtype=np.float64)
        
        # Reshape to 3D: U[i, j, c], V[j, k, c], W[k, i, c]
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        
        # Compute sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Using einsum: 'ijc,jkc,kic->ijk'
        try:
            result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        except Exception:
            return False
            
        if not np.allclose(result, 1.0, atol=1e-6):
            return False
        return True

    def schoolbook():
        """Returns the standard schoolbook decomposition of rank n*m*p."""
        rank = schoolbook_rank
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

    # Initialize with the guaranteed valid schoolbook solution
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    # Attempt to find lower rank decompositions
    # For 2x3x7, rank < 42 is non-trivial. 
    # We will perform a limited random search for ranks 40 and 41 to see if we get lucky,
    # but strictly prioritize returning the valid 42 if time runs out or search fails.
    
    # Given the small dimensions, we can try a few iterations.
    # However, random integer/half-integer search is unlikely to find a valid decomposition
    # for rank < 42 for these specific dimensions quickly.
    # To ensure we don't crash or timeout, we keep this very lightweight.
    
    search_ranks = [41, 40]
    
    for rank in search_ranks:
        if time.time() > deadline - 5:
            break
        
        # Try a small number of random attempts
        # Using {-1, 0, 1} entries for simplicity and speed
        max_attempts = 100
        for _ in range(max_attempts):
            if time.time() > deadline - 2:
                break
                
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W, rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break # Move to next (lower) rank if successful
    
    return best_result
# EVOLVE-BLOCK-END
