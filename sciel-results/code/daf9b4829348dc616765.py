import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
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
    
    # Target rank to try
    target_rank = 32
    start_time = time.time()
    max_time = 110 # Safety margin
    
    # Try to find a decomposition of target_rank
    # Using random search with entries in {0, 1}
    # This is a heuristic
    
    # We define the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
                
    # Random search
    np.random.seed(42)
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(1000):
        if time.time() - start_time > max_time:
            break
            
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Randomly populate U, V, W with 0/1
        # To increase chance of matching T, we can bias towards 1
        # But random is a start
        
        # Actually, a better heuristic:
        # Start with schoolbook and remove 4 columns, then try to fix?
        # Or just random.
        
        # Let's try a structured random search
        # Each column r corresponds to a term
        for r in range(target_rank):
            # Choose a random pattern
            # Simple: random 0/1
            U[:, r] = np.random.randint(0, 2, n*m)
            V[:, r] = np.random.randint(0, 2, m*p)
            W[:, r] = np.random.randint(0, 2, p*n)
            
        # Check if sum of outer products equals T
        # Sum_{r} U[:,r] V[:,r] W[:,r]
        # This is a tensor contraction
        # We can check a few random indices to speed up
        # If it fails, try next
        
        # Efficient check:
        # Flatten T
        T_flat = T.flatten()
        
        # Compute sum
        # This is O(R * nm * mp * pn) which is 32 * 9 * 12 * 12 ~ 40k ops. Fast.
        current_sum = np.zeros_like(T)
        for r in range(target_rank):
            current_sum += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
        if np.allclose(current_sum, T):
            best_U = U
            best_V = V
            best_W = W
            break
            
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_school,
            "V": V_school,
            "W": W_school
        }
# EVOLVE-BLOCK-END
