import time
import random
import numpy as np

def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook as fallback
    def make_schoolbook():
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
    
    best_result = make_schoolbook()
    best_rank = n * m * p  # 54
    
    # Try to find lower rank decompositions
    # Strategy: Use block structure - split the 6 columns into groups
    # and try to find efficient decompositions for each block
    
    # First, try rank 27: treat as two 3x3x3 blocks
    # If we can do 3x3x3 in rank 7 (Strassen), then 2 blocks = 14
    # But we need to verify if the blocks can be independent
    
    # Try various rank targets
    for target_rank in [21, 24, 27, 30, 33, 36, 40, 45, 50]:
        if time.time() > deadline:
            break
        
        # Use randomized search with structured initialization
        # Initialize with sparse random matrices
        random.seed(42)
        np.random.seed(42)
        
        # Try multiple random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Create random sparse initialization
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Randomly assign non-zero entries
            for r in range(target_rank):
                # Each rank-1 term picks random indices
                i = random.randint(0, n * m - 1)
                j = random.randint(0, m * p - 1)
                k = random.randint(0, p * n - 1)
                
                # Random values from {-1, 0, 1}
                U[i, r] = random.choice([-1, 0, 1])
                V[j, r] = random.choice([-1, 0, 1])
                W[k, r] = random.choice([-1, 0, 1])
            
            # Check if this is a valid decomposition
            # The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # should equal 1 if (i,j,k) corresponds to a valid multiplication term
            
            # Build the tensor from U, V, W
            U_np = np.array(U)
            V_np = np.array(V)
            W_np = np.array(W)
            
            # Reshape for tensor computation
            # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
            # We need T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
            # U index: i*m + j
            # V index: j*p + k  
            # W index: k*n + i
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1.0
                        # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        computed = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                        if abs(computed - expected) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list format
                best_rank = target_rank
                best_result = {
                    "U": U_np.tolist(),
                    "V": V_np.tolist(),
                    "W": W_np.tolist()
                }
                break
        
        if best_rank < n * m * p:
            break
    
    return best_result
