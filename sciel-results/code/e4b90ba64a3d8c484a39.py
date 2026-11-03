import time
import sys
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin before 150s kill
    
    # Schoolbook decomposition: rank = n*m*p = 105
    rank_school = n * m * p
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][t] = 1
                    V[j * p + k][t] = 1
                    W[k * n + i][t] = 1
                    t += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the target tensor T[i,j,k] = 1 (matrix multiplication tensor)
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Equivalently, for each r: u_r (vector in R^{n*m}), v_r (vector in R^{m*p}), w_r (vector in R^{p*n})
    # T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    
    # Try to find lower rank decompositions
    best_result = make_schoolbook()
    best_rank = rank_school
    
    # Random search for lower rank
    # For a given rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all (i,j,k): sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    
    # Strategy: Randomly initialize and use iterative refinement
    # Or: try structured low-rank approximations
    
    # First, try some known lower bounds / structured approaches
    # The tensor rank of matrix multiplication M_{n,m,p} is at most n*m*p (schoolbook)
    # and at least max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    
    # Try rank values from a reasonable lower bound up to schoolbook
    # Use random initialization with verification
    
    for rank in range(35, min(rank_school, 60)):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Random integer/half-integer initialization
            scale = 1.0
            U = np.random.randint(-2, 3, size=(n*m, rank)) * 0.5
            V = np.random.randint(-2, 3, size=(m*p, rank)) * 0.5
            W = np.random.randint(-2, 3, size=(p*n, rank)) * 0.5
            
            # Verify
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for t in range(rank))
                        if abs(val - 1.0) > 1e-6:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            if success:
                # Convert to lists
                result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_result = result
                best_rank = rank
                break
        
        if best_rank < rank_school:
            break
    
    return best_result

# EVOLVE-BLOCK-END
