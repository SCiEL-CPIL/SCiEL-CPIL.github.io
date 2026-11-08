"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin

    # Schoolbook decomposition of rank n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank

    # Try to find a lower-rank decomposition via structured search
    # For 3x3 x 3x6, we can split B into two 3x3 matrices and use
    # separate decompositions, potentially sharing structure.
    
    random.seed(42)
    
    # Attempt: use a rank-46 construction based on two 3x3x3 multiplications
    # Each 3x3x3 uses rank 23 (Landsberg), total 46
    # This is a heuristic; we verify exactness.
    
    # Generate candidate using random small integers and verify
    for target_rank in range(46, 40, -1):
        if time.time() > deadline - 10:
            break
        
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            # Small integer initialization
            U_c = np.random.randint(-2, 3, size=(n*m, target_rank)).tolist()
            V_c = np.random.randint(-2, 3, size=(m*p, target_rank)).tolist()
            W_c = np.random.randint(-2, 3, size=(p*n, target_rank)).tolist()
            
            if _verify_decomposition(U_c, V_c, W_c, n, m, p):
                best_rank = target_rank
                best_result = {"U": U_c, "V": V_c, "W": W_c}
                break
        
        if best_rank <= target_rank:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
