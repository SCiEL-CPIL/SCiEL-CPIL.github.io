import numpy as np
import random
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 28
    pn = p * n  # 14
    
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Try to find a decomposition with rank < 56
    # Start with schoolbook and try to reduce
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Check if a decomposition is valid
    def is_valid(U, V, W):
        # For all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    # Try to find a lower rank decomposition using local search
    # We'll try ranks from 50 down to 20, and for each try random initialization + refinement
    
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook()
    
    # Try a few different strategies
    strategies = [
        (50, "random"),
        (45, "random"),
        (40, "random"),
        (35, "random"),
        (30, "random"),
    ]
    
    for target_rank, method in strategies:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
            
        # Try random initialization with half-integers
        max_iters = 500
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random half-integers in {-1, -0.5, 0, 0.5, 1}
            vals = [-1, -0.5, 0, 0.5, 1]
            U = np.random.choice(vals, size=(nm, target_rank))
            V = np.random.choice(vals, size=(mp, target_rank))
            W = np.random.choice(vals, size=(pn, target_rank))
            
            # Ensure at least some non-zero entries
            if np.all(U == 0) or np.all(V == 0) or np.all(W == 0):
                continue
            
            # Local search: try to fix violations
            improved = False
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                    
                # Compute violations
                violations = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if abs(s - 1.0) > 1e-6:
                                violations.append((i, j, k, s))
                
                if not violations:
                    improved = True
                    break
                
                # Pick a random violation and try to fix it
                i, j, k, s = random.choice(violations)
                # Try modifying a random entry
                c = random.randint(0, target_rank - 1)
                # Try different values
                for new_val in [-1, -0.5, 0, 0.5, 1]:
                    # Try modifying U
                    old = U[i*m+j, c]
                    U[i*m+j, c] = new_val
                    if is_valid(U, V, W):
                        improved = True
                        break
                    U[i*m+j, c] = old
                    
                    # Try modifying V
                    old = V[j*p+k, c]
                    V[j*p+k, c] = new_val
                    if is_valid(U, V, W):
                        improved = True
                        break
                    V[j*p+k, c] = old
                    
                    # Try modifying W
                    old = W[k*n+i, c]
                    W[k*n+i, c] = new_val
                    if is_valid(U, V, W):
                        improved = True
                        break
                    W[k*n+i, c] = old
                
                if improved:
                    break
            
            if improved:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
    
    # Convert to list of lists with integer/half-integer entries
    def to_list(mat):
        return [[float(mat[i][j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
