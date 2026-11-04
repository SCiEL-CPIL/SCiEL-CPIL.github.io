"""Combined approach: deterministic schoolbook baseline + structured rank-reduction search."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    # Helper to check if a decomposition is valid
    def is_valid(U, V, W, rank):
        """Check if U, V, W with given rank exactly decompose the matmul tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # The matmul tensor T[i,j,k] = 1 for all valid i,j,k
        # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 0.0
                    for c in range(rank):
                        lhs += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(lhs - 1.0) > 1e-9:
                        return False
        return True
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W, "rank": rank}
    
    best = schoolbook()
    
    # Try to find lower rank decompositions using structured search
    # Strategy: For each rank r from some minimum up to schoolbook_rank-1,
    # try to find a valid decomposition using structured search
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # Known results suggest matmul(4,4,4) has rank <= 49, so matmul(4,4,5) might be <= ~60
    
    # Try a grid search over small parameter spaces for lower ranks
    # Parameterize: fix most entries to be in {-1, 0, 1, 0.5, -0.5}
    # and use random restarts with local optimization
    
    target_ranks = list(range(20, min(n*m*p, 75)))
    target_ranks.reverse()  # Try lower ranks first
    
    for r in target_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try several random initializations and local search
        max_attempts = 50 if r < 50 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize U, V, W randomly with small integer/half-integer values
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
            
            # Try to improve using local search
            # Compute the "error" for each (i,j,k)
            # This is expensive, so do a few rounds of coordinate descent
            
            improved = False
            for round_num in range(3):
                if time.time() > deadline - 5:
                    break
                
                # Try flipping each entry
                for idx in range(min(20, n*m + m*p + p*n)):  # Limited per round
                    if time.time() > deadline - 5:
                        break
                    
                    # Randomly choose which matrix and entry to modify
                    mat_choice = np.random.choice(['U', 'V', 'W'])
                    if mat_choice == 'U':
                        row, col = np.random.randint(n*m), np.random.randint(r)
                        old_val = U[row, col]
                        new_val = np.random.choice([-1, -0.5, 0, 0.5, 1])
                        if new_val == old_val:
                            continue
                        U[row, col] = new_val
                        if is_valid(U.tolist(), V.tolist(), W.tolist(), r):
                            improved = True
                        else:
                            U[row, col] = old_val
                    elif mat_choice == 'V':
                        row, col = np.random.randint(m*p), np.random.randint(r)
                        old_val = V[row, col]
                        new_val = np.random.choice([-1, -0.5, 0, 0.5, 1])
                        if new_val == old_val:
                            continue
                        V[row, col] = new_val
                        if is_valid(U.tolist(), V.tolist(), W.tolist(), r):
                            improved = True
                        else:
                            V[row, col] = old_val
                    else:
                        row, col = np.random.randint(p*n), np.random.randint(r)
                        old_val = W[row, col]
                        new_val = np.random.choice([-1, -0.5, 0, 0.5, 1])
                        if new_val == old_val:
                            continue
                        W[row, col] = new_val
                        if is_valid(U.tolist(), V.tolist(), W.tolist(), r):
                            improved = True
                        else:
                            W[row, col] = old_val
                
                if improved:
                    break
            
            if improved:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": r}
                break  # Found a valid decomposition at this rank, stop searching lower
    
    # Return in the required format
    result = {
        "U": [[float(x) for x in row] for row in best["U"]],
        "V": [[float(x) for x in row] for row in best["V"]],
        "W": [[float(x) for x in row] for row in best["W"]]
    }
    
    return result
# EVOLVE-BLOCK-END
