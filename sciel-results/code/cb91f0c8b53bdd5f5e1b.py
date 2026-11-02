import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Try to find lower rank solutions
    # Target ranks: 24, 25, 26, ... up to 35
    for target_rank in range(24, rank_schoolbook):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            break
            
        found = False
        max_iters = 5000  # Adjust based on time
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Random U, V with entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n * m, target_rank))
            V = np.random.randint(-1, 2, size=(m * p, target_rank))
            
            # Construct linear system for W
            # Equations: for each i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Unknowns: W[k*n+i, r] for all k, i, r
            # Number of unknowns: p * n * target_rank
            # Number of equations: n * m * p
            
            num_unknowns = p * n * target_rank
            num_equations = n * m * p
            
            A = np.zeros((num_equations, num_unknowns))
            b = np.ones(num_equations)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation for (i, j, k)
                        # Involves W[k*n+i, r] for all r
                        # Coefficient for W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
                        for r in range(target_rank):
                            col_idx = k * n * target_rank + i * target_rank + r
                            A[eq_idx, col_idx] = U[i * m + j, r] * V[j * p + k, r]
                        eq_idx += 1
                        
            # Solve least squares
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            W = W_flat.reshape(p * n, target_rank)
            
            # Snap W to nearest half-integer
            W_snapped = np.round(W * 2) / 2.0
            
            # Verify solution
            # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W_snapped[k*n+i, r] == 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i * m + j, r] * V[j * p + k, r] * W_snapped[k * n + i, r]
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                    
            if valid:
                # Found a solution
                # Convert to lists of integers/half-integers
                U_list = [[int(x) for x in row] for row in U]
                V_list = [[int(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W_snapped]
                
                best_rank = target_rank
                best_U = U_list
                best_V = V_list
                best_W = W_list
                found = True
                break
        
        if found:
            # If we found a solution for this rank, we can stop searching for higher ranks 
            # because we want minimal rank.
            # But we should check if an even lower rank exists? 
            # We are iterating 24, 25... so 24 is the first.
            # If we found 24, that's likely the minimum.
            # We can break the outer loop.
            break

    return {"U": best_U, "V": best_V, "W": best_W}
