import time
import random

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # 130s out of 150s
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_solution = None
    best_rank = rank_school
    
    # Try ranks from 25 to 60
    for R in range(25, 61):
        if time.time() > deadline - 10:
            break
            
        # Random restarts for this rank
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V, W with random small integers
            U = [[random.randint(-2, 2) for _ in range(R)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(R)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(R)] for _ in range(p * n)]
            
            # Local search
            max_iters = 5000
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                    
                # Compute errors
                errors = {}
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(R):
                                s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                            if s != 1:
                                valid = False
                                errors[(i, j, k)] = s - 1
                            
                if valid:
                    return {"U": U, "V": V, "W": W}
                
                # Pick a random error
                if not errors:
                    continue
                    
                (i, j, k), err = random.choice(list(errors.items()))
                
                # Try to fix by adjusting one variable
                # We pick a random r and try to adjust U, V, or W
                r = random.randint(0, R - 1)
                
                # Try adjusting U[i*m+j][r]
                # Current sum S = sum(U*V*W). We want S' = S - old_U*V*W + new_U*V*W = 1.
                # new_U*V*W = 1 - (S - old_U*V*W) = 1 - S + old_U*V*W.
                # Let target = 1 - S + old_U*V*W.
                # We need new_U * V[j*p+k][r] * W[k*n+i][r] = target.
                # If V*W != 0, new_U = target / (V*W).
                
                v_val = V[j * p + k][r]
                w_val = W[k * n + i][r]
                u_old = U[i * m + j][r]
                
                # Check U
                if v_val != 0 and w_val != 0:
                    denom = v_val * w_val
                    target = 1 - (err + 1) + u_old * denom  # Wait, S = 1 + err.
                    # S = sum. err = S - 1. So S = err + 1.
                    # We want new_S = 1.
                    # new_S = S - u_old*denom + new_U*denom.
                    # 1 = (err + 1) - u_old*denom + new_U*denom.
                    # new_U*denom = 1 - err - 1 + u_old*denom = -err + u_old*denom.
                    target_u = -err + u_old * denom
                    if target_u % denom == 0:
                        new_u = target_u // denom
                        if new_u != u_old and -10 <= new_u <= 10:
                            # Check if this move improves things (simple heuristic: just take it if it's an integer)
                            # Actually, we should check if it reduces total error.
                            # For speed, we just accept if it's a valid integer.
                            U[i * m + j][r] = new_u
                            continue
                            
                # Check V
                u_val = U[i * m + j][r]
                if u_val != 0 and w_val != 0:
                    denom = u_val * w_val
                    v_old = V[j * p + k][r]
                    target_v = -err + v_old * denom
                    if target_v % denom == 0:
                        new_v = target_v // denom
                        if new_v != v_old and -10 <= new_v <= 10:
                            V[j * p + k][r] = new_v
                            continue
                            
                # Check W
                if u_val != 0 and v_val != 0:
                    denom = u_val * v_val
                    w_old = W[k * n + i][r]
                    target_w = -err + w_old * denom
                    if target_w % denom == 0:
                        new_w = target_w // denom
                        if new_w != w_old and -10 <= new_w <= 10:
                            W[k * n + i][r] = new_w
                            continue
                            
            # End local search
        # End restarts
    # End ranks
    
    return {"U": U_school, "V": V_school, "W": W_school}
