import time
import random
import math

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = 17
    
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    rows_u = n * m
    rows_v = m * p
    rows_w = p * n
    
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
                
    num_constraints = len(constraints)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    def calculate_error(U, V, W):
        err = 0.0
        for u_idx, v_idx, w_idx in constraints:
            s = 0.0
            for r in range(target_rank):
                s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            err += (s - 1.0) ** 2
        return err

    max_restarts = 10
    restart_time_budget = 10.0
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        U = [[random.choice(values) for _ in range(target_rank)] for _ in range(rows_u)]
        V = [[random.choice(values) for _ in range(target_rank)] for _ in range(rows_v)]
        W = [[random.choice(values) for _ in range(target_rank)] for _ in range(rows_w)]
        
        current_error = calculate_error(U, V, W)
        if current_error < best_error:
            best_error = current_error
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
            if best_error < 1e-9:
                break

        T = 1.0
        cooling_rate = 0.999
        last_improvement = time.time()
        
        while time.time() < deadline and time.time() - last_improvement < restart_time_budget:
            mat_choice = random.randint(0, 2)
            if mat_choice == 0:
                mat = U
                rows = rows_u
            elif mat_choice == 1:
                mat = V
                rows = rows_v
            else:
                mat = W
                rows = rows_w
                
            row_idx = random.randint(0, rows - 1)
            col_idx = random.randint(0, target_rank - 1)
            
            old_val = mat[row_idx][col_idx]
            new_val = random.choice([v for v in values if v != old_val])
            
            mat[row_idx][col_idx] = new_val
            new_error = calculate_error(U, V, W)
            
            if new_error < current_error:
                current_error = new_error
                last_improvement = time.time()
                if current_error < best_error:
                    best_error = current_error
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
                    if best_error < 1e-9:
                        break
            else:
                if random.random() < math.exp(-(new_error - current_error) / T):
                    current_error = new_error
                else:
                    mat[row_idx][col_idx] = old_val
                    
            T *= cooling_rate
            
        if best_error < 1e-9:
            break

    if best_error < 1e-9:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
