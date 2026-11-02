import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor is all 1s
    # T[i, j, k] = 1
    
    # Schoolbook rank
    rank_school = n * m * p
    
    # Try to find a lower rank decomposition
    # We use a local search to minimize the error
    # Error = sum (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
    
    # We restrict entries to integers or half-integers (denominator 2)
    # to ensure the output format is valid.
    
    def check_solution(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check if decomposition is exact
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def random_init(rank):
        U = [[random.choice([-2, -1, 0, 1, 2]) * random.choice([1, 0.5]) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.choice([-2, -1, 0, 1, 2]) * random.choice([1, 0.5]) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.choice([-2, -1, 0, 1, 2]) * random.choice([1, 0.5]) for _ in range(rank)] for _ in range(p*n)]
        return U, V, W

    def error(U, V, W, rank):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    err += (s - 1.0) ** 2
        return err

    start_time = time.time()
    deadline = start_time + 100 # 100s limit
    
    # Try ranks from 50 down to 40
    # If we find one, return it.
    # If not, return schoolbook.
    
    best_rank = rank_school
    best_solution = None
    
    # Schoolbook solution as fallback
    U_fb = [[0] * rank_school for _ in range(n*m)]
    V_fb = [[0] * rank_school for _ in range(m*p)]
    W_fb = [[0] * rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m + j][col] = 1
                V_fb[j*p + k][col] = 1
                W_fb[k*n + i][col] = 1
                col += 1
                
    # Try to find better
    for R in range(50, 39, -1):
        if time.time() > deadline:
            break
        # Try multiple random restarts for this R
        for restart in range(5):
            if time.time() > deadline:
                break
            U, V, W = random_init(R)
            # Local search
            # We will do a simple hill climbing: pick a random variable, change it to a nearby value, if error decreases, keep it.
            # To speed up, we only check a subset of errors? No, must be exact.
            # But we can compute full error.
            
            current_err = error(U, V, W, R)
            # If error is 0, we found it
            if current_err < 1e-9:
                best_rank = R
                best_solution = (U, V, W)
                break
            
            # Simple local search steps
            # We limit steps to avoid hanging
            for step in range(500):
                if time.time() > deadline:
                    break
                # Pick a random matrix and entry
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = random.randint(0, n*m - 1)
                    r_idx = random.randint(0, R - 1)
                    old_val = U[idx][r_idx]
                    # Try changing to old_val + delta
                    delta = random.choice([-0.5, 0.5, -1, 1])
                    new_val = old_val + delta
                    U[idx][r_idx] = new_val
                    new_err = error(U, V, W, R)
                    if new_err >= current_err:
                        U[idx][r_idx] = old_val
                    else:
                        current_err = new_err
                        if current_err < 1e-9:
                            best_rank = R
                            best_solution = (U, V, W)
                            break
                elif mat == 'V':
                    idx = random.randint(0, m*p - 1)
                    r_idx = random.randint(0, R - 1)
                    old_val = V[idx][r_idx]
                    delta = random.choice([-0.5, 0.5, -1, 1])
                    new_val = old_val + delta
                    V[idx][r_idx] = new_val
                    new_err = error(U, V, W, R)
                    if new_err >= current_err:
                        V[idx][r_idx] = old_val
                    else:
                        current_err = new_err
                        if current_err < 1e-9:
                            best_rank = R
                            best_solution = (U, V, W)
                            break
                else:
                    idx = random.randint(0, p*n - 1)
                    r_idx = random.randint(0, R - 1)
                    old_val = W[idx][r_idx]
                    delta = random.choice([-0.5, 0.5, -1, 1])
                    new_val = old_val + delta
                    W[idx][r_idx] = new_val
                    new_err = error(U, V, W, R)
                    if new_err >= current_err:
                        W[idx][r_idx] = old_val
                    else:
                        current_err = new_err
                        if current_err < 1e-9:
                            best_rank = R
                            best_solution = (U, V, W)
                            break
            if best_solution is not None:
                break
        if best_solution is not None:
            break

    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
