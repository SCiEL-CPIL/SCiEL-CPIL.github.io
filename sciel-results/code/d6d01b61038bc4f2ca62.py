"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook fallback
    best_rank = n * m * p
    U_fb = [[0] * best_rank for _ in range(nm)]
    V_fb = [[0] * best_rank for _ in range(mp)]
    W_fb = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    def is_valid(Um, Vm, Wm, r, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += Um[i*m+j][c] * Vm[j*p+k][c] * Wm[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    random.seed(42)
    
    # Try ranks from 40 to 44
    for target_rank in range(40, best_rank):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        max_attempts = 200000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15.0:
                break
            
            # Sparse random initialization
            U_r = [[random.choice([0,0,0,1,-1]) for _ in range(target_rank)] for _ in range(nm)]
            V_r = [[random.choice([0,0,0,1,-1]) for _ in range(target_rank)] for _ in range(mp)]
            W_r = [[random.choice([0,0,0,1,-1]) for _ in range(target_rank)] for _ in range(pn)]
            
            if is_valid(U_r, V_r, W_r, target_rank):
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
