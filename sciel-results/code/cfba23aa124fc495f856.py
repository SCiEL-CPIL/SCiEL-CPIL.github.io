# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    # U: 6 x R
    # V: 9 x R
    # W: 6 x R
    
    # Pre-compute shapes for einsum
    # U: (2, 3, R)
    # V: (3, 3, R)
    # W: (3, 2, R)
    
    def calc_errors_fast(U, V, W, R):
        U_res = U.reshape(2, 3, R)
        V_res = V.reshape(3, 3, R)
        W_res = W.reshape(3, 2, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_res, V_res, W_res)
        return T - 1.0

    def local_search(U, V, W, R, time_limit, start_time):
        current_errs = calc_errors_fast(U, V, W, R)
        current_l2 = np.sum(current_errs * current_errs)
        
        num_u = 6 * R
        num_v = 9 * R
        num_w = 6 * R
        total_vars = num_u + num_v + num_w
        
        while time.time() - start_time < time_limit:
            var_idx = random.randint(0, total_vars - 1)
            
            if var_idx < num_u:
                mat = 'U'
                flat_idx = var_idx
                row = flat_idx // R
                col = flat_idx % R
                current_val = U[row, col]
            elif var_idx < num_u + num_v:
                mat = 'V'
                flat_idx = var_idx - num_u
                row = flat_idx // R
                col = flat_idx % R
                current_val = V[row, col]
            else:
                mat = 'W'
                flat_idx = var_idx - num_u - num_v
                row = flat_idx // R
                col = flat_idx % R
                current_val = W[row, col]
                
            best_new_l2 = current_l2
            best_new_val = None
            
            for delta in [0.5, -0.5]:
                new_val = current_val + delta
                
                if mat == 'U':
                    old = U[row, col]
                    U[row, col] = new_val
                elif mat == 'V':
                    old = V[row, col]
                    V[row, col] = new_val
                else:
                    old = W[row, col]
                    W[row, col] = new_val
                    
                new_errs = calc_errors_fast(U, V, W, R)
                new_l2 = np.sum(new_errs * new_errs)
                
                if new_l2 < best_new_l2:
                    best_new_l2 = new_l2
                    best_new_val = new_val
                    
                if mat == 'U':
                    U[row, col] = old
                elif mat == 'V':
                    V[row, col] = old
                else:
                    W[row, col] = old
            
            if best_new_val is not None:
                if mat == 'U':
                    U[row, col] = best_new_val
                elif mat == 'V':
                    V[row, col] = best_new_val
                else:
                    W[row, col] = best_new_val
                    
                current_l2 = best_new_l2
                current_errs = calc_errors_fast(U, V, W, R)
                
                if np.all(np.abs(current_errs) < 1e-9):
                    return True
            else:
                # Optional: Random jump to escape local minima?
                # If stuck, just continue.
                pass
                
        return False

    def try_rank(R):
        U = np.zeros((6, R))
        V = np.zeros((9, R))
        W = np.zeros((6, R))
        
        vals = [-1.0, -0.5, 0.5, 1.0]
        
        def sparse_init(mat, rows):
            for r in range(R):
                for i in range(rows):
                    if random.random() < 0.3:
                        mat[i, r] = random.choice(vals)
        
        max_restarts = 5
        time_per_restart = 0.2
        
        for _ in range(max_restarts):
            sparse_init(U, 6)
            sparse_init(V, 9)
            sparse_init(W, 6)
            
            start = time.time()
            success = local_search(U, V, W, R, time_per_restart, start)
            if success:
                return U.tolist(), V.tolist(), W.tolist()
                
        return None

    global_start = time.time()
    deadline = global_start + 110
    
    # Try ranks from 9 to 18
    for R in range(9, 19):
        if time.time() > deadline - 2:
            break
            
        result = try_rank(R)
        if result:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback
    U = [[0]*18 for _ in range(6)]
    V = [[0]*18 for _ in range(9)]
    W = [[0]*18 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1.0
                V[j*3+k][col] = 1.0
                W[k*2+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
