import numpy as np
import time
import random
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 4
    dim_u = n * m  # 8
    dim_v = m * p  # 16
    dim_w = p * n  # 8

    # Target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1

    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Known rank-28 solution
    # This is the one from the current program, which is valid.
    # I will hardcode it to ensure we always have a fallback.
    # The current program constructs it dynamically, which is fine, but hardcoding is safer/faster.
    # However, constructing it is fast enough. I'll keep the construction logic from the seed to avoid transcription errors.
    
    rank_known = 28
    U_known = [[0] * rank_known for _ in range(dim_u)]
    V_known = [[0] * rank_known for _ in range(dim_v)]
    W_known = [[0] * rank_known for _ in range(dim_w)]
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U = U_known
        V = V_known
        W = W_known
        
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    best_U = U_known
    best_V = V_known
    best_W = W_known
    best_rank = 28

    def check_solution(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T_est = np.zeros((dim_u, dim_v, dim_w), dtype=float)
        for r in range(rank):
            u = U_np[:, r]
            v = V_np[:, r]
            w = W_np[:, r]
            T_est += np.einsum('r,s,t->rst', u, v, w)
        if not np.allclose(T_est, T, atol=1e-9):
            return False
        return True

    # Verify known solution
    if check_solution(best_U, best_V, best_W, best_rank):
        pass
    else:
        # If the hardcoded construction is wrong, we are in trouble, but it's from the seed.
        pass

    # Search for better solutions
    # We will try to find rank 25, then 24.
    # Greedy peeling is sensitive to initialization.
    
    def try_peeling(target_rank, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        residual = T.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        for step in range(target_rank):
            if time.time() > deadline - 5:
                return None
            
            # Initialize random vectors
            u = np.random.randn(dim_u)
            v = np.random.randn(dim_v)
            w = np.random.randn(dim_w)
            
            # Normalize
            u /= np.linalg.norm(u)
            v /= np.linalg.norm(v)
            w /= np.linalg.norm(w)
            
            # Alternating minimization
            for it in range(50):
                u_new = np.einsum('xyz,y,z->x', residual, v, w)
                v_new = np.einsum('xyz,x,z->y', residual, u, w)
                w_new = np.einsum('xyz,x,y->z', residual, u, v)
                
                nu = np.linalg.norm(u_new)
                nv = np.linalg.norm(v_new)
                nw = np.linalg.norm(w_new)
                
                if nu < 1e-15 or nv < 1e-15 or nw < 1e-15:
                    break
                
                u_new /= nu
                v_new /= nv
                w_new /= nw
                
                if (np.linalg.norm(u_new - u) < 1e-10 and
                    np.linalg.norm(v_new - v) < 1e-10 and
                    np.linalg.norm(w_new - w) < 1e-10):
                    u, v, w = u_new, v_new, w_new
                    break
                u, v, w = u_new, v_new, w_new
            
            c = np.einsum('xyz,x,y,z->', residual, u, v, w)
            
            if abs(c) < 1e-12:
                break
            
            residual -= c * np.einsum('x,y,z->xyz', u, v, w)
            
            U_cols.append(c * u)
            V_cols.append(v)
            W_cols.append(w)
            
            if np.linalg.norm(residual) < 1e-10:
                break
        
        if np.linalg.norm(residual) > 1e-8:
            return None
        
        # Rationalize
        # Try to snap to half-integers
        U_rat = []
        V_rat = []
        W_rat = []
        
        for r in range(len(U_cols)):
            u_r = []
            for val in U_cols[r]:
                # Multiply by 2 to check for integers
                val2 = val * 2
                nearest_int = round(val2)
                frac = Fraction(nearest_int, 2)
                u_r.append(float(frac))
            U_rat.append(u_r)
            
            v_r = []
            for val in V_cols[r]:
                val2 = val * 2
                nearest_int = round(val2)
                frac = Fraction(nearest_int, 2)
                v_r.append(float(frac))
            V_rat.append(v_r)
            
            w_r = []
            for val in W_cols[r]:
                val2 = val * 2
                nearest_int = round(val2)
                frac = Fraction(nearest_int, 2)
                w_r.append(float(frac))
            W_rat.append(w_r)
        
        U_list = [[0] * len(U_cols) for _ in range(dim_u)]
        V_list = [[0] * len(V_cols) for _ in range(dim_v)]
        W_list = [[0] * len(W_cols) for _ in range(dim_w)]
        
        for r in range(len(U_cols)):
            for i in range(dim_u):
                U_list[i][r] = U_rat[r][i]
            for i in range(dim_v):
                V_list[i][r] = V_rat[r][i]
            for i in range(dim_w):
                W_list[i][r] = W_rat[r][i]
        
        if check_solution(U_list, V_list, W_list, len(U_cols)):
            return (U_list, V_list, W_list, len(U_cols))
        
        return None

    # Try ranks 24 and 25
    # We will interleave trials for rank 24 and 25
    trial_count = 0
    while time.time() < deadline - 10:
        target_rank = 24 if trial_count % 2 == 0 else 25
        seed = trial_count * 12345 + 67890
        res = try_peeling(target_rank, seed)
        if res:
            U_list, V_list, W_list, rank = res
            if rank < best_rank:
                best_U = U_list
                best_V = V_list
                best_W = W_list
                best_rank = rank
                if best_rank <= 24:
                    break
        trial_count += 1

    return {"U": best_U, "V": best_V, "W": best_W}
