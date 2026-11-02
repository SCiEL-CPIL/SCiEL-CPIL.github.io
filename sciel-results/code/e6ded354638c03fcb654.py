import numpy as np
import time
import scipy.optimize

def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    
    def get_rank_14():
        rank = 14
        U = [[0] * rank for _ in range(u_dim)]
        V = [[0] * rank for _ in range(v_dim)]
        W = [[0] * rank for _ in range(w_dim)]
        
        U_terms = [
            {(0,0): 1, (1,1): 1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,1): 1},
            {(0,0): 1, (0,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,1): 1, (1,1): -1},
        ]
        V_terms_block1 = [
            {(0,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (0,0): -1},
            {(1,1): 1},
            {(0,0): 1, (0,1): 1},
            {(1,0): 1, (1,1): 1},
        ]
        W_terms_block1 = [
            {(0,0): 1, (1,1): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (0,1): 1},
            {(1,0): 1, (0,0): -1},
            {(1,1): 1},
            {(0,0): 1},
        ]
        V_terms_block2 = [
            {(0,2): 1, (1,3): 1},
            {(0,2): 1},
            {(0,3): 1, (1,3): -1},
            {(1,2): 1, (0,2): -1},
            {(1,3): 1},
            {(0,2): 1, (0,3): 1},
            {(1,2): 1, (1,3): 1},
        ]
        W_terms_block2 = [
            {(2,0): 1, (3,1): 1},
            {(2,1): 1, (3,1): -1},
            {(3,0): 1, (3,1): 1},
            {(2,0): 1, (2,1): 1},
            {(3,0): 1, (2,0): -1},
            {(3,1): 1},
            {(2,0): 1},
        ]
        
        for r in range(7):
            col1 = r
            for (i, j), val in U_terms[r].items():
                u_idx = i * m + j
                U[u_idx][col1] += val
            for (j, k), val in V_terms_block1[r].items():
                v_idx = j * p + k
                V[v_idx][col1] += val
            for (k, i), val in W_terms_block1[r].items():
                w_idx = k * n + i
                W[w_idx][col1] += val
                
            col2 = 7 + r
            for (i, j), val in U_terms[r].items():
                u_idx = i * m + j
                U[u_idx][col2] += val
            for (j, k), val in V_terms_block2[r].items():
                v_idx = j * p + k
                V[v_idx][col2] += val
            for (k, i), val in W_terms_block2[r].items():
                w_idx = k * n + i
                W[w_idx][col2] += val
                
        return {"U": U, "V": V, "W": W}

    start_time = time.time()
    deadline = start_time + 100
    
    def compute_error(U, V, W):
        R = U.shape[1]
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    prod = U[u_idx] * V[v_idx] * W[w_idx]
                    s = np.sum(prod)
                    error += (s - 1.0) ** 2
        return error

    def try_rank_13():
        base = get_rank_14()
        U_base = np.array(base["U"], dtype=float)
        V_base = np.array(base["V"], dtype=float)
        W_base = np.array(base["W"], dtype=float)
        
        # Try removing column 13
        # And also try removing column 6 (last of block 1)
        cols_to_remove = [13, 6, 12, 5]
        
        for col_rm in cols_to_remove:
            if time.time() > deadline:
                break
            
            # Create R=13 matrix by removing col_rm
            mask = [i for i in range(14) if i != col_rm]
            U = U_base[:, mask]
            V = V_base[:, mask]
            W = W_base[:, mask]
            
            # Add noise
            U = U + np.random.normal(0, 0.5, U.shape)
            V = V + np.random.normal(0, 0.5, V.shape)
            W = W + np.random.normal(0, 0.5, W.shape)
            
            def flatten(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten(x):
                u_size = 4 * 13
                v_size = 8 * 13
                w_size = 8 * 13
                U = x[:u_size].reshape(4, 13)
                V = x[u_size:u_size+v_size].reshape(8, 13)
                W = x[u_size+v_size:].reshape(8, 13)
                return U, V, W
            
            def objective(x):
                U, V, W = unflatten(x)
                return compute_error(U, V, W)
                
            x0 = flatten(U, V, W)
            
            try:
                result = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 2000})
                U_opt, V_opt, W_opt = unflatten(result.x)
                
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                err = compute_error(U_snap, V_snap, W_snap)
                if err < 1e-9:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
            except Exception:
                pass
                
        return None

    sol = try_rank_13()
    if sol:
        return sol
    else:
        return get_rank_14()

# EVOLVE-BLOCK-END
