import numpy as np
import time

DIMENSIONS = [2, 3, 7]
START_TIME = time.time()
TIME_LIMIT = 110.0 # Safety margin

def construct():
    n, m, p = DIMENSIONS
    nm, mp, np_ = n*m, m*p, p*n
    target_support = set()
    target_tensor = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_support.add((u, v, w))
                target_tensor[u, v, w] = 1.0
    
    values = [-1, 0, 1]
    
    def compute_T(U, V, W):
        r = U.shape[1]
        T = np.zeros((nm, mp, np_))
        for c in range(r):
            T += np.einsum('uc,vc,wc->uvw', U[:, c], V[:, c], W[:, c])
        return T

    def solve_rank(r, values):
        best_U = None
        best_V = None
        best_W = None
        best_err = float('inf')
        
        attempts = 0
        max_attempts = 200 # Increase restarts since CD is fast
        
        while time.time() - START_TIME < TIME_LIMIT:
            attempts += 1
            if attempts > max_attempts: break
            
            U = np.random.choice(values, size=(nm, r))
            V = np.random.choice(values, size=(mp, r))
            W = np.random.choice(values, size=(np_, r))
            
            T = compute_T(U, V, W)
            err = np.sum((T - target_tensor)**2)
            
            if err < 1e-9:
                return U, V, W, err
            
            improved = True
            while improved and time.time() - START_TIME < TIME_LIMIT:
                improved = False
                
                # Update U
                for c in range(r):
                    for u in range(nm):
                        old_val = U[u, c]
                        # Precompute slice of T and Target
                        T_slice = T[u, :, :]
                        Tgt_slice = target_tensor[u, :, :]
                        Err_old_slice = np.sum((T_slice - Tgt_slice)**2)
                        
                        best_val = old_val
                        best_err_slice = Err_old_slice
                        
                        V_col = V[:, c]
                        W_col = W[:, c]
                        outer_VW = np.outer(V_col, W_col)
                        
                        for val in values:
                            if val == old_val: continue
                            delta = val - old_val
                            T_new_slice = T_slice + delta * outer_VW
                            Err_new_slice = np.sum((T_new_slice - Tgt_slice)**2)
                            if Err_new_slice < best_err_slice:
                                best_err_slice = Err_new_slice
                                best_val = val
                                
                        if best_val != old_val:
                            delta = best_val - old_val
                            U[u, c] = best_val
                            T[u, :, :] += delta * outer_VW
                            err += best_err_slice - Err_old_slice
                            improved = True
                            
                # Update V
                for c in range(r):
                    for v in range(mp):
                        old_val = V[v, c]
                        T_slice = T[:, v, :] # Shape (nm, np_)
                        Tgt_slice = target_tensor[:, v, :]
                        Err_old_slice = np.sum((T_slice - Tgt_slice)**2)
                        
                        best_val = old_val
                        best_err_slice = Err_old_slice
                        
                        U_col = U[:, c]
                        W_col = W[:, c]
                        outer_UW = np.outer(U_col, W_col) # Shape (nm, np_)
                        
                        for val in values:
                            if val == old_val: continue
                            delta = val - old_val
                            T_new_slice = T_slice + delta * outer_UW
                            Err_new_slice = np.sum((T_new_slice - Tgt_slice)**2)
                            if Err_new_slice < best_err_slice:
                                best_err_slice = Err_new_slice
                                best_val = val
                                
                        if best_val != old_val:
                            delta = best_val - old_val
                            V[v, c] = best_val
                            T[:, v, :] += delta * outer_UW
                            err += best_err_slice - Err_old_slice
                            improved = True

                # Update W
                for c in range(r):
                    for w in range(np_):
                        old_val = W[w, c]
                        T_slice = T[:, :, w] # Shape (nm, mp)
                        Tgt_slice = target_tensor[:, :, w]
                        Err_old_slice = np.sum((T_slice - Tgt_slice)**2)
                        
                        best_val = old_val
                        best_err_slice = Err_old_slice
                        
                        U_col = U[:, c]
                        V_col = V[:, c]
                        outer_UV = np.outer(U_col, V_col) # Shape (nm, mp)
                        
                        for val in values:
                            if val == old_val: continue
                            delta = val - old_val
                            T_new_slice = T_slice + delta * outer_UV
                            Err_new_slice = np.sum((T_new_slice - Tgt_slice)**2)
                            if Err_new_slice < best_err_slice:
                                best_err_slice = Err_new_slice
                                best_val = val
                                
                        if best_val != old_val:
                            delta = best_val - old_val
                            W[w, c] = best_val
                            T[:, :, w] += delta * outer_UV
                            err += best_err_slice - Err_old_slice
                            improved = True

            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if best_err < 1e-9:
                return best_U, best_V, best_W, best_err

        return best_U, best_V, best_W, best_err

    # Try increasing ranks
    for r in range(21, 43):
        # Try integers first
        res = solve_rank(r, [-1, 0, 1])
        if res:
            U, V, W, err = res
            if err < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        # If integer failed, try half-integers
        res = solve_rank(r, [-1, -0.5, 0, 0.5, 1])
        if res:
            U, V, W, err = res
            if err < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
        if time.time() - START_TIME > TIME_LIMIT * 0.8:
            break

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
