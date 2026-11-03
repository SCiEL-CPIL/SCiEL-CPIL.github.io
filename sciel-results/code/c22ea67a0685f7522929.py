import numpy as np
import time
import random
import scipy.optimize as so

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def find_3x3x3(rank):
        # Variables: U (9xR), V (9xR), W (9xR)
        size = 27 * rank
        
        def error(x):
            U = x[:9*rank].reshape(9, rank)
            V = x[9*rank:18*rank].reshape(9, rank)
            W = x[18*rank:27*rank].reshape(9, rank)
            
            # Compute T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # i,j,k in 0..2
            err = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*3 + j, c] * V[j*3 + k, c] * W[k*3 + i, c]
                        err += (s - 1.0)**2
            return err
        
        # Use scipy minimize
        # We need to be careful with the number of restarts to fit in time
        # Try a few restarts
        for restart in range(5):
            if time.time() > deadline:
                return None
                
            x0 = np.random.uniform(-1, 1, size)
            
            # Use Powell or Nelder-Mead
            # Powell is good for non-smooth or when gradient is not available, but here we have smooth error.
            # However, the solution space is discrete-like.
            # Let's try minimizing the error.
            try:
                res = so.minimize(error, x0, method='Powell', options={'maxiter': 1000, 'xtol': 1e-6, 'disp': False})
            except Exception:
                continue
                
            x = res.x
            U = x[:9*rank].reshape(9, rank)
            V = x[9*rank:18*rank].reshape(9, rank)
            W = x[18*rank:27*rank].reshape(9, rank)
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify rounded solution
            valid = True
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        s = 0.0
                        for c in range(rank):
                            s += U_r[i*3+j, c] * V_r[j*3+k, c] * W_r[k*3+i, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                return U_r, V_r, W_r
        
        return None

    # Try to find 3x3x3 decomposition
    # We know rank 27 is schoolbook. We want < 27.
    # Try ranks 23 to 26.
    for r in range(23, 27):
        if time.time() > deadline:
            break
        
        found = False
        # Increase restarts if time permits
        max_restarts = 20 if r > 24 else 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            res = find_3x3x3(r)
            if res is not None:
                U1, V1, W1 = res
                # Construct 3x3x6
                total_rank = 2 * r
                U_new = np.zeros((9, total_rank))
                V_new = np.zeros((18, total_rank))
                W_new = np.zeros((18, total_rank))
                
                # Block 1: k=0,1,2
                U_new[:, 0:r] = U1
                V_new[0:9, 0:r] = V1
                W_new[0:9, 0:r] = W1
                
                # Block 2: k=3,4,5
                U_new[:, r:2*r] = U1
                V_new[9:18, r:2*r] = V1
                W_new[9:18, r:2*r] = W1
                
                # Verify final solution
                valid = True
                for i in range(3):
                    for j in range(3):
                        for k in range(6):
                            s = 0.0
                            for c in range(total_rank):
                                s += U_new[i*3+j, c] * V_new[j*6+k, c] * W_new[k*3+i, c]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    best_rank = total_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                    found = True
                    break
        
        if found:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
