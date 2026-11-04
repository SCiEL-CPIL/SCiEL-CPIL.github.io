"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb), dtype=float)
    V_sb = np.zeros((m*p, rank_sb), dtype=float)
    W_sb = np.zeros((p*n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Work in scaled integer domain: U2=2U, V2=2V, W2=2W, target sum=8
    def verify(U2, V2, W2):
        r = U2.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U2[i*m+j, c] * V2[j*p+k, c] * W2[k*n+i, c]
                    if s != 8:
                        return False
        return True
    
    # Try to find better solutions
    for target_rank in range(15, best_rank):
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random small integers
            U2 = np.random.randint(-3, 4, size=(n*m, target_rank))
            V2 = np.random.randint(-3, 4, size=(m*p, target_rank))
            W2 = np.random.randint(-3, 4, size=(p*n, target_rank))
            
            # Local search
            def get_residuals():
                res = np.zeros((n,m,p), dtype=int)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(target_rank):
                                s += U2[i*m+j,c]*V2[j*p+k,c]*W2[k*n+i,c]
                            res[i,j,k] = s - 8
                return res
            
            res = get_residuals()
            err = np.sum(np.abs(res))
            
            max_iter = 2000
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                if err == 0:
                    break
                
                # Randomly pick which matrix to modify
                mc = random.random()
                if mc < 0.33:
                    # Modify U2
                    row = random.randint(0, n*m-1)
                    c = random.randint(0, target_rank-1)
                    i, j = divmod(row, m)
                    old = U2[row, c]
                    for delta in [-2,-1,1,2]:
                        nv = old + delta
                        if abs(nv) > 5:
                            continue
                        new_err = err
                        for k in range(p):
                            oc = old * V2[j*p+k,c] * W2[k*n+i,c]
                            nc = nv * V2[j*p+k,c] * W2[k*n+i,c]
                            new_err += abs(res[i,j,k]+nc-oc) - abs(res[i,j,k])
                        if new_err < err:
                            U2[row,c] = nv
                            for k in range(p):
                                oc = old * V2[j*p+k,c] * W2[k*n+i,c]
                                nc = nv * V2[j*p+k,c] * W2[k*n+i,c]
                                res[i,j,k] += nc-oc
                            err = new_err
                            break
                elif mc < 0.66:
                    # Modify V2
                    row = random.randint(0, m*p-1)
                    c = random.randint(0, target_rank-1)
                    j, k = divmod(row, p)
                    old = V2[row, c]
                    for delta in [-2,-1,1,2]:
                        nv = old + delta
                        if abs(nv) > 5:
                            continue
                        new_err = err
                        for i in range(n):
                            oc = U2[i*m+j,c]*old*W2[k*n+i,c]
                            nc = U2[i*m+j,c]*nv*W2[k*n+i,c]
                            new_err += abs(res[i,j,k]+nc-oc) - abs(res[i,j,k])
                        if new_err < err:
                            V2[row,c] = nv
                            for i in range(n):
                                oc = U2[i*m+j,c]*old*W2[k*n+i,c]
                                nc = U2[i*m+j,c]*nv*W2[k*n+i,c]
                                res[i,j,k] += nc-oc
                            err = new_err
                            break
                else:
                    # Modify W2
                    row = random.randint(0, p*n-1)
                    c = random.randint(0, target_rank-1)
                    k, i = divmod(row, n)
                    old = W2[row, c]
                    for delta in [-2,-1,1,2]:
                        nv = old + delta
                        if abs(nv) > 5:
                            continue
                        new_err = err
                        for j in range(m):
                            oc = U2[i*m+j,c]*V2[j*p+k,c]*old
                            nc = U2[i*m+j,c]*V2[j*p+k,c]*nv
                            new_err += abs(res[i,j,k]+nc-oc) - abs(res[i,j,k])
                        if new_err < err:
                            W2[row,c] = nv
                            for j in range(m):
                                oc = U2[i*m+j,c]*V2[j*p+k,c]*old
                                nc = U2[i*m+j,c]*V2[j*p+k,c]*nv
                                res[i,j,k] += nc-oc
                            err = new_err
                            break
            
            if err == 0:
                if verify(U2, V2, W2):
                    U = (U2/2.0).tolist()
                    V = (V2/2.0).tolist()
                    W = (W2/2.0).tolist()
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
