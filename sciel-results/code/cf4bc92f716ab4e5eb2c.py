import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    def time_left():
        return deadline - time.time()
    
    schoolbook_rank = n * m * p  # 30
    
    def make_schoolbook():
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def max_error(U, V, W):
        r = U.shape[1]
        max_e = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]))
                    e = abs(s - 1.0)
                    if e > max_e:
                        max_e = e
        return max_e
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = make_schoolbook()
    best = (U_sb, V_sb, W_sb)
    best_rank = schoolbook_rank
    
    # Try to find lower-rank decompositions
    for target_rank in range(schoolbook_rank - 1, max(1, schoolbook_rank - 20), -1):
        if time_left() < 10:
            break
        
        found = False
        n_restarts = max(50, min(500, int(time_left() / 1.0)))
        
        for restart in range(n_restarts):
            if time_left() < 2:
                break
            
            # Random init for U and V
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            
            # Solve for W: for each (i,k), sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            nw_eqs = n * p  # 10
            nw_vars = p * n * target_rank  # 10 * target_rank
            A = np.zeros((nw_eqs, nw_vars))
            b = np.ones(nw_eqs)
            
            eq_idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(target_rank):
                            w_idx = (k * n + i) * target_rank + r
                            A[eq_idx, w_idx] += U[i * m + j, r] * V[j * p + k, r]
                    eq_idx += 1
            
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(p * n, target_rank)
            except:
                continue
            
            err = max_error(U, V, W)
            if err < 1e-9:
                best = (U.copy(), V.copy(), W.copy())
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue
    
    U_out, V_out, W_out = best
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
# EVOLVE-BLOCK-END
