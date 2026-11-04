"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 30
    pn = p * n  # 20
    
    schoolbook_rank = n * m * p  # 60
    
    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    start_time = time.time()
    deadline = start_time + 130
    
    # Try rank 55
    rank = 55
    
    # We need to find U (6x55), V (30x55), W (20x55) with half-integer entries
    # such that for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a non-linear system. Let's try to use least_squares.
    
    # Flatten U, V, W into a single vector
    # x = [U.flatten(), V.flatten(), W.flatten()]
    # len(x) = 6*55 + 30*55 + 20*55 = 55*56 = 3080
    
    # The residual for each (i,j,k) is:
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
    
    # This is trilinear in x, so it's non-linear.
    
    # Let's try a random initialization with half-integer values
    np.random.seed(42)
    
    def residual(x):
        U_flat = x[:nm*rank]
        V_flat = x[nm*rank:nm*rank+mp*rank]
        W_flat = x[nm*rank+mp*rank:]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res.append(val - 1)
        return np.array(res)
    
    # Try a few random initializations
    best_x = None
    best_res = None
    
    for attempt in range(10):
        if time.time() > deadline - 20:
            break
        
        # Random half-integer initialization
        x0 = np.random.randint(-2, 3, size=nm*rank + mp*rank + pn*rank) * 0.5
        
        try:
            result = least_squares(residual, x0, method='lm', max_nfev=1000)
            res_norm = np.linalg.norm(result.fun)
            if best_res is None or res_norm < best_res:
                best_res = res_norm
                best_x = result.x
        except Exception:
            pass
    
    if best_x is not None and best_res < 1e-6:
        # Extract U, V, W
        U_flat = best_x[:nm*rank]
        V_flat = best_x[nm*rank:nm*rank+mp*rank]
        W_flat = best_x[nm*rank+mp*rank:]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        # Round to nearest half-integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            U_list = [[float(U[r, c]) for c in range(rank)] for r in range(nm)]
            V_list = [[float(V[r, c]) for c in range(rank)] for r in range(mp)]
            W_list = [[float(W[r, c]) for c in range(rank)] for r in range(pn)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return schoolbook()
# EVOLVE-BLOCK-END
