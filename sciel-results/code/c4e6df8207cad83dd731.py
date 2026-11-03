"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(nm)]
        V = [[0]*r for _ in range(mp)]
        W = [[0]*r for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    np.random.seed(42)
    
    for r in range(18, 37):
        if time.time() - start_time > DEADLINE:
            break
        
        found = False
        for attempt in range(30):
            if time.time() - start_time > DEADLINE:
                break
            
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1
            
            for it in range(300):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Solve for U
                A_U = np.zeros((nm*mp*pn, nm*r))
                b_U = np.zeros(nm*mp*pn)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c_idx in range(pn):
                            for c in range(r):
                                A_U[idx, a*r+c] = V[b,c] * W[c_idx,c]
                            b_U[idx] = T[a,b,c_idx]
                            idx += 1
                U_new, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                U_new = U_new.reshape(nm, r)
                
                # Solve for V
                A_V = np.zeros((nm*mp*pn, mp*r))
                b_V = np.zeros(nm*mp*pn)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c_idx in range(pn):
                            for c in range(r):
                                A_V[idx, b*r+c] = U_new[a,c] * W[c_idx,c]
                            b_V[idx] = T[a,b,c_idx]
                            idx += 1
                V_new, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                V_new = V_new.reshape(mp, r)
                
                # Solve for W
                A_W = np.zeros((nm*mp*pn, pn*r))
                b_W = np.zeros(nm*mp*pn)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c_idx in range(pn):
                            for c in range(r):
                                A_W[idx, c_idx*r+c] = U_new[a,c] * V_new[b,c]
                            b_W[idx] = T[a,b,c_idx]
                            idx += 1
                W_new, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                W_new = W_new.reshape(pn, r)
                
                U, V, W = U_new, V_new, W_new
                
                if it % 50 == 0:
                    error = np.sum(np.abs(np.einsum('ac,bc,dc->abd', U, V, W) - T))
                    if error < 1e-6:
                        if r < best_rank:
                            best_rank = r
                            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        found = True
                        break
            
            if found:
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END

