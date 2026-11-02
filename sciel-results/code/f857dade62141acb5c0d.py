import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    nb = 3  # number of blocks
    bs = 2  # block size
    
    # Strassen for 2x2
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    strassen_x = [
        [(0,0,1), (1,1,1)],      # M0: A00+A11
        [(1,0,1), (1,1,1)],      # M1: A10+A11
        [(0,0,1)],               # M2: A00
        [(1,1,1)],               # M3: A11
        [(0,0,1), (0,1,1)],      # M4: A00+A01
        [(1,0,1), (0,0,-1)],     # M5: A10-A00
        [(0,1,1), (1,1,-1)],     # M6: A01-A11
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],      # M0: B00+B11
        [(0,0,1)],               # M1: B00
        [(0,1,1), (1,1,-1)],     # M2: B01-B11
        [(1,0,1), (0,0,-1)],     # M3: B10-B00
        [(1,1,1)],               # M4: B11
        [(0,0,1), (0,1,1)],      # M5: B00+B01
        [(1,0,1), (1,1,1)],      # M6: B10+B11
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],      # C00 += M0, C11 += M0
        [(1,0,1), (1,1,-1)],     # C10 += M1, C11 -= M1
        [(0,1,1), (1,1,1)],      # C01 += M2, C11 += M2
        [(0,0,1), (1,0,1)],      # C00 += M3, C10 += M3
        [(0,0,-1), (0,1,1)],     # C00 -= M4, C01 += M4
        [(1,1,1)],               # C11 += M5
        [(0,0,1)],               # C00 += M6
    ]
    
    rank_bs = nb * nb * nb * 7  # 189
    
    U_bs = np.zeros((n*m, rank_bs), dtype=np.float64)
    V_bs = np.zeros((m*p, rank_bs), dtype=np.float64)
    W_bs = np.zeros((p*n, rank_bs), dtype=np.float64)
    
    col = 0
    for ib in range(nb):
        for kb in range(nb):
            for jb in range(nb):
                for s in range(7):
                    for (lr, lc, coeff) in strassen_x[s]:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U_bs[gi * m + gj, col] += coeff
                    for (lr, lc, coeff) in strassen_y[s]:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V_bs[gj * p + gk, col] += coeff
                    for (lr, lc, coeff) in strassen_c[s]:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W_bs[gk * n + gi, col] += coeff
                    col += 1
    
    # Verify correctness
    def verify(U, V, W):
        rank = U.shape[1]
        for _ in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            C_comp = np.zeros((n, p))
            for r in range(rank):
                C_comp += U[:, r].reshape(n, m) * V[:, r].reshape(m, p) * W[:, r].reshape(p, n)
            if not np.allclose(C_comp, C_true, atol=1e-9):
                return False
        return True
    
    if not verify(U_bs, V_bs, W_bs):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_s = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
        V_s = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
        W_s = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j, col] = 1.0
                    V_s[j*p + k, col] = 1.0
                    W_s[k*n + i, col] = 1.0
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Try column elimination
    remaining = list(range(rank_bs))
    max_iterations = 30
    iter_count = 0
    
    while remaining and iter_count < max_iterations:
        if time.time() > deadline - 10:
            break
        
        best_col = -1
        best_residual = float('inf')
        
        for r in remaining:
            if len(remaining) <= 1:
                break
            others = [c for c in remaining if c != r]
            U_rest = U_bs[:, others]
            V_rest = V_bs[:, others]
            W_rest = W_bs[:, others]
            
            try:
                x_u, _, _, _ = np.linalg.lstsq(U_rest, U_bs[:, r], rcond=None)
            except:
                continue
            
            v_pred = V_rest @ x_u
            w_pred = W_rest @ x_u
            
            res_u = np.linalg.norm(U_bs[:, r] - U_rest @ x_u)
            res_v = np.linalg.norm(V_bs[:, r] - v_pred)
            res_w = np.linalg.norm(W_bs[:, r] - w_pred)
            
            total_res = res_u + res_v + res_w
            
            if total_res < 1e-8 and total_res < best_residual:
                # Check if x_u is sparse (at most 1 nonzero)
                nz = np.nonzero(np.abs(x_u) > 1e-10)[0]
                if len(nz) == 1:
                    idx = nz[0]
                    scale = x_u[idx]
                    ref_col = others[idx]
                    if (np.allclose(U_bs[:, r], scale * U_bs[:, ref_col], atol=1e-9) and
                        np.allclose(V_bs[:, r], scale * V_bs[:, ref_col], atol=1e-9) and
                        np.allclose(W_bs[:, r], scale * W_bs[:, ref_col], atol=1e-9)):
                        best_residual = total_res
                        best_col = r
        
        if best_col >= 0:
            remaining.remove(best_col)
            iter_count += 1
        else:
            break
    
    if len(remaining) < rank_bs:
        final_U = U_bs[:, remaining]
        final_V = V_bs[:, remaining]
        final_W = W_bs[:, remaining]
        if verify(final_U, final_V, final_W):
            return {"U": final_U.tolist(), "V": final_V.tolist(), "W": final_W.tolist()}
    
    return {"U": U_bs.tolist(), "V": V_bs.tolist(), "W": W_bs.tolist()}
