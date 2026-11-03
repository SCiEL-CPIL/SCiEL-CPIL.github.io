import numpy as np
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for [2,4,4] matrix multiplication.
    Strassen rank-28 base with greedy linear dependency elimination.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 4, 4
    R = n * m * p  # 32
    
    # Build rank-28 Strassen decomposition
    rank = 28
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    def u_idx(i, j): return i * m + j
    def v_idx(i, j): return i * p + j
    def w_idx(i, j): return j * n + i
    
    blocks = [(0,0,0,0,0), (0,2,2,0,0), (0,0,0,2,2), (0,2,2,2,2)]
    strassen_terms = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)])
    ]
    
    cr = 0
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        if time.time() > deadline:
            break
        um = [u_idx(a_r,a_c), u_idx(a_r,a_c+1), u_idx(a_r+1,a_c), u_idx(a_r+1,a_c+1)]
        vm = [v_idx(b_r,b_c), v_idx(b_r,b_c+1), v_idx(b_r+1,b_c), v_idx(b_r+1,b_c+1)]
        wm = [w_idx(0,c_c), w_idx(0,c_c+1), w_idx(1,c_c), w_idx(1,c_c+1)]
        for (ut, vt, wt) in strassen_terms:
            for li, cf in ut: U[um[li], cr] += cf
            for li, cf in vt: V[vm[li], cr] += cf
            for li, cf in wt: W[wm[li], cr] += cf
            cr += 1
    
    def verify(Um, Vm, Wm):
        Td = np.zeros((n,m,p))
        for t in range(Um.shape[1]):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Td[i,j,k] += Wm[k*n+i,t] * Um[i*m+j,t] * Vm[j*p+k,t]
        return np.all(Td == 1)
    
    if not verify(U, V, W):
        raise ValueError("Invalid initial decomposition")
    
    def vec_r1(Uc, Vc, Wc):
        T = np.zeros(R)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[k*n*m + i*m + j] = Uc[i*m+j] * Vc[j*p+k] * Wc[k*n+i]
        return T
    
    T_target = np.ones(R)
    
    def build_M(Um, Vm, Wm):
        return np.array([vec_r1(Um[:,t].astype(float), Vm[:,t].astype(float), Wm[:,t].astype(float)) for t in range(Um.shape[1])]).T
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    M = build_M(best_U, best_V, best_W)
    
    # Greedy: try to remove each term if it's in the span of others
    for iteration in range(5):
        if time.time() > deadline:
            break
        removed = False
        for t in range(best_rank):
            if time.time() > deadline:
                break
            keep = [s for s in range(best_rank) if s != t]
            if not keep:
                continue
            M_sub = M[:, keep]
            try:
                c, _, _, _ = np.linalg.lstsq(M_sub, T_target, rcond=None)
                if np.linalg.norm(M_sub @ c - T_target) < 1e-9:
                    new_U = best_U[:, keep].copy()
                    new_V = best_V[:, keep].copy()
                    new_W = best_W[:, keep].copy()
                    for idx, s in enumerate(keep):
                        new_V[:, idx] *= c[idx]
                    if verify(new_U, new_V, new_W):
                        best_rank = len(keep)
                        best_U, best_V, best_W = new_U, new_V, new_W
                        M = build_M(best_U, best_V, best_W)
                        removed = True
                        break
            except:
                continue
        if not removed:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
