"""Find rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Build the multiplication tensor T[a,b,c] where:
    # a = i*m+j (0..9), b = j*p+k (0..24), c = k*n+i (0..9)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition (rank 50)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    def verify(U, V, W):
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(recon, T, atol=1e-9)
    
    def clean_halfint(M):
        return np.round(2*M)/2.0
    
    # Try to find lower rank
    # Lower bound from unfoldings
    T1 = T.reshape(n*m, -1)
    T2 = T.reshape(m*p, -1)
    T3 = T.reshape(p*n, -1)
    lb = max(np.linalg.matrix_rank(T1), np.linalg.matrix_rank(T2), np.linalg.matrix_rank(T3))
    
    for target_rank in range(lb, rank_sb + 1):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(777 + seed)
            
            # Initialize
            U_c = np.random.randn(n*m, target_rank) * 0.5
            V_c = np.random.randn(m*p, target_rank) * 0.5
            W_c = np.random.randn(p*n, target_rank) * 0.5
            
            # ALS iterations
            for it in range(500):
                if time.time() > deadline:
                    break
                
                # Update U
                recon = np.einsum('ac,bc,dc->abd', U_c, V_c, W_c)
                R = T - recon
                for c in range(target_rank):
                    vw = np.outer(V_c[:,c], W_c[:,c])
                    R_u = R.reshape(n*m, -1)
                    d = np.dot(vw, vw)
                    if d > 1e-15:
                        U_c[:,c] = R_u @ vw / d
                
                # Update V
                recon = np.einsum('ac,bc,dc->abd', U_c, V_c, W_c)
                R = T - recon
                for c in range(target_rank):
                    uw = np.outer(U_c[:,c], W_c[:,c])
                    R_v = R.reshape(m*p, -1)
                    d = np.dot(uw, uw)
                    if d > 1e-15:
                        V_c[:,c] = R_v @ uw / d
                
                # Update W
                recon = np.einsum('ac,bc,dc->abd', U_c, V_c, W_c)
                R = T - recon
                for c in range(target_rank):
                    uv = np.outer(U_c[:,c], V_c[:,c])
                    R_w = R.reshape(p*n, -1)
                    d = np.dot(uv, uv)
                    if d > 1e-15:
                        W_c[:,c] = R_w @ uv / d
                
                recon = np.einsum('ac,bc,dc->abd', U_c, V_c, W_c)
                if np.linalg.norm(T - recon) < 1e-12:
                    break
            
            recon = np.einsum('ac,bc,dc->abd', U_c, V_c, W_c)
            if np.allclose(recon, T, atol=1e-8):
                U2, V2, W2 = clean_halfint(U_c), clean_halfint(V_c), clean_halfint(W_c)
                if verify(U2, V2, W2):
                    best_rank = target_rank
                    best_U, best_V, best_W = U2, V2, W2
                    found = True
                    break
        
        if found:
            break
    
    def to_list(M):
        res = []
        for row in M:
            cr = []
            for v in row:
                hv = round(v*2)/2.0
                if abs(hv - round(hv)) < 1e-10:
                    cr.append(int(round(hv)))
                else:
                    cr.append(hv)
            res.append(cr)
        return res
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
