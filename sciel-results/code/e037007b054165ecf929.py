import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    Nu = n * m
    Nv = m * p
    Nw = p * n
    
    # Build target tensor T of shape (Nu, Nv, Nw)
    T = np.zeros((Nu, Nv, Nw), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    def generate_schoolbook():
        rank = n * m * p
        U = np.zeros((Nu, rank), dtype=np.float64)
        V = np.zeros((Nv, rank), dtype=np.float64)
        W = np.zeros((Nw, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    start_time = time.time()
    time_limit = 140.0
    
    # Search for lower ranks
    for r in range(23, 17, -1):
        if time.time() - start_time > time_limit:
            break
            
        for restart in range(10):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(1000 + restart * 100 + r)
            U = np.random.randn(Nu, r)
            V = np.random.randn(Nv, r)
            W = np.random.randn(Nw, r)
            
            # Normalize
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(50):
                # Update U
                K = np.einsum('vc,wc->vwc', V, W)
                T_flat = T.reshape(Nu, -1)
                K_flat = K.reshape(-1, r)
                try:
                    U, _, _, _ = np.linalg.lstsq(K_flat, T_flat.T, rcond=None)
                except:
                    pass
                    
                # Update V
                L = np.einsum('uc,wc->uwc', U, W)
                T_v = T.transpose(1, 0, 2).reshape(Nv, -1)
                L_flat = L.reshape(-1, r)
                try:
                    V, _, _, _ = np.linalg.lstsq(L_flat, T_v.T, rcond=None)
                except:
                    pass
                    
                # Update W
                M = np.einsum('uc,vc->uvc', U, V)
                T_w = T.transpose(2, 0, 1).reshape(Nw, -1)
                M_flat = M.reshape(-1, r)
                try:
                    W, _, _, _ = np.linalg.lstsq(M_flat, T_w.T, rcond=None)
                except:
                    pass
                
                # Check residual
                T_rec = np.einsum('uc,vc,wc->uvw', U, V, W)
                residual = np.linalg.norm(T - T_rec)
                if residual < 1e-7:
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    T_snap = np.einsum('uc,vc,wc->uvw', U_snap, V_snap, W_snap)
                    if np.allclose(T_snap, T, atol=1e-9):
                        U_list = [[float(x) for x in row] for row in U_snap]
                        V_list = [[float(x) for x in row] for row in V_snap]
                        W_list = [[float(x) for x in row] for row in W_snap]
                        return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback
    U, V, W = generate_schoolbook()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
