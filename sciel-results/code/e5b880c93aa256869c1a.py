import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 2, 12
    
    T = np.zeros((4, 24, 24), dtype=np.float64)
    for r in range(2):
        for i in range(2):
            for k in range(12):
                T[r*2+i, i*12+k, k*2+r] = 1.0
    
    T_a = T.reshape(4, 576)
    T_b = T.transpose(1,0,2).reshape(24, 576)
    T_c = T.transpose(2,0,1).reshape(24, 576)
    
    def try_rank(r, max_seeds=5, max_iters=150):
        if time.time() > deadline:
            return None
        for seed in range(max_seeds):
            if time.time() > deadline:
                return None
            np.random.seed(seed * 7919 + r * 31)
            U = np.random.randn(4, r)
            V = np.random.randn(24, r)
            W = np.random.randn(24, r)
            
            best_err = float('inf')
            best_UVW = None
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                D = (V[:, :, None] * W[:, None, :]).reshape(r, 576)
                U, _, _, _ = np.linalg.lstsq(D, T_a, rcond=None)
                
                E = (U[:, :, None] * W[:, None, :]).reshape(r, 576)
                V, _, _, _ = np.linalg.lstsq(E, T_b, rcond=None)
                
                F = (U[:, :, None] * V[:, None, :]).reshape(r, 576)
                W, _, _, _ = np.linalg.lstsq(F, T_c, rcond=None)
                
                # Check convergence
                if it % 10 == 9:
                    T_rec = np.einsum('as,bs,cs->abc', U, V, W)
                    err = np.linalg.norm(T - T_rec)
                    if err < best_err:
                        best_err = err
                        best_UVW = (U.copy(), V.copy(), W.copy())
                    if err < 1e-10:
                        return U, V, W
            
            if best_UVW is not None and best_err < 1e-8:
                return best_UVW
        return None
    
    for r in range(8, 30):
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            # Round to nearest 0.5 if close
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify rounded version
            T_rec = np.einsum('as,bs,cs->abc', U_r, V_r, W_r)
            err_rounded = np.linalg.norm(T - T_rec)
            
            if err_rounded < 1e-10:
                U, V, W = U_r, V_r, W_r
            else:
                # Use unrounded
                pass
            
            U_list = [[float(U[i,s]) for s in range(r)] for i in range(4)]
            V_list = [[float(V[i,s]) for s in range(r)] for i in range(24)]
            W_list = [[float(W[i,s]) for s in range(r)] for i in range(24)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: Strassen 42
    rank = 42
    U_out = [[0]*rank for _ in range(4)]
    V_out = [[0]*rank for _ in range(24)]
    W_out = [[0]*rank for _ in range(24)]
    U_forms = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]]
    for u in range(6):
        for t in range(7):
            c = u*7+t
            for i in range(4):
                U_out[i][c] = U_forms[t][i]
            if t==0: V_out[2*u][c]+=1; V_out[12+2*u+1][c]+=1
            elif t==1: V_out[2*u][c]+=1
            elif t==2: V_out[2*u+1][c]+=1; V_out[12+2*u+1][c]-=1
            elif t==3: V_out[12+2*u][c]+=1; V_out[2*u][c]-=1
            elif t==4: V_out[12+2*u+1][c]+=1
            elif t==5: V_out[2*u][c]+=1; V_out[2*u+1][c]+=1
            elif t==6: V_out[12+2*u][c]+=1; V_out[12+2*u+1][c]+=1
            if t==0: W_out[4*u][c]+=1; W_out[4*u+3][c]+=1
            elif t==1: W_out[4*u+1][c]+=1; W_out[4*u+3][c]-=1
            elif t==2: W_out[4*u+2][c]+=1; W_out[4*u+3][c]+=1
            elif t==3: W_out[4*u][c]+=1; W_out[4*u+1][c]+=1
            elif t==4: W_out[4*u][c]-=1; W_out[4*u+2][c]+=1
            elif t==5: W_out[4*u+3][c]+=1
            elif t==6: W_out[4*u][c]+=1
    return {"U": U_out, "V": V_out, "W": W_out}
