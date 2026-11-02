import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 2, 2, 12
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill

    def get_strassen_42():
        """Fallback: 6 blocks of 2x2x2 Strassen, rank 42"""
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen for 2x2x2: 7 products
        # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
        # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
        
        for b in range(6):
            for s in range(7):
                r = b * 7 + s
                col0 = 2 * b
                col1 = 2 * b + 1
                
                if s == 0:  # M1=(A00+A11)*(B00+B11)
                    U[0][r] = 1; U[3][r] = 1
                    V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
                    W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
                elif s == 1:  # M2=(A10+A11)*B00
                    U[2][r] = 1; U[3][r] = 1
                    V[0*p+col0][r] = 1
                    W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
                elif s == 2:  # M3=A00*(B01-B11)
                    U[0][r] = 1
                    V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
                    W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
                elif s == 3:  # M4=A11*(B10-B00)
                    U[3][r] = 1
                    V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
                    W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
                elif s == 4:  # M5=(A00+A01)*B11
                    U[0][r] = 1; U[1][r] = 1
                    V[1*p+col1][r] = 1
                    W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
                elif s == 5:  # M6=(A10-A00)*(B00+B01)
                    U[2][r] = 1; U[0][r] = -1
                    V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
                    W[col1*n+1][r] = 1
                elif s == 6:  # M7=(A01-A11)*(B10+B11)
                    U[1][r] = 1; U[3][r] = -1
                    V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
                    W[col0*n+0][r] = 1; W[col0*n+1][r] = -1
        
        return {"U": U, "V": V, "W": W}

    def verify_2x2x4(U, V, W, R):
        """Verify 2x2x4 decomposition: U(4xR), V(8xR), W(8xR)"""
        T_calc = np.zeros((2, 2, 4))
        for r in range(R):
            for i in range(2):
                for j in range(2):
                    u_val = U[i*2+j, r]
                    if u_val != 0:
                        for k in range(4):
                            T_calc[i, j, k] += u_val * V[j*4+k, r] * W[k*2+i, r]
        T_target = np.ones((2, 2, 4))
        return np.max(np.abs(T_calc - T_target)) < 1e-8

    def search_2x2x4(R=9, num_restarts=40):
        """Search for rank-R decomposition of 2x2x4 tensor"""
        if time.time() > deadline - 20:
            return None
            
        dim_U = 4
        dim_V = 8
        dim_W = 8
        total_dim = dim_U * R + dim_V * R + dim_W * R
        
        def objective(params):
            if time.time() > deadline - 15:
                return 1e10
            U = np.reshape(params[:dim_U*R], (dim_U, R))
            V = np.reshape(params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(params[dim_U*R+dim_V*R:], (dim_W, R))
            
            T_calc = np.zeros((2, 2, 4))
            for r in range(R):
                u_r = U[:, r]
                v_r = V[:, r]
                w_r = W[:, r]
                for i in range(2):
                    for j in range(2):
                        uv = u_r[i*2+j] * v_r[j*4+0]  # just for speed, do full
                        # Full computation
                        for k in range(4):
                            T_calc[i, j, k] += u_r[i*2+j] * v_r[j*4+k] * w_r[k*2+i]
            
            T_target = np.ones((2, 2, 4))
            return np.sum((T_calc - T_target)**2)
        
        best_error = float('inf')
        best_params = None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            if restart % 4 == 0:
                params = np.random.uniform(-1, 1, size=total_dim)
            elif restart % 4 == 1:
                params = np.random.normal(0, 0.5, size=total_dim)
            elif restart % 4 == 2:
                # Sparse init
                params = np.zeros(total_dim)
                nz = max(1, total_dim // 5)
                idx = np.random.choice(total_dim, nz, replace=False)
                params[idx] = np.random.uniform(-1, 1, nz)
            else:
                # Try structured init based on Strassen
                params = np.zeros(total_dim)
                # Initialize with Strassen-like structure for first 7 terms
                for s in range(min(7, R)):
                    u_off = s
                    v_off = dim_U * R + s
                    w_off = (dim_U + dim_V) * R + s
                    # M1
                    if s == 0:
                        params[u_off] = 1; params[u_off+3] = 1
                        params[v_off] = 1; params[v_off+5] = 1
                        params[w_off] = 1; params[w_off+5] = 1
                    elif s == 1:
                        params[u_off+2] = 1; params[u_off+3] = 1
                        params[v_off] = 1
                        params[w_off+2] = 1; params[w_off+5] = -1
                    elif s == 2:
                        params[u_off] = 1
                        params[v_off+1] = 1; params[v_off+5] = -1
                        params[w_off+1] = 1; params[w_off+5] = 1
                    elif s == 3:
                        params[u_off+3] = 1
                        params[v_off+4] = 1; params[v_off] = -1
                        params[w_off] = 1; params[w_off+2] = 1
                    elif s == 4:
                        params[u_off] = 1; params[u_off+1] = 1
                        params[v_off+5] = 1
                        params[w_off] = -1; params[w_off+1] = 1
                    elif s == 5:
                        params[u_off+2] = 1; params[u_off] = -1
                        params[v_off] = 1; params[v_off+1] = 1
                        params[w_off+5] = 1
                    elif s == 6:
                        params[u_off+1] = 1; params[u_off+3] = -1
                        params[v_off+4] = 1; params[v_off+5] = 1
                        params[w_off] = 1; params[w_off+2] = -1
            
            res = minimize(objective, params, method='L-BFGS-B',
                          options={'maxiter': 500, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x.copy()
                
            if best_error < 1e-12:
                break
        
        if best_error < 1e-8 and best_params is not None:
            U = np.reshape(best_params[:dim_U*R], (dim_U, R))
            V = np.reshape(best_params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(best_params[dim_U*R+dim_V*R:], (dim_W, R))
            
            # Snap to half-integers
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            
            if verify_2x2x4(U_s, V_s, W_s, R):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
            
            # Try integer snap
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            if verify_2x2x4(U_i, V_i, W_i, R):
                return U_i.tolist(), V_i.tolist(), W_i.tolist()
            
            # Use the raw solution if close enough
            if best_error < 1e-6:
                return U.tolist(), V.tolist(), W.tolist()
        
        return None

    def build_2x2x12_from_2x2x4(U4, V4, W4, R):
        """Build 2x2x12 decomposition from 2x2x4 block, rank = 3*R"""
        full_rank = R * 3
        full_U = [[0] * full_rank for _ in range(4)]
        full_V = [[0] * full_rank for _ in range(24)]
        full_W = [[0] * full_rank for _ in range(24)]
        
        for block in range(3):
            offset = block * R
            for s in range(R):
                r = offset + s
                for i in range(2):
                    for j in range(2):
                        full_U[i*2+j][r] = float(U4[i*2+j][s])
                for j in range(2):
                    for k in range(4):
                        full_V[j*12 + (block*4 + k)][r] = float(V4[j*4+k][s])
                for k in range(4):
                    for i in range(2):
                        full_W[(block*4 + k)*2 + i][r] = float(W4[k*2+i][s])
        
        return {"U": full_U, "V": full_V, "W": full_W}

    def verify_full(result):
        """Verify full 2x2x12 decomposition"""
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        rank = U.shape[1]
        
        T_calc = np.zeros((2, 2, 12))
        for r in range(rank):
            for i in range(2):
                for j in range(2):
                    u_val = U[i*2+j, r]
                    if u_val != 0:
                        for k in range(12):
                            T_calc[i, j, k] += u_val * V[j*12+k, r] * W[k*2+i, r]
        
        T_target = np.ones((2, 2, 12))
        return np.max(np.abs(T_calc - T_target)) < 1e-6

    # Try rank 9 for 2x2x4 (gives total rank 27)
    if time.time() < deadline - 30:
        res4 = search_2x2x4(R=9, num_restarts=50)
        if res4:
            U4, V4, W4 = res4
            full = build_2x2x12_from_2x2x4(U4, V4, W4, 9)
            if verify_full(full):
                return full

    # Try rank 10 for 2x2x4 (gives total rank 30)
    if time.time() < deadline - 25:
        res4 = search_2x2x4(R=10, num_restarts=40)
        if res4:
            U4, V4, W4 = res4
            full = build_2x2x12_from_2x2x4(U4, V4, W4, 10)
            if verify_full(full):
                return full

    # Try rank 11 for 2x2x4 (gives total rank 33)
    if time.time() < deadline - 20:
        res4 = search_2x2x4(R=11, num_restarts=30)
        if res4:
            U4, V4, W4 = res4
            full = build_2x2x12_from_2x2x4(U4, V4, W4, 11)
            if verify_full(full):
                return full

    # Try rank 12 for 2x2x4 (gives total rank 36)
    if time.time() < deadline - 15:
        res4 = search_2x2x4(R=12, num_restarts=25)
        if res4:
            U4, V4, W4 = res4
            full = build_2x2x12_from_2x2x4(U4, V4, W4, 12)
            if verify_full(full):
                return full

    # Fallback: Strassen 42
    return get_strassen_42()
