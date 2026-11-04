import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 8
    
    def verify_full(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def verify_2x3x4(U, V, W, rank):
        n2, m2, p2 = 2, 3, 4
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def build_full_from_2x3x4(U2, V2, W2, rank):
        full_rank = 2 * rank
        U = np.zeros((6, full_rank))
        V = np.zeros((24, full_rank))
        W = np.zeros((16, full_rank))
        
        U[:, :rank] = U2
        U[:, rank:] = U2
        
        for j in range(3):
            for k in range(4):
                V[j*8 + k, :rank] = V2[j*4 + k, :]
                V[j*8 + 4 + k, rank:] = V2[j*4 + k, :]
        
        for k in range(4):
            for i in range(2):
                W[k*2 + i, :rank] = W2[k*2 + i, :]
                W[(4+k)*2 + i, rank:] = W2[k*2 + i, :]
        
        U_list = [[float(U[i, r]) for r in range(full_rank)] for i in range(6)]
        V_list = [[float(V[i, r]) for r in range(full_rank)] for i in range(24)]
        W_list = [[float(W[i, r]) for r in range(full_rank)] for i in range(16)]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    def try_solve_2x3x4(target_rank):
        n2, m2, p2 = 2, 3, 4
        total_vars = target_rank * (n2*m2 + m2*p2 + p2*n2)
        
        def objective(x):
            U = x[:target_rank*n2*m2].reshape(n2*m2, target_rank)
            V = x[target_rank*n2*m2:target_rank*(n2*m2+m2*p2)].reshape(m2*p2, target_rank)
            W = x[target_rank*(n2*m2+m2*p2):].reshape(p2*n2, target_rank)
            
            residual = 0.0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r]
                        residual += (s - 1)**2
            return residual
        
        from scipy.optimize import minimize
        
        best_fun = float('inf')
        best_x = None
        
        for attempt in range(300):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 31 + 17)
            x0 = np.random.randn(total_vars) * 0.5
            
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if result.fun < best_fun:
                best_fun = result.fun
                best_x = result.x.copy()
            
            if best_fun < 1e-12:
                break
        
        if best_x is not None and best_fun < 1e-8:
            U = best_x[:target_rank*n2*m2].reshape(n2*m2, target_rank)
            V = best_x[target_rank*n2*m2:target_rank*(n2*m2+m2*p2)].reshape(m2*p2, target_rank)
            W = best_x[target_rank*(n2*m2+m2*p2):].reshape(p2*n2, target_rank)
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify_2x3x4(U_r, V_r, W_r, target_rank):
                return U_r, V_r, W_r
        
        return None
    
    # Try to find 2x3x4 decomposition
    for rank in range(12, 25):
        if time.time() > deadline - 20:
            break
        
        result = try_solve_2x3x4(rank)
        if result is not None:
            U2, V2, W2 = result
            full_decomp = build_full_from_2x3x4(U2, V2, W2, rank)
            
            U_list = full_decomp["U"]
            V_list = full_decomp["V"]
            W_list = full_decomp["W"]
            full_rank = 2 * rank
            
            if verify_full(U_list, V_list, W_list, full_rank):
                return full_decomp
    
    # Fallback: Strassen-based rank 44
    # Split m=3 into 2+1, p=8 into 2+2+2+2
    # For each (m_block, p_block): if m_block=2, p_block=2: Strassen rank 7
    #                              if m_block=1, p_block=2: schoolbook rank 4
    
    # Strassen for 2x2x2:
    # U factors (6x7), V factors (4x7), W factors (4x7)
    # C = A*B where A is 2x2, B is 2x2, C is 2x2
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # In tensor form:
    # U[i*2+j, r] for i in 0..1, j in 0..1
    # V[j*2+k, r] for j in 0..1, k in 0..1
    # W[k*2+i, r] for k in 0..1, i in 0..1
    
    # M1: U has A00 and A11, V has B00 and B11
    # U[0,0]=1, U[1,1]=1, others 0
    # V[0,0]=1, V[1,1]=1, others 0
    # W: C00 += 1, C11 += 1 → W[0,0]=1, W[3,1]=1 (k=0,i=0 and k=1,i=1)
    
    # This is getting complex. Let me just use schoolbook as fallback.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
