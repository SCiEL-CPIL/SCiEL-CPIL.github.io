import numpy as np
import time

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = start_time + 145
    
    # Schoolbook fallback
    def schoolbook():
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = None
    best_rank = n * m * p
    
    # Precompute b_vec patterns
    # For W: b_vec_W[i,k] has 1s at positions where ip=i, jp=jpp, kpp=k
    b_W = {}
    for i in range(n):
        for k in range(p):
            b = np.zeros(n*m*m*p)
            for jp in range(m):
                idx = ((i*m + jp)*m + jp)*p + k
                b[idx] = 1.0
            b_W[(i,k)] = b
    
    # For U: b_vec_U[ip,jp] has 1s where ip==i, jp==jpp, kpp==k
    b_U = {}
    for ip in range(n):
        for jp in range(m):
            b = np.zeros(n*p*m*p)
            for i in range(n):
                for k in range(p):
                    for jpp in range(m):
                        for kpp in range(p):
                            if ip == i and jp == jpp and kpp == k:
                                idx = (i*p + k)*m*p + jpp*p + kpp
                                b[idx] = 1.0
            b_U[(ip,jp)] = b
    
    # For V: b_vec_V[jpp,kpp] has 1s where ip==i, jp==jpp, kpp==k
    b_V = {}
    for jpp in range(m):
        for kpp in range(p):
            b = np.zeros(n*p*n*m)
            for i in range(n):
                for k in range(p):
                    for ip in range(n):
                        for jp in range(m):
                            if ip == i and jp == jpp and kpp == k:
                                idx = (i*p + k)*n*m + ip*m + jp
                                b[idx] = 1.0
            b_V[(jpp,kpp)] = b
    
    for R in range(45, 57):
        if time.time() > deadline - 30:
            break
        
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(123 + R * 100 + trial)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            
            success = False
            for it in range(20):
                if time.time() > deadline - 5:
                    break
                
                # Step 1: Solve for W
                AV = np.einsum('ijr,jkr->ijklr', U3, V3).reshape(-1, R)
                for i in range(n):
                    for k in range(p):
                        W3[k, i, :], _, _, _ = np.linalg.lstsq(AV, b_W[(i,k)], rcond=None)
                W = W3.reshape(p*n, R)
                
                # Step 2: Solve for U
                WV = np.einsum('kir,jkr->ikjkr', W3, V3).reshape(-1, R)
                for ip in range(n):
                    for jp in range(m):
                        U3[ip, jp, :], _, _, _ = np.linalg.lstsq(WV, b_U[(ip,jp)], rcond=None)
                U = U3.reshape(n*m, R)
                
                # Step 3: Solve for V
                UW = np.einsum('ijr,kir->ikijr', U3, W3).reshape(-1, R)
                for jpp in range(m):
                    for kpp in range(p):
                        V3[jpp, kpp, :], _, _, _ = np.linalg.lstsq(UW, b_V[(jpp,kpp)], rcond=None)
                V = V3.reshape(m*p, R)
            
            # Check
            max_res = 0
            for i in range(n):
                for k in range(p):
                    for ip in range(n):
                        for jp in range(m):
                            for jpp in range(m):
                                for kpp in range(p):
                                    val = np.dot(W[k*n+i, :], U[ip*m+jp, :] * V[jpp*p+kpp, :])
                                    expected = 1.0 if (ip == i and jp == jpp and kpp == k) else 0.0
                                    max_res = max(max_res, abs(val - expected))
            
            if max_res < 1e-6:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                max_res_r = 0
                for i in range(n):
                    for k in range(p):
                        for ip in range(n):
                            for jp in range(m):
                                for jpp in range(m):
                                    for kpp in range(p):
                                        val = np.dot(W_r[k*n+i, :], U_r[ip*m+jp, :] * V_r[jpp*p+kpp, :])
                                        expected = 1.0 if (ip == i and jp == jpp and kpp == k) else 0.0
                                        max_res_r = max(max_res_r, abs(val - expected))
                
                if max_res_r < 1e-9:
                    best_rank = R
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    break
        
        if best_result is not None:
            break
    
    if best_result is None:
        best_result = schoolbook()
    
    return best_result
