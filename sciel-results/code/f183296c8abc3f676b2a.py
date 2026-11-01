import numpy as np

def construct():
    n, m, p = 2, 4, 5
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    # Try ranks from 20 to 39
    for rank in range(20, 40):
        # Try a few random initializations
        for trial in range(5):
            np.random.seed(trial * 7919 + rank * 104729)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Alternating least squares with limited iterations
            for iteration in range(300):
                U_old, V_old, W_old = U.copy(), V.copy(), W.copy()
                
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        M_j = np.zeros((p, rank))
                        for k in range(p):
                            M_j[k] = V[j*p+k] * W[k*n+i]
                        U[i*m+j], _, _, _ = np.linalg.lstsq(M_j, np.ones(p), rcond=None)
                
                # Solve for V
                for k in range(p):
                    for i in range(n):
                        for j in range(m):
                            N_j = U[i*m+j] * W[k*n+i]
                            V[j*p+k], _, _, _ = np.linalg.lstsq(N_j, np.array([1.0]), rcond=None)
                
                # Solve for W
                for k in range(p):
                    for i in range(n):
                        for j in range(m):
                            P_j = U[i*m+j] * V[j*p+k]
                            W[k*n+i], _, _, _ = np.linalg.lstsq(P_j, np.array([1.0]), rcond=None)
                
                if np.allclose(U, U_old, atol=1e-10) and np.allclose(V, V_old, atol=1e-10) and np.allclose(W, W_old, atol=1e-10):
                    break
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        error += (val - 1)**2
            
            if error < 1e-8:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_rounded[i*m+j] * V_rounded[j*p+k] * W_rounded[k*n+i])
                            if abs(val - 1) > 1e-10:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
    
    # Fallback to schoolbook
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
