import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    T_flat = T.reshape(n*m, m*p * p*n)
    T_flat_v = T.reshape(m*p, n*m * p*n)
    T_flat_w = T.reshape(p*n, n*m * m*p)

    def check_and_quantize(U, V, W, rank):
        # Check if entries are close to k/2
        U_q = np.round(U * 2) / 2
        V_q = np.round(V * 2) / 2
        W_q = np.round(W * 2) / 2
        
        if not (np.allclose(U, U_q, atol=1e-5) and 
                np.allclose(V, V_q, atol=1e-5) and 
                np.allclose(W, W_q, atol=1e-5)):
            return None
            
        # Verify exact decomposition
        T_recon = np.zeros((n*m, m*p, p*n))
        for k in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U_q[:, k], V_q[:, k], W_q[:, k])
        
        if np.allclose(T_recon, T, atol=1e-5):
            U_list = [[float(x) for x in row] for row in U_q]
            V_list = [[float(x) for x in row] for row in V_q]
            W_list = [[float(x) for x in row] for row in W_q]
            return {"U": U_list, "V": V_list, "W": W_list}
        return None

    def als_step(U, V, W, rank, max_iter=15):
        for _ in range(max_iter):
            # Update U
            Y = np.zeros((rank, m*p * p*n))
            for k in range(rank):
                Y[k, :] = np.kron(V[:, k], W[:, k])
            try:
                U = np.linalg.lstsq(Y.T, T_flat, rcond=None)[0].T
            except:
                pass
                
            # Update V
            X = np.zeros((rank, n*m * p*n))
            for k in range(rank):
                X[k, :] = np.kron(U[:, k], W[:, k])
            try:
                V = np.linalg.lstsq(X.T, T_flat_v, rcond=None)[0].T
            except:
                pass
                
            # Update W
            Z = np.zeros((rank, n*m * m*p))
            for k in range(rank):
                Z[k, :] = np.kron(U[:, k], V[:, k])
            try:
                W = np.linalg.lstsq(Z.T, T_flat_w, rcond=None)[0].T
            except:
                pass
                
            # Check error
            T_recon = np.zeros((n*m, m*p, p*n))
            for k in range(rank):
                T_recon += np.einsum('i,j,k->ijk', U[:, k], V[:, k], W[:, k])
            err = np.linalg.norm(T_recon - T)
            if err < 1e-8:
                break
                
        return U, V, W, err

    start_time = time.time()
    deadline = start_time + 120
    
    # Try ranks from 23 to 26
    for rank in range(23, 27):
        if time.time() > deadline - 10:
            break
            
        for restart in range(100): # 100 restarts per rank
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            U, V, W, err = als_step(U, V, W, rank, max_iter=20)
            
            if err < 1e-6:
                result = check_and_quantize(U, V, W, rank)
                if result:
                    return result
                    
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
# EVOLVE-BLOCK-END
