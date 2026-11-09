import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130

    def schoolbook():
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

    def verify(U, V, W, atol=1e-10):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        rank = U_np.shape[1]
        tensor = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u_mat = U_np[:, r].reshape(n, m)
            v_mat = V_np[:, r].reshape(m, p)
            w_mat = W_np[:, r].reshape(p, n)
            prod = u_mat[:, :, np.newaxis] * v_mat[np.newaxis, :, :] * w_mat.T[:, np.newaxis, :]
            tensor += prod
        expected = np.ones((n, m, p), dtype=np.float64)
        return np.allclose(tensor, expected, atol=atol)

    def construct_rank_18():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for i in range(n):
            base = i * 9
            for l in range(9):
                col = base + l
                if l < m:
                    for j in range(m):
                        if j <= l:
                            U[i * m + j][col] = 1
                    for j in range(m):
                        for k in range(p):
                            if j == l:
                                V[j * p + k][col] = 1
                    for k in range(p):
                        W[k * n + i][col] = 1
                else:
                    lp = l - m
                    for j in range(m):
                        U[i * m + j][col] = 1
                    for j in range(m):
                        for k in range(p):
                            if k == lp + 1:
                                V[j * p + k][col] = 1
                    for k in range(p):
                        if k == lp + 1:
                            W[k * n + i][col] = 1
                            
        return {"U": U, "V": V, "W": W}

    def try_rank(rk, seed=42):
        """Try to find a decomposition of given rank using random search + refinement."""
        np.random.seed(seed)
        best_U = None
        best_V = None
        best_W = None
        best_err = float('inf')
        
        max_iters = 200
        lr = 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            U_mat = np.random.randn(n * m, rk)
            V_mat = np.random.randn(m * p, rk)
            W_mat = np.random.randn(p * n, rk)
            
            # Normalize columns to avoid scale issues
            for c in range(rk):
                norm_u = np.linalg.norm(U_mat[:, c])
                norm_v = np.linalg.norm(V_mat[:, c])
                norm_w = np.linalg.norm(W_mat[:, c])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    # Scale so product of norms is 1
                    scale = (norm_u * norm_v * norm_w) ** (-1/3)
                    U_mat[:, c] *= scale
                    V_mat[:, c] *= scale
                    W_mat[:, c] *= scale
            
            cur_err = float('inf')
            cur_U, cur_V, cur_W = U_mat.copy(), V_mat.copy(), W_mat.copy()
            
            for sub_it in range(50):
                if time.time() > deadline - 10:
                    break
                    
                # Compute tensor from current U, V, W
                tensor = np.zeros((n, m, p), dtype=np.float64)
                for r in range(rk):
                    u_mat = cur_U[:, r].reshape(n, m)
                    v_mat = cur_V[:, r].reshape(m, p)
                    w_mat = cur_W[:, r].reshape(p, n)
                    prod = u_mat[:, :, np.newaxis] * v_mat[np.newaxis, :, :] * w_mat.T[:, np.newaxis, :]
                    tensor += prod
                
                error = np.linalg.norm(tensor - np.ones((n, m, p)))
                if error < cur_err:
                    cur_err = error
                    if error < best_err:
                        best_err = error
                        best_U = cur_U.copy()
                        best_V = cur_V.copy()
                        best_W = cur_W.copy()
                
                if error < 1e-10:
                    break
                
                # Gradient-free update: perturb
                lr_cur = lr * (0.95 ** sub_it)
                for r in range(rk):
                    # Perturb each column
                    U_mat[:, r] += lr_cur * np.random.randn(n * m)
                    V_mat[:, r] += lr_cur * np.random.randn(m * p)
                    W_mat[:, r] += lr_cur * np.random.randn(p * n)
                
                # Keep U, V, W as the perturbed version
                cur_U = U_mat.copy()
                cur_V = V_mat.copy()
                cur_W = W_mat.copy()
            
            # Check if we found a good solution
            if best_err < 1e-8:
                break
        
        if best_err < 1e-8:
            # Convert to lists
            U_list = [[float(best_U[i, j]) for j in range(rk)] for i in range(n * m)]
            V_list = [[float(best_V[i, j]) for j in range(rk)] for i in range(m * p)]
            W_list = [[float(best_W[i, j]) for j in range(rk)] for i in range(p * n)]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # First try the known rank-18 construction
    result = construct_rank_18()
    if verify(result["U"], result["V"], result["W"], atol=1e-8):
        return result
    
    # Try random search for lower rank
    for rk in [17, 16, 15, 14, 13, 12]:
        if time.time() > deadline - 30:
            break
        sol = try_rank(rk)
        if sol is not None:
            if verify(sol["U"], sol["V"], sol["W"], atol=1e-8):
                return sol
    
    # Fallback to schoolbook
    return schoolbook()
