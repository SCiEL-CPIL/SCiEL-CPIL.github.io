import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    np_output = n * p  # 20
    np_inputA = n * m  # 6
    np_inputB = m * p  # 30
    
    # Build multiplication tensor T[o, a, b]
    T = np.zeros((np_output, np_inputA, np_inputB))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                out_idx = k * n + i
                a_idx = i * m + j
                b_idx = j * p + k
                T[out_idx, a_idx, b_idx] = 1.0
    
    def check_and_snap(U, V, W, r, tol=1e-8):
        """Check if decomposition is valid and try to snap to rationals."""
        if time.time() > deadline - 5:
            return None
        
        # Check residual
        residual = 0.0
        for o in range(np_output):
            for a in range(np_inputA):
                for b in range(np_inputB):
                    val = np.dot(U[a, :], V[b, :] * W[o, :])
                    residual += (val - T[o, a, b]) ** 2
        residual = np.sqrt(residual)
        
        if residual > tol:
            return None
        
        # Try to snap to rationals with small denominators
        def snap_matrix(M, max_den=100):
            snapped = np.zeros_like(M)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    val = M[i, j]
                    if abs(val) < 1e-10:
                        snapped[i, j] = 0
                        continue
                    # Try to find p/q approximation
                    best_num, best_den = int(round(val)), 1
                    best_err = abs(val - best_num / best_den)
                    for den in range(2, max_den + 1):
                        num = round(val * den)
                        err = abs(val - num / den)
                        if err < best_err:
                            best_err = err
                            best_num, best_den = num, den
                        if err < 1e-12:
                            break
                    snapped[i, j] = best_num / best_den
            return snapped
        
        U_s = snap_matrix(U)
        V_s = snap_matrix(V)
        W_s = snap_matrix(W)
        
        # Verify snapped version
        res2 = 0.0
        for o in range(np_output):
            for a in range(np_inputA):
                for b in range(np_inputB):
                    val = np.dot(U_s[a, :], V_s[b, :] * W_s[o, :])
                    res2 += (val - T[o, a, b]) ** 2
        res2 = np.sqrt(res2)
        
        if res2 < 1e-6:
            # Convert to list format
            U_list = [[U_s[i, j] for j in range(r)] for i in range(np_inputA)]
            V_list = [[V_s[i, j] for j in range(r)] for i in range(np_inputB)]
            W_list = [[W_s[i, j] for j in range(r)] for i in range(np_output)]
            return (U_list, V_list, W_list)
        
        return None
    
    def als_decompose(r, max_iters=500, tol=1e-12, n_restarts=3):
        """ALS decomposition for given rank."""
        if time.time() > deadline - 10:
            return None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                return None
            
            U = np.random.randn(np_inputA, r)
            V = np.random.randn(np_inputB, r)
            W = np.random.randn(np_output, r)
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Update W
                for o in range(np_output):
                    G = np.array([np.outer(U[:, c], V[:, c]).flatten() for c in range(r)]).T
                    rhs = T[o].flatten()
                    try:
                        W[o, :] = np.linalg.lstsq(G, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Update U
                for a in range(np_inputA):
                    G = np.array([np.outer(W[:, c], V[:, c]).flatten() for c in range(r)]).T
                    rhs = T[:, a, :].flatten()
                    try:
                        U[a, :] = np.linalg.lstsq(G, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for b in range(np_inputB):
                    G = np.array([np.outer(W[:, c], U[:, c]).flatten() for c in range(r)]).T
                    rhs = T[:, :, b].flatten()
                    try:
                        V[b, :] = np.linalg.lstsq(G, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Check convergence
                residual = np.sum((np.einsum('ac,bc,oc->abc', U, V, W) - T) ** 2)
                if residual < tol:
                    result = check_and_snap(U, V, W, r)
                    if result is not None:
                        return result
        
        return None
    
    # Try ranks from lower bound
    for r in range(30, 61):
        if time.time() > deadline - 5:
            break
        
        result = als_decompose(r)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
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
