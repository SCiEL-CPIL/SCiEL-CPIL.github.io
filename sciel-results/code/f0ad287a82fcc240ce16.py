import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for u_idx in range(n * m):
        for v_idx in range(m * p):
            for w_idx in range(p * n):
                val = sum(U[u_idx][r] * V[v_idx][r] * W[w_idx][r] for r in range(rank))
                i = u_idx // m
                j = u_idx % m
                j2 = v_idx // p
                k = v_idx % p
                k2 = w_idx // n
                i2 = w_idx % n
                if i == i2 and j == j2 and k == k2:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Fallback: Strassen rank-42 decomposition
    def strassen_decomposition():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        U_strassen = [
            {(0, 0): 1, (1, 1): 1},
            {(1, 0): 1, (1, 1): 1},
            {(0, 0): 1},
            {(1, 1): 1},
            {(0, 0): 1, (0, 1): 1},
            {(1, 0): 1, (0, 0): -1},
            {(0, 1): 1, (1, 1): -1}
        ]
        V_strassen = [
            {(0, 0): 1, (1, 1): 1},
            {(0, 0): 1},
            {(0, 1): 1, (1, 1): -1},
            {(1, 0): 1, (0, 0): -1},
            {(1, 1): 1},
            {(0, 0): 1, (0, 1): 1},
            {(1, 0): 1, (1, 1): 1}
        ]
        W_strassen = [
            {(0, 0): 1, (1, 1): 1},
            {(0, 1): 1, (1, 1): -1},
            {(1, 0): 1, (1, 1): 1},
            {(0, 0): 1, (0, 1): 1},
            {(0, 0): -1, (1, 0): 1},
            {(1, 1): 1},
            {(0, 0): 1}
        ]
        
        col = 0
        j_groups = [(0, 1), (2, 3)]
        k_groups = [(0, 1), (2, 3), (4, 5)]
        
        for jg in j_groups:
            j_base = jg[0]
            for kg in k_groups:
                k_base = kg[0]
                for s in range(7):
                    for (i_loc, j_loc), coeff in U_strassen[s].items():
                        i = i_loc
                        j = j_base + j_loc
                        u_idx = i * m + j
                        U[u_idx][col] += coeff
                    for (j_loc, k_loc), coeff in V_strassen[s].items():
                        j = j_base + j_loc
                        k = k_base + k_loc
                        v_idx = j * p + k
                        V[v_idx][col] += coeff
                    for (k_loc, i_loc), coeff in W_strassen[s].items():
                        k = k_base + k_loc
                        i = i_loc
                        w_idx = k * n + i
                        W[w_idx][col] += coeff
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try to find lower rank via optimization
    best_result = None
    
    for target_rank in range(34, 41):  # Try 34 to 40
        if time.time() > deadline - 30:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(trial * 7919 + target_rank * 31)
            
            # Initialize
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Iterative refinement
            for it in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Update W: for each (i,j), solve for W
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if consistent
                # This is complex; use gradient descent instead
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            expected = 1.0
                            err += (val - expected) ** 2
                
                # Simple gradient step
                lr = 0.01
                for r in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for rr in range(target_rank):
                                    val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                                grad_U = V[j*p+k, r] * W[k*n+i, r] * (val - 1.0)
                                grad_V = U[i*m+j, r] * W[k*n+i, r] * (val - 1.0)
                                grad_W = U[i*m+j, r] * V[j*p+k, r] * (val - 1.0)
                                U[i*m+j, r] -= lr * grad_U
                                V[j*p+k, r] -= lr * grad_V
                                W[k*n+i, r] -= lr * grad_W
                
                if err < 1e-10:
                    found = True
                    break
            
            if found:
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                U_list = [[float(U_rounded[i, r]) for r in range(target_rank)] for i in range(n*m)]
                V_list = [[float(V_rounded[i, r]) for r in range(target_rank)] for i in range(m*p)]
                W_list = [[float(W_rounded[i, r]) for r in range(target_rank)] for i in range(p*n)]
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
                else:
                    found = False
        
        if found and best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to Strassen
    result = strassen_decomposition()
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        raise ValueError("Invalid decomposition")
    return result
