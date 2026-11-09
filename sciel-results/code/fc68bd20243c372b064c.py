import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Precompute the target tensor entries
    # T[(i,j), (j',k), (i',k')] = delta(i,i')*delta(j,j')*delta(k,k')
    
    def solve_for_U(U, V, W, r):
        """Solve for U given V, W"""
        U_new = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                # Build A matrix: A[alpha, c] = V[j'*p+k, c] * W[k'*n+i', c]
                # alpha = ((j'*p + k)*p + k')*n + i'
                A = np.zeros((m * p * p * n, r))
                b = np.zeros(m * p * p * n)
                idx = 0
                for jp in range(m):  # j'
                    for k in range(p):
                        for kp in range(p):  # k'
                            for ip in range(n):  # i'
                                A[idx, :] = V[jp * p + k, :] * W[kp * n + ip, :]
                                if ip == i and jp == j and k == k:
                                    b[idx] = 1.0
                                idx += 1
                # Solve least squares
                U_new[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
        return U_new
    
    def solve_for_V(U, V, W, r):
        """Solve for V given U, W"""
        V_new = np.zeros((m * p, r))
        for jp in range(m):
            for k in range(p):
                A = np.zeros((n * m * p * n, r))
                b = np.zeros(n * m * p * n)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for kp in range(p):  # k'
                            for ip in range(n):  # i'
                                A[idx, :] = U[i * m + j, :] * W[kp * n + ip, :]
                                if ip == i and j == jp and kp == k:
                                    b[idx] = 1.0
                                idx += 1
                V_new[jp * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
        return V_new
    
    def solve_for_W(U, V, W, r):
        """Solve for W given U, V"""
        W_new = np.zeros((p * n, r))
        for kp in range(p):
            for ip in range(n):
                A = np.zeros((n * m * m * p, r))
                b = np.zeros(n * m * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                A[idx, :] = U[i * m + j, :] * V[jp * p + k, :]
                                if ip == i and jp == j and k == kp:
                                    b[idx] = 1.0
                                idx += 1
                W_new[kp * n + ip, :] = np.linalg.lstsq(A, b, rcond=None)[0]
        return W_new
    
    def compute_residual(U, V, W, r):
        """Compute the residual of the tensor equation"""
        max_res = 0.0
        for i in range(n):
            for ip in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                val = 0.0
                                for c in range(r):
                                    val += U[i * m + j, c] * V[jp * p + k, c] * W[kp * n + ip, c]
                                target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                res = abs(val - target)
                                if res > max_res:
                                    max_res = res
        return max_res
    
    def try_rank(r, max_iters=50, tol=1e-8):
        """Try to find a decomposition of rank r"""
        if time.time() > deadline:
            return None
        
        np.random.seed(42)
        best_U, best_V, best_W = None, None, None
        best_res = float('inf')
        
        for attempt in range(3):
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                U = solve_for_U(U, V, W, r)
                V = solve_for_V(U, V, W, r)
                W = solve_for_W(U, V, W, r)
                
                if it % 10 == 9:
                    res = compute_residual(U, V, W, r)
                    if res < best_res:
                        best_res = res
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if res < tol:
                        break
            
            res = compute_residual(U, V, W, r)
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_res < tol:
                break
        
        if best_res < tol:
            return best_U, best_V, best_W
        return None
    
    # Try to find a lower rank decomposition
    # Lower bound is at least max(n*m, m*p, n*p) = 24
    # Try from 30 to 47
    best_result = None
    best_rank = schoolbook_rank
    
    for r in range(30, schoolbook_rank):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break  # Found the lowest rank, stop
    
    # If no lower rank found, use schoolbook
    if best_result is None and best_rank == schoolbook_rank:
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
