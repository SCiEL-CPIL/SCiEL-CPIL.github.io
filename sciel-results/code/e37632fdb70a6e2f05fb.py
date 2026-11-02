import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = i * m + j
                    v_row = j * p + k
                    w_row = k * n + i
                    val = np.dot(U[u_row, :], V[v_row, :] * W[w_row, :])
                    if abs(val - 1) > 1e-6:
                        return False
        return True
    
    best_U, best_V, best_W = U_schoolbook, V_schoolbook, W_schoolbook
    best_rank = rank_schoolbook
    
    # Try to find a lower-rank solution
    # Strategy: Use rank-23 decomposition for 3x3x3, schoolbook for remaining 5 columns
    # Total rank: 23 + 3*3*5 = 23 + 45 = 68
    
    if time.time() < deadline - 30:
        # Construct rank-68 solution
        rank_68 = 23 + n * m * 5  # 23 + 45 = 68
        
        # For the 3x3x3 part (k=0,1,2), use rank-23 decomposition
        # For the 3x3x5 part (k=3,4,5,6,7), use schoolbook
        
        # I need the rank-23 decomposition for 3x3x3
        # This is complex, so let me try a different approach
        
        # Alternative: Try random search for rank < 72
        for target_rank in [68, 69, 70, 71]:
            if time.time() > deadline - 30:
                break
            
            found = False
            for seed in range(20):
                if time.time() > deadline - 30:
                    break
                
                np.random.seed(seed + target_rank * 1000)
                U = np.random.randn(n*m, target_rank)
                V = np.random.randn(m*p, target_rank)
                W = np.random.randn(p*n, target_rank)
                
                # Alternating minimization
                for iteration in range(100):
                    if time.time() > deadline - 30:
                        break
                    
                    # Solve for W (linear in W)
                    # For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # This is linear in W[k*n+i, r] for fixed (i,j,k)
                    
                    # Build the linear system for W
                    num_constraints = n * m * p
                    A_w = np.zeros((num_constraints, p*n*target_rank))
                    b_w = np.zeros(num_constraints)
                    
                    eq_idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                b_w[eq_idx] = 1.0
                                for r in range(target_rank):
                                    A_w[eq_idx, (k*n+i)*target_rank + r] = U[i*m+j, r] * V[j*p+k, r]
                                eq_idx += 1
                    
                    w_sol, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                    W = w_sol.reshape(p*n, target_rank)
                    
                    # Solve for U (linear in U)
                    A_u = np.zeros((num_constraints, n*m*target_rank))
                    b_u = np.zeros(num_constraints)
                    
                    eq_idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                b_u[eq_idx] = 1.0
                                for r in range(target_rank):
                                    A_u[eq_idx, (i*m+j)*target_rank + r] = V[j*p+k, r] * W[k*n+i, r]
                                eq_idx += 1
                    
                    u_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                    U = u_sol.reshape(n*m, target_rank)
                    
                    # Solve for V (linear in V)
                    A_v = np.zeros((num_constraints, m*p*target_rank))
                    b_v = np.zeros(num_constraints)
                    
                    eq_idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                b_v[eq_idx] = 1.0
                                for r in range(target_rank):
                                    A_v[eq_idx, (j*p+k)*target_rank + r] = U[i*m+j, r] * W[k*n+i, r]
                                eq_idx += 1
                    
                    v_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                    V = v_sol.reshape(m*p, target_rank)
                    
                    # Check error
                    error = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                                error += (val - 1.0) ** 2
                    
                    if error < 1e-10:
                        # Try rounding to half-integers
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        
                        error_r = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                                    error_r += (val - 1.0) ** 2
                        
                        if error_r < 1e-10:
                            best_rank = target_rank
                            best_U, best_V, best_W = U_r, V_r, W_r
                            found = True
                            break
                
                if found:
                    break
            
            if found:
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
