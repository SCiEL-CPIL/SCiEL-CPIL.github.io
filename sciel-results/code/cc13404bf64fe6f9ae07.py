"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Strategy: Start with rank = schoolbook_rank - 1 and try to solve the linear system
    # The tensor entries T_{i,j,k} = 1 for all valid i,j,k
    # We want U @ V @ W^T to equal the tensor
    
    # For each candidate rank, we can try to set up the problem as a rank-constrained matrix completion
    # or use a greedy/iterative approach.
    
    # Given the small size (3,3,6), let's try a direct search for rank < 54
    # We'll use a heuristic: randomly initialize and use gradient-free optimization (simulated annealing)
    # to minimize the residual ||T - U*V*W^T||_F
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Start with schoolbook as fallback
    U0 = [[0] * schoolbook_rank for _ in range(n * m)]
    V0 = [[0] * schoolbook_rank for _ in range(m * p)]
    W0 = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][column] = 1
                V0[j * p + k][column] = 1
                W0[k * n + i][column] = 1
                column += 1
    best_result = {"U": U0, "V": V0, "W": W0}
    
    # Build the target tensor as a 3D array
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # We want to find rank r such that T = sum_c U[:, :, c] * V[:, :, c] * W[:, :, c]
    # where U is (n,m,r), V is (m,p,r), W is (p,n,r)
    # Note the convention: W[k*n+i, c] corresponds to W_3d[i, k, c]
    
    # Let's try ranks from 53 down to, say, 40
    # For each rank, do a few random restarts with simulated annealing
    
    for target_rank in range(schoolbook_rank - 1, 35, -1):
        if time.time() > deadline - 10.0:
            break
        
        found = False
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization
            U = np.random.randn(n, m, target_rank) * 0.1
            V = np.random.randn(m, p, target_rank) * 0.1
            W = np.random.randn(p, n, target_rank) * 0.1
            
            # Simulated annealing
            T_cur = 1.0
            T_min = 1e-12
            alpha = 0.995
            max_iter = 5000
            
            best_local_U, best_local_V, best_local_W = None, None, None
            best_local_err = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 5.0:
                    break
                
                # Random perturbation
                c = np.random.randint(target_rank)
                dU = np.random.randn(n, m) * T_cur
                dV = np.random.randn(m, p) * T_cur
                dW = np.random.randn(p, n) * T_cur
                
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                U_new[:, :, c] += dU
                V_new[:, :, c] += dV
                W_new[:, :, c] += dW
                
                # Compute error
                def compute_err(Um, Vm, Wm):
                    err = 0.0
                    for c_idx in range(Um.shape[2]):
                        T_part = np.tensordot(Um[:, :, c_idx], Vm[:, :, c_idx], axes=([1], [0]))
                        T_part = T_part * Wm[:, :, c_idx]
                        err += np.sum((T - T_part)**2)
                    return err
                
                err_new = compute_err(U_new, V_new, W_new)
                err_old = compute_err(U, V, W)
                
                delta_err = err_new - err_old
                if delta_err < 0 or np.random.rand() < np.exp(-delta_err / max(T_cur, 1e-10)):
                    U, V, W = U_new, V_new, W_new
                    if err_new < best_local_err:
                        best_local_err = err_new
                        best_local_U, best_local_V, best_local_W = U_new.copy(), V_new.copy(), W_new.copy()
                
                T_cur *= alpha
                if T_cur < T_min:
                    T_cur = T_min
            
            if best_local_err < 1e-10:
                # Found a valid decomposition!
                found = True
                break
        
        if found:
            best_rank = target_rank
            # Convert to list format
            U_list = [[0] * target_rank for _ in range(n * m)]
            V_list = [[0] * target_rank for _ in range(m * p)]
            W_list = [[0] * target_rank for _ in range(p * n)]
            
            for i in range(n):
                for j in range(m):
                    for c in range(target_rank):
                        U_list[i * m + j][c] = best_local_U[i, j, c]
            
            for j in range(m):
                for k in range(p):
                    for c in range(target_rank):
                        V_list[j * p + k][c] = best_local_V[j, k, c]
            
            for k in range(p):
                for i in range(n):
                    for c in range(target_rank):
                        W_list[k * n + i][c] = best_local_W[k, i, c]
            
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    return best_result
# EVOLVE-BLOCK-END
