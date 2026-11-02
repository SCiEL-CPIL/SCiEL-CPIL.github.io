"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p  # 96
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return mat.tolist()
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    np.random.seed(42)
    
    # Try to find a lower-rank decomposition using coordinate descent
    for target_rank in [95, 94, 93, 92, 91, 90, 88, 84, 80, 72, 64, 56, 48, 40, 36, 32]:
        if time.time() > deadline - 25:
            break
        
        found = False
        for trial in range(3):
            if time.time() > deadline - 25:
                break
            
            # Initialize from truncated schoolbook + small random perturbation
            if target_rank <= rank_sb:
                U = U_sb[:, :target_rank].copy()
                V = V_sb[:, :target_rank].copy()
                W = W_sb[:, :target_rank].copy()
            else:
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
            
            # Add small random perturbation to break symmetry
            U += np.random.randn(n * m, target_rank) * 0.01
            V += np.random.randn(m * p, target_rank) * 0.01
            W += np.random.randn(p * n, target_rank) * 0.01
            
            # Coordinate descent: fix two matrices, optimize the third
            for iteration in range(200):
                if time.time() > deadline - 25:
                    break
                
                # Optimize U: for each (i,j), solve linear system
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b_vec = np.ones(p)
                        U_row, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U[i * m + j, :] = U_row
                
                # Optimize V: for each (j,k), solve linear system
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b_vec = np.ones(n)
                        V_row, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V[j * p + k, :] = V_row
                
                # Optimize W: for each (k,i), solve linear system
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b_vec = np.ones(m)
                        W_row, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W[k * n + i, :] = W_row
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err = abs(val - 1)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-8:
                    break
            
            # Final verification
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err = abs(val - 1)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-6:
                # Try to round to half-integers for exact representation
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                max_err_rounded = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_rounded[i * m + j, :], V_rounded[j * p + k, :] * W_rounded[k * n + i, :])
                            err = abs(val - 1)
                            if err > max_err_rounded:
                                max_err_rounded = err
                
                if max_err_rounded < 1e-6:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    found = True
                    break
        
        if found:
            break
    
    # Return the best found
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
