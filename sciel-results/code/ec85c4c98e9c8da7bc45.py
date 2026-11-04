import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook fallback
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(), 
        "V": V_school.tolist(), 
        "W": W_school.tolist()
    }
    best_rank = rank_school

    def is_valid_half_integer_solution(U, V, W):
        # Check if all entries are integers or half-integers
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        # Verify tensor identity
        # T_ijk = sum_c U_ic V_jc W_kc
        # We want T_ijk = 1 for all i,j,k
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # Compute tensor T
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Note the indices: U depends on i,j. V on j,k. W on k,i.
        # np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        if not np.allclose(T, 1.0, atol=1e-7):
            return False
        return True

    start_time = time.time()
    deadline = 140  # Keep safety margin

    # Start from a reasonable lower bound. 
    # For 2x3x9, rank >= max(2*3, 3*9, 9*2) is not a strict lower bound but a heuristic.
    # Actually rank >= max(n, m, p) is true.
    # Let's start from 27 (mp) as in the original, but maybe skip to 30?
    # The original started at 27. Let's stick to 27 to be safe, but speed up the solver.
    
    for rank in range(27, rank_school):
        if time.time() - start_time > deadline:
            break
        
        found = False
        # Increase attempts for smaller ranks? Or just keep it fixed but faster.
        num_attempts = 100 if rank < 35 else 50
        
        for attempt in range(num_attempts):
            if time.time() - start_time > deadline:
                break
            
            np.random.seed(rank * 1000 + attempt)
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Reshape for easier access
            # U: (n, m, R)
            # V: (m, p, R)
            # W: (p, n, R)
            U_reshaped = U.reshape(n, m, rank)
            V_reshaped = V.reshape(m, p, rank)
            W_reshaped = W.reshape(p, n, rank)
            
            # Alternating Least Squares
            for iteration in range(100):
                if time.time() - start_time > deadline:
                    break
                
                # Update U
                # For each i, j:
                # Solve (V_jk * W_ki) * u_ij = 1_k
                # Matrix M_ij (p x R) where M_ij[k, :] = V_reshaped[j, k, :] * W_reshaped[k, i, :]
                for i in range(n):
                    for j in range(m):
                        # Construct M_ij
                        # V_reshaped[j, :, :] is (p, R)
                        # W_reshaped[:, i, :] is (p, R)
                        M_ij = V_reshaped[j, :, :] * W_reshaped[:, i, :]
                        b = np.ones(p)
                        u_row, _, _, _ = np.linalg.lstsq(M_ij, b, rcond=None)
                        U_reshaped[i, j, :] = u_row
                
                # Update V
                # For each j, k:
                # Solve (U_ij * W_ki) * v_jk = 1_i
                # Matrix N_jk (n x R) where N_jk[i, :] = U_reshaped[i, j, :] * W_reshaped[k, i, :]
                for j in range(m):
                    for k in range(p):
                        # U_reshaped[:, j, :] is (n, R)
                        # W_reshaped[k, :, :] is (n, R)
                        N_jk = U_reshaped[:, j, :] * W_reshaped[k, :, :]
                        b = np.ones(n)
                        v_row, _, _, _ = np.linalg.lstsq(N_jk, b, rcond=None)
                        V_reshaped[j, k, :] = v_row
                        
                # Update W
                # For each k, i:
                # Solve (U_ij * V_jk) * w_ki = 1_j
                # Matrix O_ki (m x R) where O_ki[j, :] = U_reshaped[i, j, :] * V_reshaped[j, k, :]
                for k in range(p):
                    for i in range(n):
                        # U_reshaped[i, :, :] is (m, R)
                        # V_reshaped[:, k, :] is (m, R)
                        O_ki = U_reshaped[i, :, :] * V_reshaped[:, k, :]
                        b = np.ones(m)
                        w_row, _, _, _ = np.linalg.lstsq(O_ki, b, rcond=None)
                        W_reshaped[k, i, :] = w_row
                
                # Check convergence
                # T = einsum
                T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
                residual = np.sum((T - 1.0) ** 2)
                
                if residual < 1e-10:
                    # Round to nearest half-integer
                    U_rounded = np.round(U * 2) / 2.0
                    V_rounded = np.round(V * 2) / 2.0
                    W_rounded = np.round(W * 2) / 2.0
                    
                    # Check validity of rounded solution
                    if is_valid_half_integer_solution(U_rounded, V_rounded, W_rounded):
                        best_result = {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        best_rank = rank
                        found = True
                        break
                    else:
                        # If rounding failed, we might have found a real solution but not a half-integer one.
                        # The problem requires half-integer entries.
                        # We should probably break or continue? 
                        # If we found a low rank real solution, maybe we can try to find a half-integer one?
                        # Or just accept that this rank doesn't have a half-integer solution easily.
                        # Let's break to try next attempt/rank.
                        break
            
            if found:
                break
        
        if found:
            break
    
    return best_result
