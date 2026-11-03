import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def build_rank55():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0 (2 x 10 = 20 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        
        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
        
        return U, V, W, rank
    
    def verify(U, V, W, r, tol=1e-9):
        # Vectorized verification
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Target: T[i, j, k] = 1
        # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Reshape U to (n, m, r)
        U_reshaped = U.reshape(n, m, r)
        # Reshape V to (m, p, r)
        V_reshaped = V.reshape(m, p, r)
        # Reshape W to (p, n, r)
        W_reshaped = W.reshape(p, n, r)
        
        # Compute contraction
        # We want C[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # Intermediate: M[j, k, i, r] ? No.
        # Let's do:
        # Temp1[j, k, r] = sum_i ? No.
        
        # Efficient way:
        # Contract U and V over j? No, j is free in output.
        # Contract V and W over k? No, k is free in output.
        
        # Let's compute A[i, j, k]
        # A[i, j, k] = sum_r U[i,j,r] * (V[j,:,r] * W[:,i,r])_k ?
        
        # Vectorized:
        # 1. Multiply V and W element-wise after aligning indices?
        # V is (m, p, r). W is (p, n, r).
        # We need sum_r V[j, k, r] * W[k, i, r].
        # Let's compute M[j, k, i] = sum_r V[j, k, r] * W[k, i, r].
        
        # V[j, :, :] * W[:, i, :] -> (p, r) for fixed j, i.
        # This is a batched multiplication.
        
        # M[j, k, i] = einsum('jkr,kr->jki', V_reshaped, W_reshaped)
        M_jki = np.einsum('jkr,kr->jki', V_reshaped, W_reshaped)
        
        # Now C[i, j, k] = sum_r U[i, j, r] * M[j, k, i] ?
        # Wait, M depends on j, k, i.
        # C[i, j, k] = sum_r U[i, j, r] * (sum_r' V[j, k, r'] W[k, i, r']) ?
        # No, the rank index r must be the same for U, V, W.
        # So C[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r].
        
        # We can compute:
        # Temp[j, k, r] = V[j, k, r]
        # Temp2[k, i, r] = W[k, i, r]
        # Product[j, k, i, r] = V[j, k, r] * W[k, i, r]
        # C[i, j, k] = sum_r U[i, j, r] * Product[j, k, i, r]
        
        # einsum: 'jkr,kr,ijr->ijk'
        # V_reshaped: jkr
        # W_reshaped: kr (needs to be aligned with i? No, W is kr i? No W is (p, n, r) -> k, i, r)
        # U_reshaped: i, j, r
        
        # C = einsum('jkr,kr,ijr->ijk', V_reshaped, W_reshaped, U_reshaped)
        # Wait, W_reshaped is (p, n, r). Indices are k, i, r.
        # So 'jkr,kr,ijr->ijk' is correct if W is indexed k, i, r.
        
        C = np.einsum('jkr,kr,ijr->ijk', V_reshaped, W_reshaped, U_reshaped)
        
        err = np.max(np.abs(C - 1.0))
        return err < tol, err

    # Start with known rank-55 solution
    U_best, V_best, W_best, rank_best = build_rank55()
    valid, err = verify(U_best, V_best, W_best, rank_best)
    if not valid:
        # Fallback, though this should never happen
        pass

    # Try to reduce rank
    # We iterate from rank_best - 1 downwards
    for target_rank in range(rank_best - 1, 45, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        # More restarts for higher target ranks
        max_restarts = 20 if target_rank >= 50 else 10
        
        for trial in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(trial * 7919 + target_rank * 13)
            
            # Initialize near rank-55 solution
            # We take the first 'target_rank' columns of the rank-55 solution
            # This preserves the structure
            U_init = U_best[:, :target_rank].copy()
            V_init = V_best[:, :target_rank].copy()
            W_init = W_best[:, :target_rank].copy()
            
            # Add small noise to escape local minima
            U_init += 0.01 * np.random.randn(nm, target_rank)
            V_init += 0.01 * np.random.randn(mp, target_rank)
            W_init += 0.01 * np.random.randn(pn, target_rank)
            
            U_cur, V_cur, W_cur = U_init, V_init, W_init
            best_err = float('inf')
            
            # ALS Optimization
            for iteration in range(300):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                for idx in range(nm):
                    i, j = divmod(idx, m)
                    A_mat = np.zeros((p, target_rank))
                    for k in range(p):
                        A_mat[k, :] = V_cur[j*p + k, :] * W_cur[k*n + i, :]
                    b_vec = np.ones(p)
                    u_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U_cur[idx, :] = u_sol
                    
                # Update V
                for idx in range(mp):
                    j, k = divmod(idx, p)
                    A_mat = np.zeros((n, target_rank))
                    for i in range(n):
                        A_mat[i, :] = U_cur[i*m + j, :] * W_cur[k*n + i, :]
                    b_vec = np.ones(n)
                    v_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V_cur[idx, :] = v_sol
                    
                # Update W
                for idx in range(pn):
                    k, i = divmod(idx, n)
                    A_mat = np.zeros((m, target_rank))
                    for j in range(m):
                        A_mat[j, :] = U_cur[i*m + j, :] * V_cur[j*p + k, :]
                    b_vec = np.ones(m)
                    w_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_cur[idx, :] = w_sol
                    
                # Check error
                valid, err_val = verify(U_cur, V_cur, W_cur, target_rank)
                
                if err_val < best_err:
                    best_err = err_val
                    if err_val < 1e-9:
                        found = True
                        break
                        
                if found:
                    break
            
            if found:
                U_best, V_best, W_best, rank_best = U_cur, V_cur, W_cur, target_rank
                break
                
    # Final verification
    valid, err = verify(U_best, V_best, W_best, rank_best)
    if not valid:
        # If optimization failed, return the initial valid rank-55 solution
        U_best, V_best, W_best, rank_best = build_rank55()
        valid, err = verify(U_best, V_best, W_best, rank_best)
        
    # Convert to lists
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
