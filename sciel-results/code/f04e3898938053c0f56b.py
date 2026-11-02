import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Part 1: A_L (2x2) * B_T (2x9)
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            # Strassen
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        # Last col of B_T
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        # Part 2: A_R (2x1) * B_B (1x9)
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}

    def verify_fast(U, V, W):
        """Fast verification using numpy"""
        r = len(U[0])
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if val != 1:
                        return False
        return True

    def find_m223_decomposition(rank, max_trials=100, max_iters=1000):
        """
        Find a rank-r decomposition for 2x2x3 matrix multiplication.
        A is 2x2, B is 2x3, C is 2x3.
        U: 4 x rank
        V: 6 x rank
        W: 6 x rank
        """
        # Tensor T[i, j, k] = 1 if i==j (in A) and j==k (in B)? No.
        # T[i, j, k] such that C[i, k] = sum_j A[i, j] B[j, k]
        # A indices: i in 0..1, j in 0..1
        # B indices: j in 0..1, k in 0..2
        # C indices: i in 0..1, k in 0..2
        
        # Non-zero entries of T:
        # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
        # So T[i, j, k] = 1 for all i in 0..1, j in 0..1, k in 0..2.
        # Wait, the tensor definition is T_{i,j,k} = 1 if the product A_{i,j} B_{j,k} is part of the sum for C_{i,k}.
        # Yes, for matrix multiplication, T_{i,j,k} = 1 for all valid i,j,k.
        # Here i=0,1; j=0,1; k=0,1,2.
        # So T is a 2x2x3 tensor with all ones.
        
        T = np.ones((2, 2, 3))
        
        best_sol = None
        best_err = float('inf')
        
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                break
            
            # Initialize with random small integers
            U = np.random.randint(-2, 3, size=(4, rank))
            V = np.random.randint(-2, 3, size=(6, rank))
            W = np.random.randint(-2, 3, size=(6, rank))
            
            # Ensure some non-zeros
            if np.all(U == 0) or np.all(V == 0) or np.all(W == 0):
                continue
                
            err = np.sum((T - np.einsum('ic,jc,kc->ijk', U, V, W))**2)
            if err == 0:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            # Local search: block coordinate descent
            for it in range(max_iters):
                if time.time() > deadline - 20:
                    break
                    
                # Pick a random matrix to update
                mat_idx = random.randint(0, 2)
                
                if mat_idx == 0: # Update U
                    # For each column c of U, we have U[:, c] * (V[:, c] * W[:, c])
                    # Let M_c = V[:, c] * W[:, c] (element-wise product, shape 6)
                    # We want U[:, c] to satisfy sum_i U[i,c] * M_c[i, j, k] ... no.
                    # The error is sum_c U[i,c] V[j,c] W[k,c].
                    # For fixed V, W, the term for column c is U[:, c] * (V[:, c] * W[:, c])_reshaped?
                    # No, the tensor product is U[:, c] \otimes V[:, c] \otimes W[:, c].
                    # So for a fixed c, we are adding a rank 1 tensor.
                    # We can't easily update one column of U without knowing V, W.
                    # However, we can view the problem as:
                    # T = sum_c U_c \otimes V_c \otimes W_c
                    # Fix V_c, W_c for all c.
                    # Then T - sum_{c != c0} ... = U_{c0} \otimes V_{c0} \otimes W_{c0}
                    # This is a rank 1 approximation problem for the residual.
                    # Residual R = T - sum_{c != c0} U_c \otimes V_c \otimes W_c
                    # We want to find U_{c0} such that U_{c0} \otimes V_{c0} \otimes W_{c0} approx R.
                    # This is equivalent to finding a vector u such that u \otimes v \otimes w approx R.
                    # This is a tensor completion / low rank approximation problem.
                    # For rank 1, we can use SVD on the matricization.
                    
                    # Compute residual for column c0
                    c0 = random.randint(0, rank - 1)
                    
                    # Current contribution of all columns
                    # We need to subtract the contribution of c0
                    # This is expensive if done naively.
                    # Instead, let's just do a simple random perturbation and accept if it improves.
                    
                    # Random perturbation
                    row = random.randint(0, 3)
                    delta = random.choice([-1, 1])
                    U[row, c0] += delta
                    
                    new_err = np.sum((T - np.einsum('ic,jc,kc->ijk', U, V, W))**2)
                    if new_err < err:
                        err = new_err
                        if err == 0:
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    else:
                        U[row, c0] -= delta
                        
                elif mat_idx == 1: # Update V
                    c0 = random.randint(0, rank - 1)
                    row = random.randint(0, 5)
                    delta = random.choice([-1, 1])
                    V[row, c0] += delta
                    
                    new_err = np.sum((T - np.einsum('ic,jc,kc->ijk', U, V, W))**2)
                    if new_err < err:
                        err = new_err
                        if err == 0:
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    else:
                        V[row, c0] -= delta
                        
                else: # Update W
                    c0 = random.randint(0, rank - 1)
                    row = random.randint(0, 5)
                    delta = random.choice([-1, 1])
                    W[row, c0] += delta
                    
                    new_err = np.sum((T - np.einsum('ic,jc,kc->ijk', U, V, W))**2)
                    if new_err < err:
                        err = new_err
                        if err == 0:
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    else:
                        W[row, c0] -= delta
            
            if err < best_err:
                best_err = err
                best_sol = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
        return best_sol if best_err == 0 else None

    # Try to find a better solution
    best_result = None
    
    # Try ranks 8, 9, 10 for M223
    for r in [8, 9, 10]:
        if time.time() > deadline - 30:
            break
            
        # We need to find a decomposition for M223 of rank r
        # Then assemble the full solution
        # Full solution rank = 3 * r + 18
        
        # Note: The M223 decomposition found must be exact (err=0)
        # The search above returns None if err != 0
        m223_sol = find_m223_decomposition(r, max_trials=20, max_iters=500)
        
        if m223_sol:
            # Assemble full solution
            # U_full: 6 x (3*r + 18)
            # V_full: 27 x (3*r + 18)
            # W_full: 18 x (3*r + 18)
            
            total_rank = 3 * r + 18
            U_full = [[0] * total_rank for _ in range(6)]
            V_full = [[0] * total_rank for _ in range(27)]
            W_full = [[0] * total_rank for _ in range(18)]
            
            U_m223 = np.array(m223_sol["U"])
            V_m223 = np.array(m223_sol["V"])
            W_m223 = np.array(m223_sol["W"])
            
            # Place 3 copies of M223
            # M223 corresponds to A (2x2) * B (2x3) -> C (2x3)
            # In the full problem:
            # A is 2x3. We split into A1 (2x2) and A2 (2x1).
            # B is 3x9. We split into B1 (2x9) and B2 (1x9).
            # B1 is 2x9. We split B1 into three 2x3 blocks: B1_0, B1_1, B1_2.
            # So we use M223 for (A1, B1_0), (A1, B1_1), (A1, B1_2).
            
            for block in range(3):
                offset = block * r
                
                # U: rows 0-3 correspond to A1 (2x2)
                for i in range(4):
                    for c in range(r):
                        U_full[i][offset + c] = U_m223[i, c]
                        
                # V: rows 0-5 correspond to B1_block (2x3)
                # B1_block is B[0:2, block*3 : block*3+3]
                # V_m223 rows: 0-1 are B_row0, 2-5 are B_row1?
                # M223 V shape is 6x r.
                # B is 2x3. Indices (j, k) with j=0,1; k=0,1,2.
                # V[j, k] in M223 corresponds to V[j*p + k] in full?
                # In M223, V is flattened. Let's say V_m223[j*3+k, c].
                # In full V, the index is j*9 + (block*3 + k).
                for j in range(2):
                    for k in range(3):
                        v_idx_m223 = j * 3 + k
                        v_idx_full = j * 9 + (block * 3 + k)
                        for c in range(r):
                            V_full[v_idx_full][offset + c] = V_m223[v_idx_m223, c]
                            
                # W: rows 0-5 correspond to C_block (2x3)
                # C is 2x9. Indices (i, k) with i=0,1; k=0..8.
                # W_m223[i, k] corresponds to W[i, k] in M223.
                # In full W, the index is i*9 + (block*3 + k).
                # Wait, W in tensor decomposition is indexed by (i, k) of C.
                # W_full[i, k, c] = W_m223[i, k, c].
                for i in range(2):
                    for k in range(3):
                        w_idx_m223 = i * 3 + k
                        w_idx_full = i * 9 + (block * 3 + k)
                        for c in range(r):
                            W_full[w_idx_full][offset + c] = W_m223[w_idx_m223, c]
                            
            # Add M219 part (rank 18)
            offset_m219 = 3 * r
            # A2 is 2x1. Rows 4,5 of U.
            # B2 is 1x9. Rows 18-26 of V.
            # C is 2x9. All rows of W.
            
            # U: rows 4,5
            # V: rows 18-26
            # W: rows 0-17
            
            for k in range(9):
                # Term k: A[0,2] * B[2,k] -> C[0,k]
                # U[4, offset+k] = 1
                # V[18+k, offset+k] = 1
                # W[k, offset+k] = 1
                U_full[4][offset_m219 + k] = 1
                V_full[18 + k][offset_m219 + k] = 1
                W_full[k][offset_m219 + k] = 1
                
                # Term k+9: A[1,2] * B[2,k] -> C[1,k]
                # U[5, offset+k+9] = 1
                # V[18+k, offset+k+9] = 1
                # W[9+k, offset+k+9] = 1
                U_full[5][offset_m219 + 9 + k] = 1
                V_full[18 + k][offset_m219 + 9 + k] = 1
                W_full[9 + k][offset_m219 + 9 + k] = 1

            # Verify
            if verify_fast(U_full, V_full, W_full):
                best_result = {"U": U_full, "V": V_full, "W": W_full}
                break
                
    if best_result:
        return best_result
        
    # Fallback
    return current_solution()
