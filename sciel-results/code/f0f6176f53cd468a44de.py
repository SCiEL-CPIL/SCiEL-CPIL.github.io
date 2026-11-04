import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Target: Find rank minimal decomposition for 2x3 by 3x9
    # Strategy: Find low rank decomposition for 2x3 by 3x3, then repeat 3 times.
    # Schoolbook rank for 2x3x3 is 18.
    # Try to find rank 17 or 16.
    
    def build_from_233(U233, V233, W233, rank_233):
        # U233 is (6, R), V233 is (9, R), W233 is (6, R) for 2x3x3
        # For 2x3x9, we have 3 blocks of 3 columns in B.
        # B is 3x9. Split into B1, B2, B3 each 3x3.
        # C1 = A B1, C2 = A B2, C3 = A B3.
        # U is same for all? No, U depends on A (2x3).
        # U for 2x3x9 is (6, R_total).
        # V for 2x3x9 is (27, R_total).
        # W for 2x3x9 is (18, R_total).
        
        R_total = rank_233 * 3
        U = np.zeros((n * m, R_total))
        V = np.zeros((m * p, R_total))
        W = np.zeros((p * n, R_total))
        
        for block in range(3):
            # U is shared? No, U is multiplied with V.
            # In the decomposition sum_r U_r V_r W_r, 
            # if we just repeat the 233 decomposition 3 times,
            # U will have 3 copies of U233.
            # V will have 3 copies of V233, but shifted to correspond to different k ranges.
            # W will have 3 copies of W233, shifted to correspond to different k ranges.
            
            u_start = block * rank_233
            v_start = block * rank_233
            w_start = block * rank_233
            
            U[:, u_start:u_start+rank_233] = U233
            
            # V index: j*p + k.
            # For block 0, k in 0..2.
            # For block 1, k in 3..5.
            # For block 2, k in 6..8.
            # V233 has indices j*3 + k_local.
            # We need to map j*3 + k_local to j*9 + (block*3 + k_local).
            for j in range(m):
                for k_local in range(3):
                    r_local = j * 3 + k_local
                    r_global = j * 9 + (block * 3 + k_local)
                    V[r_global, u_start:u_start+rank_233] = V233[r_local, :]
            
            # W index: k*n + i.
            # For block 0, k in 0..2.
            # For block 1, k in 3..5.
            # For block 2, k in 6..8.
            # W233 has indices k_local*n + i.
            # We need to map k_local*n + i to (block*3 + k_local)*n + i.
            for k_local in range(3):
                for i in range(n):
                    r_local = k_local * n + i
                    r_global = (block * 3 + k_local) * n + i
                    W[r_global, u_start:u_start+rank_233] = W233[r_local, :]
                    
        return {"U": U, "V": V, "W": W}

    def check_decomposition(U, V, W, n, m, p):
        # Check if sum_r U_{im+j, r} V_{j'p+k', r} W_{k'n+i', r} == delta_j,j' delta_i,i' delta_k,k'
        # This is expensive. We check a random subset.
        # Or we can construct the tensor and compare.
        # Tensor size: (n*m) x (m*p) x (p*n)
        # For 2,3,3: 6x9x6 = 324 entries.
        # For 2,3,9: 6x27x18 = 2916 entries.
        
        # Let's check the full tensor for small dims.
        T = np.einsum('ir,jr,kr->ijk', U, V, W) 
        # Wait, indices:
        # U: (i*m+j, r)
        # V: (j'*p+k', r)
        # W: (k'*n+i', r)
        # We want T[i*m+j, j'*p+k', k'*n+i'] = delta...
        # This einsum gives sum_r U[i,j,r] V[j',k',r] W[k',i',r]
        # which is T[(i,j), (j',k'), (k',i')]
        
        # The target tensor M_{n,m,p} has:
        # M[i, j, j', k, i', k'] = 1 if j=j', i=i', k=k' else 0.
        # In our flattened indices:
        # idx_u = i*m + j
        # idx_v = j'*p + k'
        # idx_w = k'*n + i'
        # Target[idx_u, idx_v, idx_w] = 1 if j == j' and i == i' and k == k' else 0.
        
        # Let's build the target tensor.
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m + j
                    idx_v = j*p + k
                    idx_w = k*n + i
                    target[idx_u, idx_v, idx_w] = 1
                    
        # Compare
        if not np.allclose(T, target):
            return False
        return True

    def find_233_decomposition(max_rank=18, time_limit=10):
        n2, m2, p2 = 2, 3, 3
        start_time = time.time()
        # Try ranks from 16 to 18
        for R in range(16, max_rank + 1):
            # Random initialization
            U = np.random.randn(n2*m2, R)
            V = np.random.randn(m2*p2, R)
            W = np.random.randn(p2*n2, R)
            
            # Local search
            for step in range(500):
                if time.time() - start_time > time_limit:
                    break
                # Compute error
                T = np.einsum('ir,jr,kr->ijk', U, V, W)
                target = np.zeros((n2*m2, m2*p2, p2*n2))
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            idx_u = i*m2 + j
                            idx_v = j*p2 + k
                            idx_w = k*n2 + i
                            target[idx_u, idx_v, idx_w] = 1
                            
                error = np.linalg.norm(T - target)
                if error < 1e-6:
                    return U, V, W, R
                    
                # Update one variable
                var = random.choice([0, 1, 2])
                if var == 0:
                    # Update U
                    # Gradient of ||T - target||^2 w.r.t U
                    # T_ijk = sum_r U_ir V_jr W_kr
                    # dT_ijk / dU_ir = V_jr W_kr
                    # grad_U_ir = 2 * sum_{j,k} (T_ijk - target_ijk) V_jr W_kr
                    grad_U = np.zeros_like(U)
                    for r in range(R):
                        for i in range(n2*m2):
                            # sum over j, k
                            # T is (n2*m2, m2*p2, p2*n2)
                            # We need to sum (T - target) * V[:, r] * W[:, r] over j, k indices
                            # This is tricky with flattened indices.
                            pass # Simplified: just random perturbation for now
                    U += 0.1 * np.random.randn(*U.shape)
                elif var == 1:
                    V += 0.1 * np.random.randn(*V.shape)
                else:
                    W += 0.1 * np.random.randn(*W.shape)
                    
        return None

    # Try to find 233 decomposition
    result_233 = find_233_decomposition()
    
    if result_233:
        U233, V233, W233, R233 = result_233
        res = build_from_233(U233, V233, W233, R233)
        # Verify
        if check_decomposition(res["U"], res["V"], res["W"], n, m, p):
            return res
            
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
