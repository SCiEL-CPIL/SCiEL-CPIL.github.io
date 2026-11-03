import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = 2, 3, 6
    
    # Build target tensor T for 2x3x6
    # T[i*m+j, j*p+k, k*n+i] = 1.0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Strategy: Decompose 2x3x6 into two 2x3x3 blocks
    # The 2x3x6 multiplication tensor can be seen as two independent 2x3x3 multiplications
    # Block 1: k in {0,1,2}, Block 2: k in {3,4,5}
    #
    # For each block, we need a rank-r decomposition of the 2x3x3 multiplication tensor
    # The 2x3x3 multiplication tensor has rank 15 (proven by Landsberg, 2010)
    # So the full 2x3x6 tensor can be decomposed with rank 2*15 = 30
    
    n2, m2, p2 = 2, 3, 3
    T233 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T233[i*m2+j, j*p2+k, k*n2+i] = 1.0
    
    R233 = 15
    rows_u = n2 * m2  # 6
    rows_v = m2 * p2  # 9
    rows_w = p2 * n2  # 6
    
    # Use ALS to find a rank-15 decomposition of the 2x3x3 tensor
    # with integer/half-integer entries
    
    def generate_Z_VW(V, W, R):
        Z = np.zeros((R, rows_v * rows_w))
        for r in range(R):
            outer = np.outer(V[:, r], W[:, r])
            Z[r, :] = outer.flatten()
        return Z
    
    def generate_Z_UW(U, W, R):
        Z = np.zeros((R, rows_u * rows_w))
        for r in range(R):
            outer = np.outer(U[:, r], W[:, r])
            Z[r, :] = outer.flatten()
        return Z
    
    def generate_Z_UV(U, V, R):
        Z = np.zeros((R, rows_u * rows_v))
        for r in range(R):
            outer = np.outer(U[:, r], V[:, r])
            Z[r, :] = outer.flatten()
        return Z
    
    def solve_U(V, W, T_target, R):
        Z = generate_Z_VW(V, W, R)
        T_flat = T_target.flatten().reshape(rows_u, -1)
        try:
            U_sol, _, _, _ = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
            return U_sol.T
        except:
            return None
    
    def solve_V(U, W, T_target, R):
        Z = generate_Z_UW(U, W, R)
        T_flat = T_target.transpose(1, 0, 2).flatten().reshape(rows_v, -1)
        try:
            V_sol, _, _, _ = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
            return V_sol.T
        except:
            return None
    
    def solve_W(U, V, T_target, R):
        Z = generate_Z_UV(U, V, R)
        T_flat = T_target.transpose(2, 0, 1).flatten().reshape(rows_w, -1)
        try:
            W_sol, _, _, _ = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
            return W_sol.T
        except:
            return None
    
    def als_decompose(T_target, R, max_iter=200, tol=1e-10, seed=0):
        rng = np.random.RandomState(seed)
        U = rng.randn(rows_u, R) * 0.5
        V = rng.randn(rows_v, R) * 0.5
        W = rng.randn(rows_w, R) * 0.5
        
        for it in range(max_iter):
            U_new = solve_U(V, W, T_target, R)
            if U_new is None:
                return None
            V_new = solve_V(U_new, W, T_target, R)
            if V_new is None:
                return None
            W_new = solve_W(U_new, V_new, T_target, R)
            if W_new is None:
                return None
            
            diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
            U, V, W = U_new, V_new, W_new
            
            if diff < tol:
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                T_check = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.allclose(T_check, T_target, atol=1e-9):
                    return (U_r, V_r, W_r)
                
                # Try rounding to nearest integer
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                T_check_i = np.einsum('ir,jr,kr->ijk', U_i, V_i, W_i)
                if np.allclose(T_check_i, T_target, atol=1e-9):
                    return (U_i, V_i, W_i)
        
        return None
    
    # Try multiple random restarts to find a valid rank-15 decomposition
    best_233 = None
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline - 30:
            break
        
        result = als_decompose(T233, R233, max_iter=100, seed=restart)
        if result is not None:
            best_233 = result
            break
    
    if best_233 is not None:
        U233, V233, W233 = best_233
        
        # Embed into 2x3x6
        # Block 1: k in {0,1,2}, Block 2: k in {3,4,5}
        #
        # U233 has shape (6, 15) - rows indexed by i*3+j for i in {0,1}, j in {0,1,2}
        # V233 has shape (9, 15) - rows indexed by j*3+k for j in {0,1,2}, k in {0,1,2}
        # W233 has shape (6, 15) - rows indexed by k*2+i for k in {0,1,2}, i in {0,1}
        #
        # For 2x3x6:
        # U236 has shape (6, 30) - rows indexed by i*3+j
        # V236 has shape (18, 30) - rows indexed by j*6+k for j in {0,1,2}, k in {0,1,2,3,4,5}
        # W236 has shape (12, 30) - rows indexed by k*2+i for k in {0,1,2,3,4,5}, i in {0,1}
        #
        # Block 1 (k in {0,1,2}):
        #   U236[:, 0:15] = U233
        #   V236[j*6+k, 0:15] = V233[j*3+k, :] for j in {0,1,2}, k in {0,1,2}
        #   W236[k*2+i, 0:15] = W233[k*2+i, :] for k in {0,1,2}, i in {0,1}
        #
        # Block 2 (k in {3,4,5}):
        #   U236[:, 15:30] = U233
        #   V236[j*6+k, 15:30] = V233[j*3+(k-3), :] for j in {0,1,2}, k in {3,4,5}
        #   W236[k*2+i, 15:30] = W233[(k-3)*2+i, :] for k in {3,4,5}, i in {0,1}
        
        R236 = 2 * R233
        U236 = np.zeros((n*m, R236))
        V236 = np.zeros((m*p, R236))
        W236 = np.zeros((p*n, R236))
        
        # Block 1
        U236[:, :R233] = U233
        for j in range(m2):
            for k in range(p2):
                V236[j*p2 + k, :R233] = V233[j*p2 + k, :]
                W236[k*n2 + (j % n2), :R233] = W233[k*n2 + (j % n2), :]
        
        # Wait, that's not right. Let me re-derive.
        #
        # In the 2x3x3 tensor:
        #   T[i*3+j, j*3+k, k*2+i] = 1
        #   i in {0,1}, j in {0,1,2}, k in {0,1,2}
        #
        # In the 2x3x6 tensor:
        #   T[i*3+j, j*6+k, k*2+i] = 1
        #   i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4,5}
        #
        # Block 1: k in {0,1,2}
        #   T[i*3+j, j*6+k, k*2+i] = 1
        #   This matches the 2x3x3 tensor pattern with k unchanged
        #
        # Block 2: k in {3,4,5}
        #   T[i*3+j, j*6+k, k*2+i] = 1
        #   Let k' = k - 3, so k' in {0,1,2}
        #   T[i*3+j, j*6+k'+3, (k'+3)*2+i] = 1
        #   T[i*3+j, j*6+k'+3, k'*2+6+i] = 1
        #   This matches the 2x3x3 tensor pattern with k replaced by k'
        #   and the W index offset by 6
        
        # Let me redo the embedding properly.
        #
        # For block 1 (k in {0,1,2}):
        #   U index: i*3+j -> same as 2x3x3
        #   V index: j*6+k -> in 2x3x3 it's j*3+k, so we need to map
        #   W index: k*2+i -> same as 2x3x3
        
        # For block 2 (k in {3,4,5}):
        #   U index: i*3+j -> same as 2x3x3
        #   V index: j*6+k where k in {3,4,5} -> in 2x3x3 it's j*3+(k-3)
        #   W index: k*2+i where k in {3,4,5} -> in 2x3x3 it's (k-3)*2+i
        
        # Let's build the mapping explicitly.
        #
        # V233 rows: j*3+k for j in {0,1,2}, k in {0,1,2}
        #   row 0: j=0,k=0
        #   row 1: j=0,k=1
        #   row 2: j=0,k=2
        #   row 3: j=1,k=0
        #   row 4: j=1,k=1
        #   row 5: j=1,k=2
        #   row 6: j=2,k=0
        #   row 7: j=2,k=1
        #   row 8: j=2,k=2
        #
        # V236 rows: j*6+k for j in {0,1,2}, k in {0,1,2,3,4,5}
        #   row 0: j=0,k=0
        #   row 1: j=0,k=1
        #   row 2: j=0,k=2
        #   row 3: j=0,k=3
        #   row 4: j=0,k=4
        #   row 5: j=0,k=5
        #   row 6: j=1,k=0
        #   row 7: j=1,k=1
        #   row 8: j=1,k=2
        #   row 9: j=1,k=3
        #   row 10: j=1,k=4
        #   row 11: j=1,k=5
        #   row 12: j=2,k=0
        #   row 13: j=2,k=1
        #   row 14: j=2,k=2
        #   row 15: j=2,k=3
        #   row 16: j=2,k=4
        #   row 17: j=2,k=5
        #
        # W233 rows: k*2+i for k in {0,1,2}, i in {0,1}
        #   row 0: k=0,i=0
        #   row 1: k=0,i=1
        #   row 2: k=1,i=0
        #   row 3: k=1,i=1
        #   row 4: k=2,i=0
        #   row 5: k=2,i=1
        #
        # W236 rows: k*2+i for k in {0,1,2,3,4,5}, i in {0,1}
        #   row 0: k=0,i=0
        #   row 1: k=0,i=1
        #   row 2: k=1,i=0
        #   row 3: k=1,i=1
        #   row 4: k=2,i=0
        #   row 5: k=2,i=1
        #   row 6: k=3,i=0
        #   row 7: k=3,i=1
        #   row 8: k=4,i=0
        #   row 9: k=4,i=1
        #   row 10: k=5,i=0
        #   row 11: k=5,i=1
        
        U236 = np.zeros((n*m, R236))
        V236 = np.zeros((m*p, R236))
        W236 = np.zeros((p*n, R236))
        
        # Block 1: k in {0,1,2}
        U236[:, :R233] = U233
        for j in range(m2):
            for k in range(p2):
                v_src = j * p2 + k  # index in V233
                v_dst = j * p + k   # index in V236
                V236[v_dst, :R233] = V233[v_src, :]
                w_src = k * n2 + (j % n2)  # Hmm, this doesn't seem right
                # Actually, W index in 2x3x3 is k*2+i, and i is independent
                # In the tensor, W index is k*n+i = k*2+i
                # So for a given (k, i), the W row is k*2+i
                # The V row is j*3+k in 2x3x3, j*6+k in 2x3x6
                # The U row is i*3+j in both
                #
                # So for block 1, for each (i, j, k) with k in {0,1,2}:
                #   U[i*3+j, :15] = U233[i*3+j, :]
                #   V[j*6+k, :15] = V233[j*3+k, :]
                #   W[k*2+i, :15] = W233[k*2+i, :]
        
        # Let me just do it directly.
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    u_idx = i * m2 + j
                    v_idx_233 = j * p2 + k
                    w_idx_233 = k * n2 + i
                    v_idx_236_b1 = j * p + k
                    w_idx_236_b1 = k * n2 + i
                    
                    U236[u_idx, :R233] = U233[u_idx, :]
                    V236[v_idx_236_b1, :R233] = V233[v_idx_233, :]
                    W236[w_idx_236_b1, :R233] = W233[w_idx_233, :]
                    
                    # Block 2
                    v_idx_236_b2 = j * p + (k + p2)
                    w_idx_236_b2 = (k + p2) * n2 + i
                    U236[u_idx, R233:] = U233[u_idx, :]
                    V236[v_idx_236_b2, R233:] = V233[v_idx_233, :]
                    W236[w_idx_236_b2, R233:] = W233[w_idx_233, :]
        
        # Verify full tensor
        T_final = np.einsum('ir,jr,kr->ijk', U236, V236, W236)
        if np.allclose(T_final, T, atol=1e-9):
            return {
                "U": U236.tolist(),
                "V": V236.tolist(),
                "W": W236.tolist()
            }
    
    # Fallback to schoolbook rank 36
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
