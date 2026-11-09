import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 35s safety margin
    
    # Target tensor T[i, l, k] = 1 for all valid indices
    # C[i, k] = sum_l A[i, l] B[l, k]
    T = np.ones((n, m, p))
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                U_school[i * m + l, col] = 1
                V_school[l * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Strategy: Use Strassen's rank-7 decomposition for 2x2 matrix multiplication
    # applied to the 2x3 by 3x9 structure.
    # 
    # For 2x2 * 2x2, Strassen uses 7 multiplications:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For our 2x3 * 3x9 case, we can think of it as:
    # A is 2x3, B is 3x9
    # We can partition B into three 3x3 blocks: B = [B1 | B2 | B3]
    # Then C = A * B = [A*B1 | A*B2 | A*B3]
    # Each A*Bj is 2x3, which we can decompose.
    #
    # Actually, let's think differently. The tensor T[i,l,k] = 1 means:
    # C[i,k] = sum_l A[i,l] B[l,k]
    # This is a bilinear map from A (2x3) and B (3x9) to C (2x9).
    #
    # We can use the fact that this is equivalent to 3 independent 2x1 * 1x9 multiplications
    # combined, but that's not helpful.
    #
    # Better: View A as [A1; A2] where A1, A2 are 1x3, and B as a 3x9 matrix.
    # C1 = A1 * B (1x9), C2 = A2 * B (1x9)
    # C1[k] = sum_l A1[l] B[l,k]
    # C2[k] = sum_l A2[l] B[l,k]
    #
    # This is a linear map from (A1, A2, B) to (C1, C2).
    # The tensor rank is the rank of the bilinear form.
    #
    # Let's try a direct approach: find a low-rank decomposition by solving
    # the system of equations.
    
    # Try ranks from 21 up to 53
    for rank in range(21, 54):
        if time.time() > deadline:
            break
            
        # Initialize with structured random
        np.random.seed(42 + rank)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # ALS iterations with better initialization
        for it in range(100):
            if time.time() > deadline:
                break
                
            # Solve for V: for each (l,k), M[i,c] = U[i*m+l,c] * W[k*n+i,c]
            V_new = np.zeros((m * p, rank))
            for l in range(m):
                for k in range(p):
                    M_vk = np.array([U[i * m + l, :] * W[k * n + i, :] for i in range(n)])
                    t_vk = np.ones(n)
                    try:
                        v_vk, _, _, _ = np.linalg.lstsq(M_vk, t_vk, rcond=None)
                        V_new[l * p + k, :] = v_vk
                    except:
                        pass
            V = V_new
            
            # Solve for U
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for l in range(m):
                    M_uk = np.array([V[l * p + k, :] * W[k * n + i, :] for k in range(p)])
                    t_uk = np.ones(p)
                    try:
                        u_uk, _, _, _ = np.linalg.lstsq(M_uk, t_uk, rcond=None)
                        U_new[i * m + l, :] = u_uk
                    except:
                        pass
            U = U_new
            
            # Solve for W
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    M_wk = np.array([U[i * m + l, :] * V[l * p + k, :] for l in range(m)])
                    t_wk = np.ones(m)
                    try:
                        w_wk, _, _, _ = np.linalg.lstsq(M_wk, t_wk, rcond=None)
                        W_new[k * n + i, :] = w_wk
                    except:
                        pass
            W = W_new
            
        # Check residual
        T_recon = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            for i in range(n):
                for l in range(m):
                    for k in range(p):
                        T_recon[i, l, k] += u_c[i, l] * v_c[l, k] * w_c[k, i]
                        
        residual = np.linalg.norm(T - T_recon)
        
        if residual < 1e-6:
            # Try to round to half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            # Verify exactness with rounded values
            T_exact = np.zeros((n, m, p))
            for c in range(rank):
                u_c = U_round[:, c].reshape(n, m)
                v_c = V_round[:, c].reshape(m, p)
                w_c = W_round[:, c].reshape(p, n)
                for i in range(n):
                    for l in range(m):
                        for k in range(p):
                            T_exact[i, l, k] += u_c[i, l] * v_c[l, k] * w_c[k, i]
                            
            if np.allclose(T, T_exact, atol=1e-9):
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                
    return best_result
