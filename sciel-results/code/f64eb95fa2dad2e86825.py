import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n

    start_time = time.time()
    deadline = 140.0

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    best_result = get_schoolbook()

    # The strategy is to decompose the 2x3x7 multiplication into blocks.
    # We can split the 7 columns of B into 3+3+1.
    # This corresponds to 2x3x3, 2x3x3, and 2x3x1 multiplications.
    # The rank of 2x3x1 is 6 (trivial).
    # The rank of 2x3x3 is the key. If we can find a rank-11 decomposition for 2x3x3,
    # the total rank would be 11 + 11 + 6 = 28.
    # Schoolbook is 42. 28 is a significant improvement.
    # Even better, if we can find a rank-10 or rank-9 decomposition for 2x3x3,
    # the total rank would be even lower.
    # Known lower bound for 2x3x3 is 11 (Strassen's bound? No, Strassen is 7 for 2x2x2).
    # For 2x3x3, the lower bound is 11.
    # So 11 is the theoretical minimum.
    # If we find rank 11, total rank is 28.

    def find_2x3x3_decomposition(target_rank):
        nn, mm, pp = 2, 3, 3
        NU, NV, NW = nn*mm, mm*pp, pp*nn
        R = target_rank
        
        # Target Tensor
        T = np.zeros((NU, NV, NW), dtype=np.float64)
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    alpha = i * mm + j
                    beta = j * pp + k
                    gamma = k * nn + i
                    T[alpha, beta, gamma] = 1.0
        T_flat = T.flatten()
        
        def make_residual_func(R):
            def res(params):
                len_U = NU * R
                len_V = NV * R
                U = params[:len_U].reshape(NU, R)
                V = params[len_U:len_U+len_V].reshape(NV, R)
                W = params[len_U+len_V:].reshape(NW, R)
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                return (T_calc.flatten() - T_flat)
            return res
        
        res_func = make_residual_func(R)
        total_vars = (NU + NV + NW) * R
        
        best_sol = None
        best_cost = np.inf
        
        # Try a few restarts
        # We need to be careful with the number of restarts to stay within time limits.
        num_restarts = 50
        for _ in range(num_restarts):
            if time.time() > deadline - 20: # Leave time for rest
                break
            x0 = np.random.uniform(-1, 1, total_vars)
            try:
                sol = least_squares(res_func, x0, method='lm', max_nfev=1000)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
            except Exception:
                continue
        
        if best_sol is None or best_cost > 1e-6:
            return None
        
        # Round to half-integers
        x_rounded = np.round(best_sol * 2) / 2.0
        res_rounded = res_func(x_rounded)
        if np.linalg.norm(res_rounded) < 1e-9:
            len_U = NU * R
            len_V = NV * R
            U = x_rounded[:len_U].reshape(NU, R)
            V = x_rounded[len_U:len_U+len_V].reshape(NV, R)
            W = x_rounded[len_U+len_V:].reshape(NW, R)
            return U, V, W
            
        return None

    def get_2x3x1_rank6():
        nn, mm, pp = 2, 3, 1
        NU, NV, NW = nn*mm, mm*pp, pp*nn
        R = 6
        U = np.zeros((NU, R))
        V = np.zeros((NV, R))
        W = np.zeros((NW, R))
        
        col = 0
        for j in range(mm):
            # Term a_0j * b_j -> c_0
            U[0*mm + j, col] = 1
            V[j*pp + 0, col] = 1 # V index j
            W[0*nn + 0, col] = 1 # W index 0
            col += 1
            # Term a_1j * b_j -> c_1
            U[1*mm + j, col] = 1
            V[j*pp + 0, col] = 1
            W[1*nn + 0, col] = 1
            col += 1
            
        return U, V, W

    # Try to find rank 11 first, then 12, etc.
    # Since 11 is the lower bound, it's the most likely candidate for improvement.
    U_block = None
    V_block = None
    W_block = None
    block_rank = 0

    # Attempt rank 11
    if time.time() < deadline - 30:
        res11 = find_2x3x3_decomposition(11)
        if res11 is not None:
            U_block, V_block, W_block = res11
            block_rank = 11

    # If rank 11 failed, try rank 12 (just in case, though 11 is expected)
    if U_block is None and time.time() < deadline - 20:
        res12 = find_2x3x3_decomposition(12)
        if res12 is not None:
            U_block, V_block, W_block = res12
            block_rank = 12

    if U_block is not None:
        # Construct full 2x3x7 solution
        # Split B (3x7) into B1 (3x3), B2 (3x3), B3 (3x1)
        # C (2x7) into C1 (2x3), C2 (2x3), C3 (2x1)
        
        R_block1 = block_rank
        R_block2 = block_rank
        R_block3 = 6
        R_total = R_block1 + R_block2 + R_block3
        
        U_full = np.zeros((N_U, R_total))
        V_full = np.zeros((N_V, R_total))
        W_full = np.zeros((N_W, R_total))
        
        # Block 1: Cols 0-2 of B, Cols 0-2 of C
        U_full[:, 0:R_block1] = U_block
        # V maps local (j, k in 0..2) to global (j, k in 0..2)
        # V_block has rows indexed by j*3+k. Global V rows are j*7+k.
        for j in range(3):
            for k in range(3):
                local_idx = j * 3 + k
                global_idx = j * 7 + k
                V_full[global_idx, 0:R_block1] = V_block[local_idx, :]
        # W maps local (k, i) to global (k, i). k in 0..2, i in 0..1.
        # W_block has rows indexed by k*2+i. Global W rows are k*2+i. (Same structure)
        W_full[:, 0:R_block1] = W_block
        
        # Block 2: Cols 3-5 of B, Cols 3-5 of C
        offset1 = R_block1
        U_full[:, offset1:offset1+R_block2] = U_block
        for j in range(3):
            for k in range(3):
                local_idx = j * 3 + k
                global_idx = j * 7 + (k + 3)
                V_full[global_idx, offset1:offset1+R_block2] = V_block[local_idx, :]
        # W for C2: Cols 3-5. Global W index is (k+3)*2 + i?
        # No, W index is k*n + i. Here k is 3,4,5.
        # So rows 3*2+0=6, 3*2+1=7, 4*2+0=8, ...
        # W_block rows are 0..5 (k=0,1,2).
        # We need to shift W_block to rows 6..11.
        W_full[6:12, offset1:offset1+R_block2] = W_block
        
        # Block 3: Col 6 of B, Col 6 of C
        offset2 = R_block1 + R_block2
        U6, V6, W6 = get_2x3x1_rank6()
        U_full[:, offset2:offset2+R_block3] = U6
        # V6 rows indexed by j (0..2). Global V index j*7 + 6.
        for j in range(3):
            local_idx = j
            global_idx = j * 7 + 6
            V_full[global_idx, offset2:offset2+R_block3] = V6[local_idx, :]
        # W6 rows indexed by i (0..1). Global W index 6*2 + i = 12 + i.
        W_full[12:14, offset2:offset2+R_block3] = W6
        
        # Verify
        T_calc = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
        
        # Build Target
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
                    
        if np.allclose(T_calc, T_target, atol=1e-9):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
            
    return best_result
