import time
import numpy as np
import itertools
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def snap_to_integer(arr):
        return np.round(arr)

    def try_snap_validate(U, V, W):
        # Try half-integers first
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = snap_to_integer(U)
        V_i = snap_to_integer(V)
        W_i = snap_to_integer(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        # Try quarters
        U_q = np.round(U * 4) / 4
        V_q = np.round(V * 4) / 4
        W_q = np.round(W * 4) / 4
        if validate(U_q, V_q, W_q):
            return True, U_q, V_q, W_q
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1

            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1

            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1

            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)

        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1

        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

        return U, V, W

    def try_reduce_rank(U_init, V_init, W_init, target_rank, max_steps=500):
        """
        Try to reduce rank by removing columns and re-optimizing with adaptive GD + momentum.
        """
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        current_rank = U.shape[1]
        
        if current_rank <= target_rank:
            return U, V, W, current_rank

        # Try removing each column one by one
        for col_to_remove in range(current_rank):
            if time.time() > deadline - 25:
                break
            
            mask = [i for i in range(current_rank) if i != col_to_remove]
            U_new = U[:, mask].copy()
            V_new = V[:, mask].copy()
            W_new = W[:, mask].copy()
            
            # Optimize the reduced set with momentum-based adaptive GD
            U_opt, V_opt, W_opt = U_new.copy(), V_new.copy(), W_new.copy()
            
            # Add small noise to break symmetry
            U_opt += np.random.randn(*U_opt.shape) * 0.01
            V_opt += np.random.randn(*V_opt.shape) * 0.01
            W_opt += np.random.randn(*W_opt.shape) * 0.01
            
            # Momentum terms
            vU = np.zeros_like(U_opt)
            vV = np.zeros_like(V_opt)
            vW = np.zeros_like(W_opt)
            
            beta1, beta2 = 0.9, 0.99
            lr_init = 0.02
            epsilon = 1e-8
            step_count = 0
            mU = np.zeros_like(U_opt)
            mV = np.zeros_like(V_opt)
            mW = np.zeros_like(W_opt)
            vU_sq = np.zeros_like(U_opt)
            vV_sq = np.zeros_like(V_opt)
            vW_sq = np.zeros_like(W_opt)
            
            success = False
            best_loss = np.inf
            best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
            
            for step in range(max_steps):
                if time.time() > deadline - 15:
                    break
                
                prod = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt, optimize=True)
                diff = prod - T
                loss = np.sum(diff * diff)
                
                if loss < 1e-10:
                    success = True
                    break
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                
                gradU = np.einsum('abc,br,cr->ar', diff, V_opt, W_opt, optimize=True)
                gradV = np.einsum('abc,ar,cr->br', diff, U_opt, W_opt, optimize=True)
                gradW = np.einsum('abc,ar,br->cr', diff, U_opt, V_opt, optimize=True)
                
                step_count += 1
                # Adam-style update
                mU = beta1 * mU + (1 - beta1) * gradU
                mV = beta1 * mV + (1 - beta1) * gradV
                mW = beta1 * mW + (1 - beta1) * gradW
                mU_hat = mU / (1 - beta1**step_count)
                mV_hat = mV / (1 - beta1**step_count)
                mW_hat = mW / (1 - beta1**step_count)
                
                vU_sq = beta2 * vU_sq + (1 - beta2) * (gradU ** 2)
                vV_sq = beta2 * vV_sq + (1 - beta2) * (gradV ** 2)
                vW_sq = beta2 * vW_sq + (1 - beta2) * (gradW ** 2)
                vU_hat = vU_sq / (1 - beta2**step_count)
                vV_hat = vV_sq / (1 - beta2**step_count)
                vW_hat = vW_sq / (1 - beta2**step_count)
                
                lr = lr_init / (1 + step_count * 0.001)
                
                U_opt -= lr * mU_hat / (np.sqrt(vU_hat) + epsilon)
                V_opt -= lr * mV_hat / (np.sqrt(vV_hat) + epsilon)
                W_opt -= lr * mW_hat / (np.sqrt(vW_hat) + epsilon)

            if success:
                valid, U_s, V_s, W_s = try_snap_validate(U_opt, V_opt, W_opt)
                if valid:
                    return U_s, V_s, W_s, target_rank
            
            # Also try snapping the best found solution
            if best_loss < 1e-6:
                valid, U_s, V_s, W_s = try_snap_validate(best_U, best_V, best_W)
                if valid:
                    return U_s, V_s, W_s, target_rank
        
        return None, None, None, current_rank

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to naive rank 50
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_U, best_V, best_W = U46, V46, W46
    best_rank = 46

    np.random.seed(42)

    # Try to reduce to 45 with multiple attempts using different random seeds
    for seed in [42, 123, 456]:
        if time.time() > deadline - 35:
            break
        np.random.seed(seed)
        U_res, V_res, W_res, res_rank = try_reduce_rank(U46, V46, W46, 45)
        if U_res is not None and res_rank == 45:
            best_U, best_V, best_W = U_res, V_res, W_res
            best_rank = 45
            break

    # Try to reduce to 44 if we have time
    if best_rank == 45 and time.time() < deadline - 20:
        for seed in [42, 123]:
            if time.time() > deadline - 15:
                break
            np.random.seed(seed)
            U_res, V_res, W_res, res_rank = try_reduce_rank(best_U, best_V, best_W, 44)
            if U_res is not None and res_rank == 44:
                best_U, best_V, best_W = U_res, V_res, W_res
                best_rank = 44
                break

    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
# EVOLVE-BLOCK-END
