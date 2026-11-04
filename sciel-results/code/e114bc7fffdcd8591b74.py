import numpy as np
import time
from scipy.optimize import least_squares

n, m, p = 2, 5, 6

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def check_decomposition(U, V, W, n, m, p):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_optimize_subproblem(n_sub, m_sub, p_sub, target_rank, max_restarts=20):
        """
        Try to find a decomposition of rank target_rank for n_sub x m_sub x p_sub.
        Returns (U, V, W) or None.
        """
        if time.time() > deadline - 10:
            return None
            
        size_U = n_sub * m_sub * target_rank
        size_V = m_sub * p_sub * target_rank
        size_W = p_sub * n_sub * target_rank
        total_size = size_U + size_V + size_W
        
        u_indices, v_indices, w_indices = [], [], []
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    u_indices.append(i * m_sub + j)
                    v_indices.append(j * p_sub + k)
                    w_indices.append(k * n_sub + i)
        u_indices = np.array(u_indices)
        v_indices = np.array(v_indices)
        w_indices = np.array(w_indices)

        def residuals(x):
            U = x[:size_U].reshape((n_sub*m_sub, target_rank))
            V = x[size_U:size_U+size_V].reshape((m_sub*p_sub, target_rank))
            W = x[size_U+size_V:].reshape((p_sub*n_sub, target_rank))
            U_sel = U[u_indices, :]
            V_sel = V[v_indices, :]
            W_sel = W[w_indices, :]
            res = np.sum(U_sel * V_sel * W_sel, axis=1) - 1.0
            return res

        best_result = None
        best_cost = np.inf
        
        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U_init = np.random.randn(n_sub*m_sub, target_rank) * 0.5
            V_init = np.random.randn(m_sub*p_sub, target_rank) * 0.5
            W_init = np.random.randn(p_sub*n_sub, target_rank) * 0.5
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=500,
                                       ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result
                    if result.cost < 1e-10:
                        break
            except Exception:
                pass
        
        if best_result is not None and best_result.cost < 1e-8:
            U = best_result.x[:size_U].reshape((n_sub*m_sub, target_rank))
            V = best_result.x[size_U:size_U+size_V].reshape((m_sub*p_sub, target_rank))
            W = best_result.x[size_U+size_V:].reshape((p_sub*n_sub, target_rank))
            
            # Snap to integers/half-integers if possible
            for scale in [2, 1]:
                U_snap = np.round(U * scale) / scale
                V_snap = np.round(V * scale) / scale
                W_snap = np.round(W * scale) / scale
                if check_decomposition(U_snap, V_snap, W_snap, n_sub, m_sub, p_sub):
                    return (U_snap, V_snap, W_snap)
            
            # If snapping fails, return the floating point result if it's very accurate
            if best_result.cost < 1e-10:
                return (U, V, W)
                
        return None

    def assemble_solution(U_233, V_233, W_233, U_223, V_223, W_223):
        """
        Assemble 2x5x6 from 2x3x3 and 2x2x3 components.
        5 = 3 + 2
        6 = 3 + 3
        A (2x5) = [A_L (2x3) | A_R (2x2)]
        B (5x6) = [B_LL (3x3) B_LR (3x3); B_RL (2x3) B_RR (2x3)]
        
        C (2x6) = A_L B_L + A_R B_R
        C[:, 0:3] = A_L B_LL + A_R B_RL
        C[:, 3:6] = A_L B_LR + A_R B_RR
        
        We need to map the indices correctly.
        The components are:
        1. A_L B_LL: 2x3x3
        2. A_R B_RL: 2x2x3
        3. A_L B_LR: 2x3x3
        4. A_R B_RR: 2x2x3
        
        We have decompositions for 2x3x3 and 2x2x3.
        Let R_233 be the rank of 2x3x3.
        Let R_223 be the rank of 2x2x3.
        
        The total rank will be 2*R_233 + 2*R_223.
        
        We need to construct U, V, W for the full problem.
        U has shape (2*5, R_total) = (10, R_total)
        V has shape (5*6, R_total) = (30, R_total)
        W has shape (6*2, R_total) = (12, R_total)
        
        Indices:
        U: i (0,1), j (0..4)
        V: j (0..4), k (0..5)
        W: k (0..5), i (0,1)
        
        Component 1 (A_L B_LL):
        i in 0..1, j in 0..2, k in 0..2
        Uses U_233[i*3+j, :], V_233[j*3+k, :], W_233[k*2+i, :]
        
        Component 2 (A_R B_RL):
        i in 0..1, j in 3..4, k in 0..2
        A_R is rows 3,4 of A.
        B_RL is rows 3,4 of B, cols 0..2.
        Uses U_223[i*2+(j-3), :], V_223[(j-3)*3+k, :], W_223[k*2+i, :]
        
        Component 3 (A_L B_LR):
        i in 0..1, j in 0..2, k in 3..5
        B_LR is rows 0..2 of B, cols 3..5.
        Uses U_233[i*3+j, :], V_233[j*3+(k-3), :], W_233[(k-3)*2+i, :]
        
        Component 4 (A_R B_RR):
        i in 0..1, j in 3..4, k in 3..5
        B_RR is rows 3,4 of B, cols 3..5.
        Uses U_223[i*2+(j-3), :], V_223[(j-3)*3+(k-3), :], W_223[(k-3)*2+i, :]
        """
        R_233 = U_233.shape[1]
        R_223 = U_223.shape[1]
        R_total = 2 * R_233 + 2 * R_223
        
        U_full = np.zeros((10, R_total))
        V_full = np.zeros((30, R_total))
        W_full = np.zeros((12, R_total))
        
        # Component 1: A_L B_LL (2x3x3) -> Rank R_233
        # U_233 shape (6, R_233)
        # V_233 shape (9, R_233)
        # W_233 shape (6, R_233)
        
        # Offset 0
        U_full[0:6, 0:R_233] = U_233
        V_full[0:9, 0:R_233] = V_233
        W_full[0:6, 0:R_233] = W_233
        
        # Component 2: A_R B_RL (2x2x3) -> Rank R_223
        # U_223 shape (4, R_223)
        # V_223 shape (6, R_223)
        # W_223 shape (6, R_223)
        
        # Offset R_233
        # U indices: i (0,1), j' (0,1) -> j = j' + 3
        # U_full[i*5 + j, R_233 + r] = U_223[i*2 + j', r]
        for i in range(2):
            for j_prime in range(2):
                j = j_prime + 3
                U_full[i*5 + j, R_233 : R_233 + R_223] = U_223[i*2 + j_prime, :]
                
        # V indices: j' (0,1), k (0..2) -> j = j' + 3
        # V_full[j*6 + k, R_233 + r] = V_223[j'*3 + k, r]
        for j_prime in range(2):
            for k in range(3):
                j = j_prime + 3
                V_full[j*6 + k, R_233 : R_233 + R_223] = V_223[j_prime*3 + k, :]
                
        # W indices: k (0..2), i (0,1)
        # W_full[k*2 + i, R_233 + r] = W_223[k*2 + i, r]
        W_full[0:6, R_233 : R_233 + R_223] = W_223
        
        # Component 3: A_L B_LR (2x3x3) -> Rank R_233
        # Offset 2*R_233
        
        # U indices: i (0,1), j (0..2)
        # U_full[i*5 + j, 2*R_233 + r] = U_233[i*3 + j, r]
        U_full[0:6, 2*R_233 : 3*R_233] = U_233
        
        # V indices: j (0..2), k' (0..2) -> k = k' + 3
        # V_full[j*6 + k, 2*R_233 + r] = V_233[j*3 + k', r]
        for j in range(3):
            for k_prime in range(3):
                k = k_prime + 3
                V_full[j*6 + k, 2*R_233 : 3*R_233] = V_233[j*3 + k_prime, :]
                
        # W indices: k' (0..2), i (0,1) -> k = k' + 3
        # W_full[k*2 + i, 2*R_233 + r] = W_233[k'*2 + i, r]
        W_full[6:12, 2*R_233 : 3*R_233] = W_233
        
        # Component 4: A_R B_RR (2x2x3) -> Rank R_223
        # Offset 2*R_233 + R_223
        
        # U indices: i (0,1), j' (0,1) -> j = j' + 3
        # U_full[i*5 + j, 2*R_233 + R_223 + r] = U_223[i*2 + j', r]
        for i in range(2):
            for j_prime in range(2):
                j = j_prime + 3
                U_full[i*5 + j, 2*R_233 + R_223 : 3*R_233 + 2*R_223] = U_223[i*2 + j_prime, :]
                
        # V indices: j' (0,1), k' (0..2) -> j = j' + 3, k = k' + 3
        # V_full[j*6 + k, 2*R_233 + R_223 + r] = V_223[j'*3 + k', r]
        for j_prime in range(2):
            for k_prime in range(3):
                j = j_prime + 3
                k = k_prime + 3
                V_full[j*6 + k, 2*R_233 + R_223 : 3*R_233 + 2*R_223] = V_223[j_prime*3 + k_prime, :]
                
        # W indices: k' (0..2), i (0,1) -> k = k' + 3
        # W_full[k*2 + i, 2*R_233 + R_223 + r] = W_223[k'*2 + i, r]
        W_full[6:12, 2*R_233 + R_223 : 3*R_233 + 2*R_223] = W_223
        
        return U_full, V_full, W_full

    # Try to solve sub-problems
    # 2x3x3
    sol_233 = None
    for R in [10, 11, 12]:
        sol_233 = try_optimize_subproblem(2, 3, 3, R, max_restarts=10)
        if sol_233 is not None:
            break
            
    # 2x2x3
    sol_223 = None
    for R in [10, 11, 12]:
        sol_223 = try_optimize_subproblem(2, 2, 3, R, max_restarts=10)
        if sol_223 is not None:
            break
            
    if sol_233 is not None and sol_223 is not None:
        U, V, W = assemble_solution(sol_233[0], sol_233[1], sol_233[2], 
                                    sol_223[0], sol_223[1], sol_223[2])
        if check_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to original optimizer
    def try_optimize_rank(R, max_restarts=8):
        if time.time() > deadline - 25:
            return None

        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W

        u_indices, v_indices, w_indices = [], [], []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_indices.append(i * m + j)
                    v_indices.append(j * p + k)
                    w_indices.append(k * n + i)
        u_indices = np.array(u_indices)
        v_indices = np.array(v_indices)
        w_indices = np.array(w_indices)

        def residuals(x):
            U = x[:size_U].reshape((n*m, R))
            V = x[size_U:size_U+size_V].reshape((m*p, R))
            W = x[size_U+size_V:].reshape((p*n, R))
            U_sel = U[u_indices, :]
            V_sel = V[v_indices, :]
            W_sel = W[w_indices, :]
            res = np.sum(U_sel * V_sel * W_sel, axis=1) - 1.0
            return res

        def try_initialization(x0):
            if time.time() > deadline - 20:
                return None

            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=300,
                                       ftol=1e-10, xtol=1e-10, gtol=1e-10)
            except Exception:
                return None

            if result.cost < 1e-6:
                try:
                    result2 = least_squares(residuals, result.x, method='lm', max_nfev=200,
                                            ftol=1e-14, xtol=1e-14, gtol=1e-14)
                    if result2.cost < result.cost:
                        result = result2
                except Exception:
                    pass

            if result.cost < 1e-10:
                U = result.x[:size_U].reshape((n*m, R))
                V = result.x[size_U:size_U+size_V].reshape((m*p, R))
                W = result.x[size_U+size_V:].reshape((p*n, R))

                for scale in [2, 1]:
                    U_snap = np.round(U * scale) / scale
                    V_snap = np.round(V * scale) / scale
                    W_snap = np.round(W * scale) / scale

                    if check_decomposition(U_snap, V_snap, W_snap, n, m, p):
                        return (U_snap, V_snap, W_snap)

        for attempt in range(max_restarts):
            if time.time() > deadline - 20:
                break

            if attempt == 0:
                U_s, V_s, W_s = get_schoolbook(n, m, p)
                if R <= 60:
                    U_init = U_s[:, :R].copy()
                    V_init = V_s[:, :R].copy()
                    W_init = W_s[:, :R].copy()
                    U_init += np.random.randn(n*m, R) * 0.1
                    V_init += np.random.randn(m*p, R) * 0.1
                    W_init += np.random.randn(p*n, R) * 0.1
                else:
                    U_init = np.random.randn(n*m, R) * 0.5
                    V_init = np.random.randn(m*p, R) * 0.5
                    W_init = np.random.randn(p*n, R) * 0.5
            else:
                U_init = np.random.randn(n*m, R) * 0.5
                V_init = np.random.randn(m*p, R) * 0.5
                W_init = np.random.randn(p*n, R) * 0.5

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            result = try_initialization(x0)
            if result is not None:
                return result

        return None

    for R in range(59, 44, -1):
        if time.time() > deadline - 30:
            break
        result = try_optimize_rank(R, max_restarts=6 if R > 55 else 4)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = get_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
