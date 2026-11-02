import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    dim_U = n * m  # 9
    dim_V = m * p  # 21
    dim_W = p * n  # 21
    
    # Build the target tensor T for M_{3,3,7}
    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def tensor_from_factors(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def verify_solution(U, V, W, T_target, tol=1e-8):
        T_est = tensor_from_factors(U, V, W)
        return np.allclose(T_target, T_est, atol=tol)

    def is_close_to_half_integer(x, tol=1e-4):
        rounded = np.round(2*x)/2
        return np.all(np.abs(x - rounded) < tol)

    def round_to_half_integer(x):
        return np.round(2*x)/2

    # Try to find a rank-R decomposition for the 3x3x3 core
    def find_333_decomposition(R, max_iters=2000, time_limit=25):
        T_333 = np.zeros((9, 9, 9), dtype=float)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + j
                    b = j * 3 + k
                    c = k * 3 + i
                    T_333[a, b, k*3+i] = 1
        
        start_time = time.time()
        best_err = np.inf
        best_U, best_V, best_W = None, None, None
        
        # Multiple restarts with different strategies
        for restart in range(30):
            if time.time() - start_time > time_limit:
                break
            
            np.random.seed(12345 + restart * 7919)
            
            # Different initialization strategies
            if restart % 3 == 0:
                # Random initialization
                U = np.random.randn(9, R) * 0.1
                V = np.random.randn(9, R) * 0.1
                W = np.random.randn(9, R) * 0.1
            elif restart % 3 == 1:
                # Sparse initialization
                U = np.zeros((9, R))
                V = np.zeros((9, R))
                W = np.zeros((9, R))
                for r in range(R):
                    U[np.random.choice(9, size=2, replace=False), r] = np.random.choice([-1, 1], size=2)
                    V[np.random.choice(9, size=2, replace=False), r] = np.random.choice([-1, 1], size=2)
                    W[np.random.choice(9, size=2, replace=False), r] = np.random.choice([-1, 1], size=2)
            else:
                # Structured initialization based on Strassen-like patterns
                U = np.zeros((9, R))
                V = np.zeros((9, R))
                W = np.zeros((9, R))
                # Try to initialize with known Strassen structure patterns
                for r in range(min(R, 7)):
                    # Strassen's 7 products for 2x2x2, extended heuristically
                    idx = np.random.choice(9, size=3, replace=False)
                    U[idx, r] = np.random.choice([-1, 0, 1], size=3)
                    V[np.random.choice(9, size=3, replace=False), r] = np.random.choice([-1, 0, 1], size=3)
                    W[np.random.choice(9, size=3, replace=False), r] = np.random.choice([-1, 0, 1], size=3)
            
            # ALS optimization
            for it in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U
                M_U = np.einsum('jr,kr->jkr', V, W).reshape(81, R)
                T_2 = T_333.reshape(9, 81)
                try:
                    U = np.linalg.lstsq(M_U.T, T_2.T, rcond=None)[0].T
                except:
                    pass
                
                # Update V
                M_V = np.einsum('ir,kr->ikr', U, W).reshape(81, R)
                T_2 = T_333.transpose(1, 0, 2).reshape(9, 81)
                try:
                    V = np.linalg.lstsq(M_V.T, T_2.T, rcond=None)[0].T
                except:
                    pass
                
                # Update W
                M_W = np.einsum('ir,jr->ijr', U, V).reshape(81, R)
                T_2 = T_333.transpose(2, 0, 1).reshape(9, 81)
                try:
                    W = np.linalg.lstsq(M_W.T, T_2.T, rcond=None)[0].T
                except:
                    pass
                
                T_est = tensor_from_factors(U, V, W)
                err = np.linalg.norm(T_333 - T_est)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    # Check if we can round to half-integers
                    if is_close_to_half_integer(U) and is_close_to_half_integer(V) and is_close_to_half_integer(W):
                        U_r = round_to_half_integer(U)
                        V_r = round_to_half_integer(V)
                        W_r = round_to_half_integer(W)
                        if verify_solution(U_r, V_r, W_r, T_333, tol=1e-6):
                            return U_r, V_r, W_r
                    # Try to polish: project to nearest half-integers and verify
                    U_r = round_to_half_integer(U)
                    V_r = round_to_half_integer(V)
                    W_r = round_to_half_integer(W)
                    if verify_solution(U_r, V_r, W_r, T_333, tol=1e-4):
                        return U_r, V_r, W_r
        
        # Return best found even if not perfect (for debugging)
        if best_err < 1e-6:
            return best_U, best_V, best_W
        return None, None, None

    # Schoolbook fallback
    U_school = np.zeros((dim_U, 63), dtype=float)
    V_school = np.zeros((dim_V, 63), dtype=float)
    W_school = np.zeros((dim_W, 63), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 105  # Safety margin
    
    best_rank = 63
    best_factors = (U_school, V_school, W_school)
    
    # Try to find M_333 rank < 27
    # If we find R_333, total rank is R_333 + 36
    for R_333 in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        
        U_333, V_333, W_333 = find_333_decomposition(R_333, time_limit=20)
        if U_333 is not None:
            # Verify the 333 solution
            T_333 = np.zeros((9, 9, 9), dtype=float)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        a = i * 3 + j
                        b = j * 3 + k
                        c = k * 3 + i
                        T_333[a, b, c] = 1
            if not verify_solution(U_333, V_333, W_333, T_333, tol=1e-6):
                continue
            
            # Construct M_337 solution
            total_rank = R_333 + 36
            if total_rank < best_rank:
                U_total = np.zeros((9, total_rank), dtype=float)
                V_total = np.zeros((21, total_rank), dtype=float)
                W_total = np.zeros((21, total_rank), dtype=float)
                
                # Copy 333 part
                U_total[:, :R_333] = U_333
                # Map V_333 (9xR) to V_total (21xR)
                for j in range(3):
                    for k in range(3):
                        V_total[j*7+k, :R_333] = V_333[j*3+k, :R_333]
                # Map W_333 (9xR) to W_total (21xR)
                for k in range(3):
                    for i in range(3):
                        W_total[k*3+i, :R_333] = W_333[k*3+i, :R_333]
                
                # Add 4 blocks of 9 for remaining columns
                for k in range(3, 7):
                    offset = R_333 + (k-3)*9
                    for i in range(3):
                        for j in range(3):
                            col_idx = offset + i*3 + j
                            U_total[i*3+j, col_idx] = 1
                            V_total[j*7+k, col_idx] = 1
                            W_total[k*3+i, col_idx] = 1
                
                # Verify full solution
                if verify_solution(U_total, V_total, W_total, T, tol=1e-6):
                    best_rank = total_rank
                    best_factors = (U_total, V_total, W_total)
                    if R_333 == 23:
                        break  # Can't do better than 23 for 333
    
    U, V, W = best_factors
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
