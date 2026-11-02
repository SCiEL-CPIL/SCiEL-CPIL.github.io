import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_u = n * m  # 9
    num_v = m * p  # 21
    num_w = p * n  # 21
    
    # Build the target tensor T
    T = np.zeros((num_u, num_v, num_w))
    for u in range(num_u):
        i = u // m
        j = u % m
        for v in range(num_v):
            j_v = v // p
            k = v % p
            if j != j_v:
                continue
            for w in range(num_w):
                k_w = w // n
                i_w = w % n
                if k == k_w and i == i_w:
                    T[u, v, w] = 1.0

    # Unfolds for ALS
    T_1 = T.reshape(num_u, num_v * num_w)
    T_2 = T.transpose(1, 2, 0).reshape(num_v, num_w * num_u)
    T_3 = T.transpose(2, 0, 1).reshape(num_w, num_u * num_v)
    
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    def check_solution(U, V, W, tol=1e-4):
        """Check if U, V, W exactly decompose T (within tolerance)."""
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        diff = T_calc - T
        max_err = np.max(np.abs(diff))
        return max_err < tol

    def try_round_and_verify(U, V, W):
        """Try to round U, V, W to integers or half-integers and verify."""
        # Try integer rounding
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if check_solution(U_int, V_int, W_int):
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        
        # Try half-integer rounding
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        if check_solution(U_half, V_half, W_half):
            return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
        
        return None

    def als_decompose(R, U_init=None, V_init=None, W_init=None, max_iters=200, tol=1e-8):
        """Run ALS to find rank-R decomposition."""
        if U_init is not None:
            U = U_init.copy()
        else:
            U = np.random.randn(num_u, R)
        
        if V_init is not None:
            V = V_init.copy()
        else:
            V = np.random.randn(num_v, R)
        
        if W_init is not None:
            W = W_init.copy()
        else:
            W = np.random.randn(num_w, R)
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline:
                return None, np.inf
            
            # Update U
            K_u = np.einsum('vr, wr -> vw r', V, W).reshape(num_v * num_w, R)
            try:
                U, _, _, _ = np.linalg.lstsq(K_u.T, T_1.T, rcond=None)
            except np.linalg.LinAlgError:
                return None, np.inf
            
            # Update V
            K_v = np.einsum('wr, ur -> wu r', W, U).reshape(num_w * num_u, R)
            try:
                V, _, _, _ = np.linalg.lstsq(K_v.T, T_2.T, rcond=None)
            except np.linalg.LinAlgError:
                return None, np.inf
                
            # Update W
            K_w = np.einsum('ur, vr -> uv r', U, V).reshape(num_u * num_v, R)
            try:
                W, _, _, _ = np.linalg.lstsq(K_w.T, T_3.T, rcond=None)
            except np.linalg.LinAlgError:
                return None, np.inf
            
            # Check error
            T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
            err = np.linalg.norm(T_calc - T)
            
            if err < tol:
                return (U, V, W), err
                
            if abs(prev_err - err) < 1e-12:
                break
            prev_err = err
        
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        final_err = np.linalg.norm(T_calc - T)
        return (U, V, W), final_err

    def try_rank_with_structured_init(R, max_restarts=30):
        """Try rank R with multiple initialization strategies."""
        best_result = None
        best_err = np.inf
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + R * 1000 + restart)
            
            # Choose initialization strategy based on restart number
            strategy = restart % 5
            
            if strategy == 0:
                # Pure random
                U = np.random.randn(num_u, R)
                V = np.random.randn(num_v, R)
                W = np.random.randn(num_w, R)
            elif strategy == 1:
                # Kronecker-like structure
                U = np.random.randn(num_u, R)
                V = np.random.randn(num_v, R)
                W = np.random.randn(num_w, R)
                # Add some structure
                for r in range(R):
                    U[:, r] *= np.random.uniform(0.5, 2.0)
                    V[:, r] *= np.random.uniform(0.5, 2.0)
                    W[:, r] *= np.random.uniform(0.5, 2.0)
            elif strategy == 2:
                # Sparse-like initialization
                U = np.zeros((num_u, R))
                V = np.zeros((num_v, R))
                W = np.zeros((num_w, R))
                for r in range(R):
                    # Random sparse pattern
                    U[np.random.choice(num_u, size=min(num_u, R//3+1), replace=False), r] = np.random.randn(min(num_u, R//3+1))
                    V[np.random.choice(num_v, size=min(num_v, R//3+1), replace=False), r] = np.random.randn(min(num_v, R//3+1))
                    W[np.random.choice(num_w, size=min(num_w, R//3+1), replace=False), r] = np.random.randn(min(num_w, R//3+1))
            elif strategy == 3:
                # Block diagonal-like
                U = np.random.randn(num_u, R)
                V = np.random.randn(num_v, R)
                W = np.random.randn(num_w, R)
                # Scale columns
                for r in range(R):
                    scale = np.random.uniform(0.1, 5.0)
                    U[:, r] *= scale
                    V[:, r] /= scale
                    W[:, r] *= 1.0
            else:
                # Mixed: some columns from schoolbook
                U = np.random.randn(num_u, R)
                V = np.random.randn(num_v, R)
                W = np.random.randn(num_w, R)
                # Inject some schoolbook patterns
                for r in range(min(3, R)):
                    i, j, k = r // 7, (r // 3) % 3, r % 3
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
            
            result, err = als_decompose(R, U, V, W)
            if result is not None:
                if err < best_err:
                    best_err = err
                    best_result = result
                
                # If we found a good solution, try to round
                if err < 1e-6:
                    rounded = try_round_and_verify(*result)
                    if rounded is not None:
                        return rounded
        
        return None

    # Schoolbook fallback
    def schoolbook():
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

    # Try ranks from low to high
    # Known lower bound is around 21-23 for 3x3x3 (Strassen), so 3x3x7 should be higher
    # Try a range of ranks
    for R in range(21, 50):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(42 + R * 7)
        result = try_rank_with_structured_init(R)
        
        if result is not None:
            return result
        
        # Also try with more restarts for promising ranks
        if R in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
            if time.time() > deadline - 10:
                break
            np.random.seed(999 + R * 13)
            result = try_rank_with_structured_init(R, max_restarts=50)
            if result is not None:
                return result
    
    # Final attempt with higher ranks and more restarts
    for R in range(36, 45):
        if time.time() > deadline - 3:
            break
        np.random.seed(777 + R * 11)
        result = try_rank_with_structured_init(R, max_restarts=40)
        if result is not None:
            return result
    
    # Fallback: Schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
