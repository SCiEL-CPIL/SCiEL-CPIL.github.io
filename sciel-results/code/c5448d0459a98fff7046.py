import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)
    
    def is_half_integer(mat, tol=1e-6):
        scaled = 2.0 * mat
        rounded = np.round(scaled)
        return np.all(np.abs(scaled - rounded) < tol)
    
    def try_solve_for_U(V, W, T, r):
        """Given V and W, solve for U exactly."""
        M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
        if np.linalg.matrix_rank(M) < r:
            return None
        
        U = np.zeros((nm, r))
        for a in range(nm):
            target = T[a, :, :].flatten()
            try:
                sol, residuals, rank_m, sv = np.linalg.lstsq(M, target, rcond=None)
                if residuals[0] > 1e-8:
                    return None
                U[a, :] = sol
            except:
                return None
        return U
    
    def try_find_decomposition(r, max_trials=200):
        """Try to find a rank-r decomposition."""
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(1000 + r * 100 + trial)
            
            # Strategy 1: Block-structured V and W
            # Divide the 9 columns into 3 groups of 3
            # For each group, use a pattern based on 2x3 x 3x3 structure
            
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Assign rank indices to blocks
            # Try different block structures
            block_size = r // 3
            remainder = r % 3
            
            for block in range(3):
                start_idx = block * block_size + min(block, remainder)
                end_idx = (block + 1) * block_size + min(block + 1, remainder)
                block_cols = slice(start_idx, end_idx)
                
                # For block b (corresponding to columns 3b to 3b+2)
                # V rows: j*p + k where k is in [3b, 3b+2]
                # W rows: k*n + i where k is in [3b, 3b+2]
                
                for j in range(m):
                    for local_k in range(3):
                        k = block * 3 + local_k
                        v_row = j * p + k
                        # Random pattern for V
                        V[v_row, block_cols] = np.random.choice([-1, 0, 1], size=(end_idx - start_idx,))
                
                for local_k in range(3):
                    k = block * 3 + local_k
                    for i in range(n):
                        w_row = k * n + i
                        W[w_row, block_cols] = np.random.choice([-1, 0, 1], size=(end_idx - start_idx,))
            
            U = try_solve_for_U(V, W, T, r)
            if U is not None and is_half_integer(U):
                U_snapped = np.round(2.0 * U) / 2.0
                if verify_solution(U_snapped, V, W, T):
                    return (U_snapped, V, W)
            
            # Strategy 2: Fully random with {-1, 0, 1}
            V = np.random.randint(-1, 2, size=(mp, r))
            W = np.random.randint(-1, 2, size=(pn, r))
            
            U = try_solve_for_U(V, W, T, r)
            if U is not None and is_half_integer(U):
                U_snapped = np.round(2.0 * U) / 2.0
                if verify_solution(U_snapped, V, W, T):
                    return (U_snapped, V, W)
            
            # Strategy 3: Sparse random
            V = (np.random.rand(mp, r) < 0.3).astype(int) * np.random.choice([-1, 1], size=(mp, r))
            W = (np.random.rand(pn, r) < 0.3).astype(int) * np.random.choice([-1, 1], size=(pn, r))
            
            U = try_solve_for_U(V, W, T, r)
            if U is not None and is_half_integer(U):
                U_snapped = np.round(2.0 * U) / 2.0
                if verify_solution(U_snapped, V, W, T):
                    return (U_snapped, V, W)
            
            # Strategy 4: Use known 2x3x3 decomposition structure
            # The 2x3x3 matrix multiplication has rank 7 (vs schoolbook 6)
            # We can use this as a building block
            if r >= 7:
                # Create a decomposition using the structure
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                
                # For each of the 3 blocks of 3 columns
                for block in range(3):
                    start_idx = block * (r // 3) + min(block, r % 3)
                    end_idx = (block + 1) * (r // 3) + min(block + 1, r % 3)
                    block_cols = slice(start_idx, end_idx)
                    block_rank = end_idx - start_idx
                    
                    if block_rank >= 7:
                        # Use a 2x3x3-like structure for this block
                        # This is a heuristic based on known low-rank decompositions
                        for j in range(m):
                            for local_k in range(3):
                                k = block * 3 + local_k
                                v_row = j * p + k
                                # Use a structured pattern
                                for r_idx in range(block_rank):
                                    V[v_row, start_idx + r_idx] = np.random.choice([-1, 0, 1])
                        
                        for local_k in range(3):
                            k = block * 3 + local_k
                            for i in range(n):
                                w_row = k * n + i
                                for r_idx in range(block_rank):
                                    W[w_row, start_idx + r_idx] = np.random.choice([-1, 0, 1])
                
                U = try_solve_for_U(V, W, T, r)
                if U is not None and is_half_integer(U):
                    U_snapped = np.round(2.0 * U) / 2.0
                    if verify_solution(U_snapped, V, W, T):
                        return (U_snapped, V, W)
        
        return None
    
    deadline = time.time() + 135
    
    # Try ranks from low to high
    # Theoretical lower bound: max(nm, mp, pn) = max(6, 27, 18) = 27
    # Known results suggest rank might be around 36-45
    for r in range(27, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
        
        result = try_find_decomposition(r, max_trials=100)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break
    
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
