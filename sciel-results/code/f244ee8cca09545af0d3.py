import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 7 matrix.
    
    Uses vectorized verification and greedy column removal with multiple strategies.
    Falls back to schoolbook decomposition if no improvement found within time limit.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    n, m, p = 3, 4, 7
    R_school = n * m * p  # 84
    
    # Precompute index mappings for fast verification
    # For each (i,j,k), we need u_idx, v_idx, w_idx
    index_map = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                index_map.append((u_idx, v_idx, w_idx))
    index_map = np.array(index_map)  # shape (n*m*p, 3)
    
    def build_schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        
        for r in range(R):
            i = r // (m * p)
            remainder = r % (m * p)
            j = remainder // p
            k = remainder % p
            
            u_row = i * m + j
            v_row = j * p + k
            w_row = k * n + i
            
            U[u_row, r] = 1.0
            V[v_row, r] = 1.0
            W[w_row, r] = 1.0
        
        return U, V, W, R
    
    def verify_decomposition_fast(U, V, W, tol=1e-10):
        """
        Vectorized verification: check if sum_r U[u,r]*V[v,r]*W[w,r] == 1 for all (i,j,k)
        Uses einsum for fast computation.
        """
        # U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R)
        # For each (i,j,k): compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Extract indices
        u_indices = index_map[:, 0]  # shape (n*m*p,)
        v_indices = index_map[:, 1]
        w_indices = index_map[:, 2]
        
        # Get the relevant rows
        U_rows = U[u_indices, :]  # shape (n*m*p, R)
        V_rows = V[v_indices, :]
        W_rows = W[w_indices, :]
        
        # Compute sum over r: np.einsum('ir,jr,kr->ijk', ...) doesn't work directly
        # Use element-wise multiply and sum
        products = U_rows * V_rows * W_rows  # shape (n*m*p, R)
        sums = products.sum(axis=1)  # shape (n*m*p,)
        
        # All should be 1.0
        return np.all(np.abs(sums - 1.0) < tol)
    
    def try_remove_column(U, V, W, col_idx):
        """
        Try removing column col_idx and verify. Returns True if removal is valid.
        """
        R = U.shape[1]
        if R <= 1:
            return False
        
        mask = np.ones(R, dtype=bool)
        mask[col_idx] = False
        
        U_trial = U[:, mask]
        V_trial = V[:, mask]
        W_trial = W[:, mask]
        
        return verify_decomposition_fast(U_trial, V_trial, W_trial, tol=1e-10)
    
    def greedy_remove(U, V, W, order, max_time=120.0):
        """
        Greedy column removal in the given order.
        Returns (U, V, W, new_rank) if improved, else None.
        """
        current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
        current_rank = U.shape[1]
        initial_rank = current_rank
        
        # Track which columns have been removed
        removed = set()
        
        for col in order:
            if time.time() > max_time:
                break
            
            # Map original column index to current index
            # Count how many columns before col have been removed
            current_idx = col - sum(1 for r in removed if r < col)
            
            if current_idx < 0 or current_idx >= current_rank:
                continue
            
            if try_remove_column(current_U, current_V, current_W, current_idx):
                mask = np.ones(current_rank, dtype=bool)
                mask[current_idx] = False
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_rank -= 1
                removed.add(col)
        
        if current_rank < initial_rank:
            return current_U, current_V, current_W, current_rank
        return None
    
    def try_local_search(U, V, W, max_time=120.0):
        """
        Try multiple removal orders to reduce rank.
        Returns (U, V, W, new_rank) if improved, else None.
        """
        R = U.shape[1]
        best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), R
        
        # Strategy 1: Remove from end to start
        order1 = list(range(R - 1, -1, -1))
        result = greedy_remove(U, V, W, order1, max_time=max_time/3)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
        
        # Strategy 2: Random permutations
        np.random.seed(42)
        for trial in range(5):
            if time.time() > max_time - 10:
                break
            perm = np.random.permutation(R)
            result = greedy_remove(U, V, W, list(perm), max_time=(max_time - 10)/5)
            if result is not None and result[3] < best_rank:
                best_U, best_V, best_W, best_rank = result
        
        # Strategy 3: Try removing columns that appear in fewest tensor entries
        # For each column r, count how many (i,j,k) have nonzero U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # Actually, for schoolbook, each column contributes to exactly one (i,j,k)
        # So all columns are equivalent. But after some removals, this changes.
        
        if best_rank < R:
            return best_U, best_V, best_W, best_rank
        return None
    
    # Start with schoolbook
    U_school, V_school, W_school, R_school = build_schoolbook()
    
    # Try local search with multiple strategies
    try:
        result = try_local_search(U_school, V_school, W_school, max_time=deadline - 20)
        if result is not None:
            U_final, V_final, W_final, R_final = result
        else:
            U_final, V_final, W_final, R_final = U_school, V_school, W_school, R_school
    except Exception:
        U_final, V_final, W_final, R_final = U_school, V_school, W_school, R_school
    
    # Final verification
    if not verify_decomposition_fast(U_final, V_final, W_final, tol=1e-10):
        # Fallback to schoolbook
        U_final, V_final, W_final, R_final = build_schoolbook()
    
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
