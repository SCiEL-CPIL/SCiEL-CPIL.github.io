import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    dim_U, dim_V, dim_W = n * m, m * p, p * n  # 12, 16, 12
    
    def build_target_tensor():
        T = np.zeros((dim_U, dim_V, dim_W))
        for row_a in range(3):
            for col_a in range(4):
                for col_c in range(4):
                    i = row_a * 4 + col_a
                    j = col_a * 4 + col_c
                    k = col_c * 3 + row_a
                    T[i, j, k] = 1.0
        return T
    
    def get_base_solution():
        """Build the rank-44 Strassen-based solution."""
        rank = 44
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen part (cols 0-6)
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        # Extra part (cols 7-10)
        U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
        U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
        U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
        U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
        
        def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            col_offset = block_idx * 11
            for r in range(3):
                for c in range(2):
                    local_a_idx = r * 2 + c
                    global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                    U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                    V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                    W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

        add_block(0, 0, 0, 0, 0, 0, 0)
        add_block(1, 0, 2, 2, 0, 0, 0)
        add_block(2, 0, 0, 0, 2, 0, 2)
        add_block(3, 0, 2, 2, 2, 0, 2)
        
        return U, V, W
    
    def verify_exact(U, V, W, T):
        """Verify exact decomposition using einsum."""
        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(computed, T, atol=1e-9)
    
    def try_reduce_rank(U, V, W, T, target_rank, time_limit):
        """
        Try to reduce rank by removing pairs of columns and checking
        if the residual can be represented with fewer terms.
        
        Strategy: For each pair of columns to remove, compute the residual
        tensor, then try to fit it using SVD-based low-rank approximation
        of the associated matrix slices.
        """
        current_rank = U.shape[1]
        
        if current_rank <= target_rank:
            return U, V, W
        
        # Try removing pairs of columns
        # The residual after removing columns r1, r2 is:
        # T_residual = T - sum_{r not in {r1,r2}} u_r v_r w_r
        # We need to check if T_residual can be written as a sum of rank-1 terms
        # using only the remaining columns (adjusted)
        
        # More practical: try removing 1 column at a time, then check if
        # the residual is "close" to being representable with the remaining structure
        
        # Better approach: for each pair (r1, r2), remove them and try to
        # find coefficients for the remaining columns that still reproduce T.
        # This is a linear system if we fix the supports.
        
        # Actually, the cleanest approach: 
        # 1. Remove a pair of columns
        # 2. The residual R = T - (product without those columns)
        # 3. Try to express R as a single rank-1 term (or check if it's zero)
        
        # Let's try: remove pairs and see if residual has rank <= 1
        # (i.e., can be absorbed into one new term or is zero)
        
        all_cols = list(range(current_rank))
        pairs_checked = 0
        max_pairs = 2000  # Limit to avoid timeout
        
        for r1, r2 in combinations(all_cols, 2):
            if time.time() > time_limit:
                break
            pairs_checked += 1
            if pairs_checked > max_pairs:
                break
            
            # Remove columns r1 and r2
            keep = [c for c in all_cols if c != r1 and c != r2]
            U_keep = U[:, keep]
            V_keep = V[:, keep]
            W_keep = W[:, keep]
            
            # Compute what the remaining terms produce
            prod_keep = np.einsum('ir,jr,kr->ijk', U_keep, V_keep, W_keep)
            residual = T - prod_keep
            
            # Check if residual is zero (within tolerance)
            if np.max(np.abs(residual)) < 1e-9:
                # We can remove both columns!
                return U_keep, V_keep, W_keep
            
            # Check if residual is rank-1: residual[i,j,k] = a_i * b_j * c_k
            # Test by checking if all 2x2 minors of slices vanish
            # Quick test: flatten and check SVD
            # Reshape residual as matrix for each fixed k
            # If rank-1 tensor, then for each k, residual[:,:,k] = a * (b*c_k)^T
            # so each slice has rank <= 1
            
            # Quick check: pick a few slices
            nonzero_slices = np.any(residual != 0, axis=(0,1))
            if not np.any(nonzero_slices):
                return U_keep, V_keep, W_keep
            
            # Check if residual is rank 1 by looking at matrix unfolding
            # Unfold along k: shape (12*16, 12)
            mat = residual.reshape(dim_U * dim_V, dim_W)
            # Compute rank via SVD (just check singular values)
            try:
                s = np.linalg.svd(mat, compute_uv=False)
                if len(s) == 0 or s[0] < 1e-9:
                    return U_keep, V_keep, W_keep
                if len(s) >= 2 and s[1] / s[0] < 1e-7:
                    # Essentially rank 1
                    # Try to extract the rank-1 factors
                    # mat ≈ sigma * u * v^T
                    # residual[i,j,k] ≈ sigma * u[i*16+j] * v[k]
                    # This gives us a new rank-1 term
                    # But we'd need to add it back, so net reduction is 1
                    # For now, just note it
                    pass
            except:
                pass
        
        return None
    
    def try_single_column_removal(U, V, W, T, time_limit):
        """Try removing single columns and checking if residual is representable."""
        current_rank = U.shape[1]
        
        for r in range(current_rank):
            if time.time() > time_limit:
                break
            
            keep = [c for c in range(current_rank) if c != r]
            U_keep = U[:, keep]
            V_keep = V[:, keep]
            W_keep = W[:, keep]
            
            prod_keep = np.einsum('ir,jr,kr->ijk', U_keep, V_keep, W_keep)
            residual = T - prod_keep
            
            if np.max(np.abs(residual)) < 1e-9:
                return U_keep, V_keep, W_keep
            
            # Check if residual is rank-1
            mat = residual.reshape(dim_U * dim_V, dim_W)
            try:
                s = np.linalg.svd(mat, compute_uv=False)
                if len(s) > 0 and s[0] > 1e-9:
                    if len(s) >= 2 and s[1] / s[0] < 1e-7:
                        # Residual is rank-1, so we need 1 new term
                        # Net change: remove 1, add 1 = no gain
                        # But if we can merge with existing, maybe gain
                        pass
            except:
                pass
        
        return None
    
    def refine_with_svd(U, V, W, T, max_iter=100):
        """
        Try to reduce rank by iteratively:
        1. Finding the column that contributes least
        2. Removing it
        3. Trying to fit the residual with SVD
        """
        U_cur = U.copy()
        V_cur = V.copy()
        W_cur = W.copy()
        
        for _ in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            current_rank = U_cur.shape[1]
            if current_rank <= 30:
                break
            
            # Find the column with smallest "importance"
            # Importance = norm of contribution
            importances = []
            for r in range(current_rank):
                contrib = np.outer(np.outer(U_cur[:, r], V_cur[:, r]), W_cur[:, r])
                importances.append(np.sum(np.abs(contrib)))
            
            # Try removing the least important column
            min_idx = np.argmin(importances)
            keep = [c for c in range(current_rank) if c != min_idx]
            U_try = U_cur[:, keep]
            V_try = V_cur[:, keep]
            W_try = W_cur[:, keep]
            
            prod_try = np.einsum('ir,jr,kr->ijk', U_try, V_try, W_try)
            residual = T - prod_try
            
            if np.max(np.abs(residual)) < 1e-9:
                U_cur, V_cur, W_cur = U_try, V_try, W_try
                continue
            
            # Check if residual is rank-1
            mat = residual.reshape(dim_U * dim_V, dim_W)
            try:
                s = np.linalg.svd(mat, compute_uv=False)
                if len(s) > 0 and s[0] > 1e-9:
                    if len(s) >= 2 and s[1] / s[0] < 1e-6:
                        # Rank-1 residual: extract factors
                        # mat = sigma * u * v^T
                        # residual[i,j,k] = sigma * u[i*16+j] * v[k]
                        # We need u_vec of length 12, v_vec of length 16, w_vec of length 12
                        # But the unfolding mixes i and j...
                        # This approach doesn't directly give us the 3-way factors
                        pass
            except:
                pass
        
        return U_cur, V_cur, W_cur
    
    # Build base solution
    U, V, W = get_base_solution()
    T = build_target_tensor()
    
    # Verify base solution
    if not verify_exact(U, V, W, T):
        # Fallback to trivial
        rank = 48
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        for row_a in range(3):
            for col_a in range(4):
                for col_c in range(4):
                    idx = (row_a * 4 + col_a) * 4 + col_c
                    i = row_a * 4 + col_a
                    j = col_a * 4 + col_c
                    k = col_c * 3 + row_a
                    U[i, idx] = 1
                    V[j, idx] = 1
                    W[k, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try single column removal
    if time.time() < deadline - 30:
        result = try_single_column_removal(U, V, W, T, deadline - 30)
        if result is not None:
            U2, V2, W2 = result
            if verify_exact(U2, V2, W2, T):
                best_U, best_V, best_W = U2, V2, W2
                best_rank = U2.shape[1]
    
    # Try pair column removal
    if time.time() < deadline - 20 and best_rank > 42:
        result = try_reduce_rank(best_U, best_V, best_W, T, best_rank - 1, deadline - 20)
        if result is not None:
            U2, V2, W2 = result
            if verify_exact(U2, V2, W2, T):
                best_U, best_V, best_W = U2, V2, W2
                best_rank = U2.shape[1]
    
    # Try iterative refinement
    if time.time() < deadline - 15 and best_rank > 40:
        U3, V3, W3 = refine_with_svd(best_U, best_V, best_W, T, max_iter=50)
        if verify_exact(U3, V3, W3, T):
            if U3.shape[1] < best_rank:
                best_U, best_V, best_W = U3, V3, W3
                best_rank = U3.shape[1]
    
    # Clean up: remove zero columns
    nonzero = np.any(best_U != 0, axis=0) | np.any(best_V != 0, axis=0) | np.any(best_W != 0, axis=0)
    best_U = best_U[:, nonzero]
    best_V = best_V[:, nonzero]
    best_W = best_W[:, nonzero]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
