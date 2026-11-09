"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import product, combinations

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for evaluator kill at 150s

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify_exact(U, V, W):
        """
        Exact verification: check that sum_r U[:,r] V[:,r] W[:,r] == matmul tensor.
        The matmul tensor T[i,j,k] = 1 if j is the shared index, i.e., T[i,j,k]=1 for all valid (i,j,k)
        where the multiplication C[i,k] = sum_j A[i,j] B[j,k].
        
        The tensor decomposition requires:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
        
        We check this by computing the tensor and comparing.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        R = U_arr.shape[1]
        
        # Compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # This is a 3D tensor of shape (n, m, p)
        # U_reshaped: (n, m, R)
        # V_reshaped: (m, p, R)
        # W_reshaped: (p, n, R)
        
        U_r = U_arr.reshape(n, m, R)
        V_r = V_arr.reshape(m, p, R)
        W_r = W_arr.reshape(p, n, R)
        
        # For each (i,j,k): sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # Use einsum: 'ijr,jkr,kir->ijk'
        tensor = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        # Expected: all ones for valid indices (every i,j,k contributes 1)
        expected = np.ones((n, m, p))
        
        return np.allclose(tensor, expected, atol=1e-9)

    def verify_fast(U, V, W, num_tests=5):
        """Random sampling verification (faster for quick checks during search)."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        R = U_arr.shape[1]
        
        for _ in range(num_tests):
            A = np.random.randint(-3, 4, (n, m)).astype(float)
            B = np.random.randint(-3, 4, (m, p)).astype(float)
            C_expected = A @ B
            
            A_flat = A.flatten()
            S = U_arr.T @ A_flat  # Shape (R,)
            B_flat = B.flatten()
            T = V_arr.T @ B_flat  # Shape (R,)
            
            ST = S * T  # Shape (R,)
            W_reshaped = W_arr.reshape(p, n, R)
            C_computed = np.einsum('pnr,r->np', W_reshaped, ST)
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    def local_optimize(U, V, W, max_iter=2000):
        """Try to zero out entries while maintaining exact validity."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        R = U_arr.shape[1]
        
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    row, col = random.randint(0, U_arr.shape[0]-1), random.randint(0, R-1)
                    if U_arr[row, col] == 0:
                        continue
                    old_val = U_arr[row, col]
                    U_arr[row, col] = 0
                    if verify_fast(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), num_tests=3):
                        best_U = U_arr.copy()
                        improved = True
                    else:
                        U_arr[row, col] = old_val
                elif mat == 'V':
                    row, col = random.randint(0, V_arr.shape[0]-1), random.randint(0, R-1)
                    if V_arr[row, col] == 0:
                        continue
                    old_val = V_arr[row, col]
                    V_arr[row, col] = 0
                    if verify_fast(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), num_tests=3):
                        best_V = V_arr.copy()
                        improved = True
                    else:
                        V_arr[row, col] = old_val
                else:
                    row, col = random.randint(0, W_arr.shape[0]-1), random.randint(0, R-1)
                    if W_arr[row, col] == 0:
                        continue
                    old_val = W_arr[row, col]
                    W_arr[row, col] = 0
                    if verify_fast(U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), num_tests=3):
                        best_W = W_arr.copy()
                        improved = True
                    else:
                        W_arr[row, col] = old_val
            
            if not improved:
                break
        
        return best_U.tolist(), best_V.tolist(), best_W.tolist()

    def try_merge_pair(U, V, W, col_a, col_b):
        """
        Try to replace two schoolbook columns (col_a, col_b) with a single combined column.
        Returns (success, new_U_col, new_V_col, new_W_col) or (False, None, None, None).
        
        For the merge to work, we need:
        U_a * V_a * W_a + U_b * V_b * W_b = U_new * V_new * W_new
        
        In the schoolbook, each column has exactly one 1 in each matrix.
        Column for (i,j,k): U has 1 at (i,j), V has 1 at (j,k), W has 1 at (k,i).
        
        To merge two terms (i1,j1,k1) and (i2,j2,k2) into one term, we need
        a rank-1 decomposition of the sum of two rank-1 tensors.
        This is only possible if the two tensors are "compatible".
        
        Specifically, the sum T = T1 + T2 where T1 = u1 ⊗ v1 ⊗ w1 and T2 = u2 ⊗ v2 ⊗ w2.
        T can be written as u ⊗ v ⊗ w (rank 1) only in very special cases.
        
        One case: if they share the same i, same j, same k -> they're the same term (no merge needed).
        Another: if u1 = u2, v1 = v2, w1 = w2 -> same.
        If u1 = u2 but v1 != v2: T1+T2 = u1 ⊗ (v1*w1 + v2*w2) -- not rank 1 in general.
        
        Actually, for two outer products to sum to a single outer product, we need
        the sum to be rank 1 in each "slice". This is very restrictive.
        
        Let's think differently: we can try to find u, v, w such that
        u[i*m+j] * v[j*p+k] * w[k*n+i] = delta(i,j,k; i1,j1,k1) + delta(i,j,k; i2,j2,k2)
        
        This means u[i*m+j]*v[j*p+k]*w[k*n+i] is 1 at (i1,j1,k1) and (i2,j2,k2), and 0 elsewhere.
        
        For this to work with a single product u*v*w:
        - At (i1,j1,k1): u[i1*m+j1]*v[j1*p+k1]*w[k1*n+i1] = 1
        - At (i2,j2,k2): u[i2*m+j2]*v[j2*p+k2]*w[k2*n+i2] = 1
        - At all other (i,j,k): u[i*m+j]*v[j*p+k]*w[k*n+i] = 0
        
        The zero constraints are very strong. Let's check when this is possible.
        
        If i1=i2 and j1=j2 but k1!=k2:
        Then u[i1*m+j1] is shared. We need v[j1*p+k1]*w[k1*n+i1] = 1/u[i1*m+j1] and
        v[j1*p+k2]*w[k2*n+i1] = 1/u[i1*m+j1].
        And for any other k': v[j1*p+k']*w[k'*n+i1] = 0.
        And for any other (i,j): u[i*m+j] = 0 or v[j*p+k'] = 0 for all relevant k'.
        
        This gets complicated. Let me just try it computationally for the specific case.
        """
        # Extract the unit vectors for columns col_a and col_b
        u_a = U[:, col_a].copy()
        v_a = V[:, col_a].copy()
        w_a = W[:, col_a].copy()
        u_b = U[:, col_b].copy()
        v_b = V[:, col_b].copy()
        w_b = W[:, col_b].copy()
        
        # In schoolbook, each has exactly one 1. Find positions.
        # But after modifications, they might be different. Let's work generally.
        
        # We want to find u, v, w such that u⊗v⊗w = u_a⊗v_a⊗w_a + u_b⊗v_b⊗w_b
        # This is a tensor rank problem. The sum of two rank-1 tensors has rank <= 2.
        # It has rank 1 iff the two tensors are linearly dependent in a specific way.
        
        # Check if the sum is rank 1 by checking if we can factor it.
        # The tensor T = T_a + T_b of shape (n*m, m*p, p*n).
        # T is rank 1 iff T[i,j,k] = a[i]*b[j]*c[k] for some vectors a, b, c.
        
        # Flatten approach: reshape to (n*m, m*p*(p*n)) and check rank.
        # Actually, let's just try to solve it directly.
        
        # For the sum to be a single outer product, we need:
        # (u_a[i]*v_a[j]*w_a[k] + u_b[i]*v_b[j]*w_b[k]) = u[i]*v[j]*w[k] for all i,j,k
        
        # This is a system of equations. Let's check if it's solvable.
        # Pick a reference point where the sum is nonzero.
        
        sum_tensor = u_a[:, None, None] * v_a[None, :, None] * w_a[None, None, :] + \
                     u_b[:, None, None] * v_b[None, :, None] * w_b[None, None, :]
        
        # Check if sum_tensor is rank 1
        # Reshape to (n*m, m*p * p*n) and check if rank is 1
        flat = sum_tensor.reshape(n * m, m * p * p * n)
        try:
            rank = np.linalg.matrix_rank(flat, tol=1e-9)
        except:
            return False, None, None, None
        
        if rank > 1:
            return False, None, None, None
        
        # If rank is 1, extract the factors
        # Find a nonzero entry
        nonzero_idx = np.unravel_index(np.argmax(np.abs(flat)), flat.shape)
        if abs(flat[nonzero_idx]) < 1e-9:
            return False, None, None, None
        
        ref_val = flat[nonzero_idx]
        i_ref = nonzero_idx[0]
        jk_ref = nonzero_idx[1]
        
        # u[i_ref] * v[jk_ref part] * w[...] -- this is getting messy with the flattening.
        # Let's do it more carefully.
        
        # T[i, j, k] where i in [0,n*m), j in [0,m*p), k in [0,p*n)
        # Find a triple (i0, j0, k0) where T is nonzero
        T = sum_tensor
        # Find first nonzero
        found = False
        for i0 in range(n * m):
            if found:
                break
            for j0 in range(m * p):
                if found:
                    break
                for k0 in range(p * n):
                    if abs(T[i0, j0, k0]) > 1e-9:
                        found = True
                        break
        
        if not found:
            return False, None, None, None
        
        # T[i,j,k] = u[i]*v[j]*w[k]
        # u[i0]*v[j0]*w[k0] = T[i0,j0,k0]
        # We can set u[i0] = 1, then v[j0]*w[k0] = T[i0,j0,k0]
        # v[j0] = T[i0,j0,k0] / w[k0]
        # Need another equation. If there's another nonzero T[i0, j0, k1] with k1 != k0:
        # u[i0]*v[j0]*w[k1] = T[i0,j0,k1]
        # So w[k1]/w[k0] = T[i0,j0,k1]/T[i0,j0,k0]
        
        # Let's just try to factor it by assuming u[i0]=1
        u_new = np.zeros(n * m)
        v_new = np.zeros(m * p)
        w_new = np.zeros(p * n)
        
        u_new[i0] = 1.0
        v_new[j0] = T[i0, j0, k0]
        w_new[k0] = 1.0
        
        # Now verify and adjust
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    if abs(T[i, j, k]) > 1e-9:
                        if u_new[i] == 0 or v_new[j] == 0 or w_new[k] == 0:
                            # Need to determine values
                            pass
                    else:
                        # T[i,j,k] should be 0
                        if u_new[i] * v_new[j] * w_new[k] != 0:
                            return False, None, None, None
        
        # This approach is too slow and error-prone. Let's use a different strategy.
        # Since rank is 1, we can extract factors via SVD-like approach.
        # Reshape T to (n*m, m*p, p*n).
        # Fix k=k0: T[:,:,k0] = u * (v * w[k0]) = u * v' where v' = v * w[k0]
        # So T[:,:,k0] should be rank 1.
        mat_k0 = T[:, :, k0]
        try:
            U_svd, s, Vt = np.linalg.svd(mat_k0, full_matrices=False)
        except:
            return False, None, None, None
        
        if s[0] < 1e-9:
            return False, None, None, None
        if len(s) > 1 and s[1] > 1e-9 * s[0]:
            return False, None, None, None
        
        # u_part = U_svd[:,0] * sqrt(s[0])
        # v_part = Vt[0,:] * sqrt(s[0])
        u_part = U_svd[:, 0] * np.sqrt(s[0])
        v_part = Vt[0, :] * np.sqrt(s[0])
        
        # Now w[k0] * v = v_part, so v = v_part / w[k0]
        # Set w[k0] = 1, so v = v_part
        # Then for other k: T[:,:,k] = u * (v * w[k]) = u * v_part * w[k]
        # So T[:,:,k] / T[:,:,k0] = w[k] / w[k0] = w[k] (since w[k0]=1)
        # w[k] = T[i,j,k] / T[i,j,k0] for any i,j where T[i,j,k0] != 0
        
        w_new = np.zeros(p * n)
        w_new[k0] = 1.0
        for k in range(p * n):
            if k == k0:
                continue
            # Find (i,j) where T[i,j,k0] != 0
            found_ratio = False
            for i in range(n * m):
                for j in range(m * p):
                    if abs(T[i, j, k0]) > 1e-9:
                        w_new[k] = T[i, j, k] / T[i, j, k0]
                        found_ratio = True
                        break
                if found_ratio:
                    break
        
        u_new = u_part.copy()
        v_new = v_part.copy()
        
        # Verify
        reconstructed = u_new[:, None, None] * v_new[None, :, None] * w_new[None, None, :]
        if not np.allclose(reconstructed, T, atol=1e-9):
            return False, None, None, None
        
        return True, u_new, v_new, w_new

    def merge_search():
        """
        Start from schoolbook and try to merge pairs of columns to reduce rank.
        """
        base = get_schoolbook()
        U = np.array(base['U'], dtype=float)
        V = np.array(base['V'], dtype=float)
        W = np.array(base['W'], dtype=float)
        full_rank = n * m * p
        
        # Try all pairs of columns to see which can be merged
        mergeable_pairs = []
        for col_a in range(full_rank):
            for col_b in range(col_a + 1, full_rank):
                if time.time() > deadline - 10.0:
                    break
                success, u_new, v_new, w_new = try_merge_pair(U, V, W, col_a, col_b)
                if success:
                    mergeable_pairs.append((col_a, col_b, u_new, v_new, w_new))
            if time.time() > deadline - 10.0:
                break
        
        if not mergeable_pairs:
            return None
        
        # Greedily apply merges
        # Sort by some heuristic (e.g., prefer merges that involve more "overlapping" structure)
        random.shuffle(mergeable_pairs)
        
        # Track which columns are still active
        active_cols = set(range(full_rank))
        U_work = U.copy()
        V_work = V.copy()
        W_work = W.copy()
        
        num_merges = 0
        for col_a, col_b, u_new, v_new, w_new in mergeable_pairs:
            if time.time() > deadline - 10.0:
                break
            if col_a not in active_cols or col_b not in active_cols:
                continue
            
            # Replace columns col_a and col_b with a single new column
            # We'll put the new column at position col_a and mark col_b as inactive
            U_work[:, col_a] = u_new
            V_work[:, col_a] = v_new
            W_work[:, col_a] = w_new
            active_cols.discard(col_a)
            active_cols.discard(col_b)
            
            # Add a new active column at a free position
            # Find first inactive position
            new_col = None
            for c in range(full_rank):
                if c not in active_cols:
                    new_col = c
                    break
            if new_col is None:
                break
            
            active_cols.add(new_col)
            U_work[:, new_col] = u_new
            V_work[:, new_col] = v_new
            W_work[:, new_col] = w_new
            # Remove from col_a position
            U_work[:, col_a] = 0
            V_work[:, col_a] = 0
            W_work[:, col_a] = 0
            
            num_merges += 1
        
        if num_merges == 0:
            return None
        
        # Extract active columns
        active_list = sorted(active_cols)
        U_result = U_work[:, active_list]
        V_result = V_work[:, active_list]
        W_result = W_work[:, active_list]
        
        # Verify exact
        if verify_exact(U_result.tolist(), V_result.tolist(), W_result.tolist()):
            return {"U": U_result.tolist(), "V": V_result.tolist(), "W": W_result.tolist()}
        
        return None

    def random_search(target_rank, num_trials=100):
        for _ in range(num_trials):
            if time.time() > deadline - 10.0:
                break
            
            # Random entries from {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify_fast(U, V, W, num_tests=5):
                if verify_exact(U, V, W):
                    U_opt, V_opt, W_opt = local_optimize(U, V, W, max_iter=300)
                    return {"U": U_opt, "V": V_opt, "W": W_opt}
        
        return None

    best_result = get_schoolbook()
    best_rank = n * m * p
    
    # Try merge search first (most structured approach)
    if time.time() < deadline - 30.0:
        merge_result = merge_search()
        if merge_result is not None:
            best_result = merge_result
            best_rank = len(merge_result['U'][0])
    
    # Try decreasing ranks with random search
    for target_rank in range(44, 40, -1):
        if time.time() > deadline - 20.0:
            break
        
        result = random_search(target_rank, num_trials=50)
        if result is not None:
            best_result = result
            best_rank = target_rank
    
    # Final exact verification of best result
    if verify_exact(best_result['U'], best_result['V'], best_result['W']):
        return best_result
    
    return get_schoolbook()
# EVOLVE-BLOCK-END
