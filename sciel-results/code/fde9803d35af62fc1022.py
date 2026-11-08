import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin

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
        for factor in [2, 4, 1]:
            U_s = np.round(U * factor) / factor
            V_s = np.round(V * factor) / factor
            W_s = np.round(W * factor) / factor
            if validate(U_s, V_s, W_s):
                return True, U_s, V_s, W_s
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

    def als_optimize(U, V, W, max_iter=100, tol=1e-12, time_limit=None):
        """Alternating Least Squares for tensor decomposition."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        prev_loss = float('inf')
        
        for it in range(max_iter):
            if time_limit and time.time() > time_limit:
                break
            
            # Update U: for each a, solve linear system
            for a in range(U_dim):
                # T[a,:,:] = sum_r U[a,r] V[:,r] W[:,r]
                # For fixed a: T[a,:,:] = U[a,:] @ (V ⊙ W) where ⊙ is element-wise outer
                # Reshape: T[a] is (V_dim, W_dim), V⊙W is (V_dim, W_dim, R)
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Fix a: T[a,:,:] = sum_r U[a,r] (V[:,r] ⊗ W[:,r])
                # So T[a].reshape(-1) = U[a,:] @ (V⊙W).reshape(R, -1)
                VW = (V * W).T  # (R, V_dim, W_dim) -> reshape
                VW_flat = VW.reshape(R, -1)  # (R, V_dim*W_dim)
                Ta_flat = T[a].reshape(-1)
                # Solve: U[a,:] @ VW_flat = Ta_flat
                try:
                    U[a, :] = np.linalg.lstsq(VW_flat.T, Ta_flat, rcond=None)[0]
                except:
                    pass
            
            # Update V
            for b in range(V_dim):
                UW = (U * W).T  # (R, U_dim, W_dim)
                UW_flat = UW.reshape(R, -1)
                Tb_flat = T[:, b, :].reshape(-1)
                try:
                    V[b, :] = np.linalg.lstsq(UW_flat.T, Tb_flat, rcond=None)[0]
                except:
                    pass
            
            # Update W
            for c in range(W_dim):
                UV = (U * V).T  # (R, U_dim, V_dim)
                UV_flat = UV.reshape(R, -1)
                Tc_flat = T[:, :, c].reshape(-1)
                try:
                    W[c, :] = np.linalg.lstsq(UV_flat.T, Tc_flat, rcond=None)[0]
                except:
                    pass
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            loss = np.sum((prod - T) ** 2)
            
            if abs(prev_loss - loss) < tol:
                break
            if loss < 1e-14:
                break
            prev_loss = loss
        
        return U, V, W

    def gradient_optimize(U, V, W, max_iter=300, lr=0.01, time_limit=None):
        """Gradient descent with adaptive learning rate."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        prev_loss = float('inf')
        
        for it in range(max_iter):
            if time_limit and time.time() > time_limit:
                break
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            
            if loss < 1e-14:
                break
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            grad_norm = np.sqrt(np.sum(gradU**2) + np.sum(gradV**2) + np.sum(gradW**2))
            if grad_norm < 1e-15:
                break
            
            # Adaptive learning rate
            effective_lr = lr * min(1.0, 1e-8 / max(grad_norm, 1e-15))
            
            U -= effective_lr * gradU
            V -= effective_lr * gradV
            W -= effective_lr * gradW
            
            if loss > prev_loss * 1.01:
                # Revert if loss increased significantly
                U += effective_lr * gradU
                V += effective_lr * gradV
                W += effective_lr * gradW
                lr *= 0.5
            else:
                lr = max(lr * 1.01, 0.001)
            
            prev_loss = loss
        
        return U, V, W

    def try_reduce_rank(U46, V46, W46, target_rank, method='als'):
        """Try to reduce rank by removing columns and re-optimizing."""
        current_rank = U46.shape[1]
        if current_rank <= target_rank:
            return U46, V46, W46
        
        # Compute column importance scores
        col_scores = []
        for c in range(current_rank):
            # Count non-zeros
            nz_U = np.count_nonzero(U46[:, c])
            nz_V = np.count_nonzero(V46[:, c])
            nz_W = np.count_nonzero(W46[:, c])
            score = nz_U + nz_V + nz_W
            col_scores.append(score)
        
        col_scores = np.array(col_scores)
        # Remove columns with lowest scores first
        removal_order = np.argsort(col_scores)
        
        for remove_count in range(1, current_rank - target_rank + 1):
            if time.time() > deadline - 20:
                break
            
            cols_to_remove = removal_order[:remove_count]
            mask = [i for i in range(current_rank) if i not in cols_to_remove]
            
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()
            
            # Try multiple optimization approaches
            for opt_method in ['als', 'grad']:
                if time.time() > deadline - 15:
                    break
                
                U_try = U_test.copy()
                V_try = V_test.copy()
                W_try = W_test.copy()
                
                # Add small noise to escape local minima
                U_try += np.random.randn(*U_try.shape) * 0.01
                V_try += np.random.randn(*V_try.shape) * 0.01
                W_try += np.random.randn(*W_try.shape) * 0.01
                
                if opt_method == 'als':
                    U_opt, V_opt, W_opt = als_optimize(U_try, V_try, W_try, max_iter=50, time_limit=deadline - 10)
                else:
                    U_opt, V_opt, W_opt = gradient_optimize(U_try, V_try, W_try, max_iter=200, lr=0.005, time_limit=deadline - 10)
                
                valid, U_s, V_s, W_s = try_snap_validate(U_opt, V_opt, W_opt)
                if valid:
                    return U_s, V_s, W_s
            
            # If we found a valid solution at this rank, return it
            # Otherwise continue removing more columns
        
        return None, None, None

    def merge_columns(U, V, W, c1, c2, max_iter=50):
        """Try to merge two columns into one."""
        R = U.shape[1]
        mask = [i for i in range(R) if i not in (c1, c2)]
        
        # Start with the sum of the two columns
        U_merged = U[:, c1] + U[:, c2]
        V_merged = V[:, c1] + V[:, c2]
        W_merged = W[:, c1] + W[:, c2]
        
        U_test = np.column_stack([U[:, mask], U_merged])
        V_test = np.column_stack([V[:, mask], V_merged])
        W_test = np.column_stack([W[:, mask], W_merged])
        
        # Optimize
        U_opt, V_opt, W_opt = als_optimize(U_test, V_test, W_test, max_iter=max_iter, time_limit=deadline - 10)
        
        valid, U_s, V_s, W_s = try_snap_validate(U_opt, V_opt, W_opt)
        if valid:
            return U_s, V_s, W_s
        return None, None, None

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback: trivial decomposition
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

    np.random.seed(42)
    best_result = None

    # Strategy 1: Try removing single columns (rank 45)
    for target_rank in [45, 44, 43]:
        if time.time() > deadline - 30:
            break
        
        result = try_reduce_rank(U46, V46, W46, target_rank, method='als')
        if result[0] is not None:
            U_r, V_r, W_r = result
            if validate(U_r, V_r, W_r):
                best_result = (U_r, V_r, W_r)
                break
    
    # Strategy 2: Try column merging
    if best_result is None and time.time() < deadline - 25:
        # Find pairs of columns with similar support patterns
        for c1 in range(46):
            if time.time() > deadline - 20:
                break
            for c2 in range(c1 + 1, 46):
                if time.time() > deadline - 18:
                    break
                
                # Check if columns have overlapping support
                support_U1 = set(np.nonzero(U46[:, c1])[0])
                support_U2 = set(np.nonzero(U46[:, c2])[0])
                support_V1 = set(np.nonzero(V46[:, c1])[0])
                support_V2 = set(np.nonzero(V46[:, c2])[0])
                support_W1 = set(np.nonzero(W46[:, c1])[0])
                support_W2 = set(np.nonzero(W46[:, c2])[0])
                
                overlap = (len(support_U1 & support_U2) + len(support_V1 & support_V2) + 
                          len(support_W1 & support_W2))
                
                if overlap >= 3:
                    result = merge_columns(U46, V46, W46, c1, c2, max_iter=30)
                    if result[0] is not None:
                        U_r, V_r, W_r = result
                        if validate(U_r, V_r, W_r):
                            best_result = (U_r, V_r, W_r)
                            break
            if best_result is not None:
                break

    if best_result is not None:
        U_r, V_r, W_r = best_result
        return {
            "U": [[float(x) for x in row] for row in U_r],
            "V": [[float(x) for x in row] for row in V_r],
            "W": [[float(x) for x in row] for row in W_r]
        }

    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
