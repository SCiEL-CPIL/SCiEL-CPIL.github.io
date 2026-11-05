"""Rewritten program: tensor rank minimization via iterative compression from schoolbook."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    
    Strategy: Start with schoolbook (rank 60), then iteratively compress by finding
    linear dependencies among the factor vectors. Use SVD-based rank reduction and
    local search to find lower-rank exact decompositions.
    
    The multiplication tensor T[i,j,k] = 1 for all i,j,k (each A[i,j] pairs with B[j,k]).
    We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 140  # 10s safety margin
    
    # Initialize with schoolbook decomposition (rank = n*m*p = 60)
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
    
    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_compress(U, V, W, max_iter=50):
        """
        Try to reduce rank by finding and eliminating linearly dependent columns.
        Uses the fact that if column r is a linear combination of others,
        we can potentially eliminate it.
        """
        U, V, W = U.copy(), V.copy(), W.copy()
        current_rank = U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            if current_rank <= 1:
                break
            
            # Try to find a column that can be eliminated
            # A column r is redundant if there exist coefficients alpha_r' 
            # such that for all i,j,k:
            # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
            #   = sum_{r' != r} alpha[r'] * U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r']
            
            # This is a nonlinear condition. Instead, use a heuristic:
            # Check if column r of U is in the span of other columns,
            # AND column r of V is in the span of other columns,
            # AND column r of W is in the span of other columns.
            
            best_col = -1
            best_score = float('inf')
            
            for r in range(current_rank):
                if time.time() > deadline:
                    break
                
                # Check if U[:, r] is in span of U[:, [0..r-1, r+1..]]
                U_other = np.delete(U, r, axis=1)
                V_other = np.delete(V, r, axis=1)
                W_other = np.delete(W, r, axis=1)
                
                # Check rank
                rank_U_other = np.linalg.matrix_rank(U_other)
                rank_U_with = np.linalg.matrix_rank(np.hstack([U_other, U[:, r:r+1]]))
                
                rank_V_other = np.linalg.matrix_rank(V_other)
                rank_V_with = np.linalg.matrix_rank(np.hstack([V_other, V[:, r:r+1]]))
                
                rank_W_other = np.linalg.matrix_rank(W_other)
                rank_W_with = np.linalg.matrix_rank(np.hstack([W_other, W[:, r:r+1]]))
                
                # If all three are dependent, this column might be eliminable
                if (rank_U_with == rank_U_other and 
                    rank_V_with == rank_V_other and 
                    rank_W_with == rank_W_other):
                    # Try to eliminate and verify
                    U_new = U_other
                    V_new = V_other
                    W_new = W_other
                    if verify(U_new, V_new, W_new, tol=1e-8):
                        best_col = r
                        best_score = 0
                        break
            
            if best_col >= 0:
                U = np.delete(U, best_col, axis=1)
                V = np.delete(V, best_col, axis=1)
                W = np.delete(W, best_col, axis=1)
                current_rank = U.shape[1]
            else:
                # No simple elimination found; try more sophisticated approach
                # Use random perturbations and SVD-based compression
                break
        
        return U, V, W, current_rank
    
    # First pass: simple column elimination
    U, V, W, rank = try_compress(U, V, W, max_iter=100)
    
    # Second pass: Try to find lower-rank decompositions using local search
    # Represent the tensor as a 3-way array and use tensor rank algorithms
    
    def tensor_to_matricization(U, V, W, mode):
        """Create matricization of the tensor defined by (U, V, W)."""
        r = U.shape[1]
        if mode == 0:  # i-mode: (n*m) x (m*p)*(p*n)
            T = np.zeros((n * m, m * p * p * n))
            for i in range(n):
                for j in range(m):
                    for jp in range(m * p):
                        for kn in range(p * n):
                            val = 0.0
                            for rr in range(r):
                                val += U[i * m + j, rr] * V[jp, rr] * W[kn, rr]
                            T[i * m + j, jp * (p * n) + kn] = val
            return T
        elif mode == 1:  # j-mode: (m*p) x (n*m)*(p*n)
            T = np.zeros((m * p, n * m * p * n))
            for jp in range(m * p):
                for im in range(n * m):
                    for kn in range(p * n):
                        val = 0.0
                        for rr in range(r):
                            val += U[im, rr] * V[jp, rr] * W[kn, rr]
                        T[jp, im * (p * n) + kn] = val
            return T
        else:  # k-mode: (p*n) x (n*m)*(m*p)
            T = np.zeros((p * n, n * m * m * p))
            for kn in range(p * n):
                for im in range(n * m):
                    for jp in range(m * p):
                        val = 0.0
                        for rr in range(r):
                            val += U[im, rr] * V[jp, rr] * W[kn, rr]
                        T[kn, im * (m * p) + jp] = val
            return T
    
    # Use SVD-based approximation to find lower rank
    # The target tensor has all entries = 1 in the appropriate positions
    
    # Create the target tensor as a 3D array
    T_target = np.ones((n, m, p))
    
    # Current tensor from our decomposition
    def current_tensor(U, V, W):
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 0.0
                    for r in range(U.shape[1]):
                        T[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        return T
    
    # Try to find a lower-rank decomposition using iterative method
    # Start from current solution and try to reduce rank
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try different initializations and local optimizations
    np.random.seed(42)
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Try to find a rank-r' decomposition where r' < current_rank
        target_rank = max(20, rank - 5 * (trial + 1))
        if target_rank < 1:
            break
        
        # Random initialization for lower rank
        U_rand = np.random.randn(n * m, target_rank) * 0.5
        V_rand = np.random.randn(m * p, target_rank) * 0.5
        W_rand = np.random.randn(p * n, target_rank) * 0.5
        
        # Gradient descent to minimize ||T_decomposed - T_target||^2
        lr = 0.01
        for gd_iter in range(200):
            if time.time() > deadline:
                break
            
            T_current = current_tensor(U_rand, V_rand, W_rand)
            error = np.sum((T_current - T_target) ** 2)
            
            if error < 1e-12:
                break
            
            # Compute gradients (approximate)
            eps = 1e-6
            grad_U = np.zeros_like(U_rand)
            grad_V = np.zeros_like(V_rand)
            grad_W = np.zeros_like(W_rand)
            
            # Only compute a few gradient components for speed
            num_samples = min(50, n * m * target_rank)
            indices = np.random.choice(n * m * target_rank, size=num_samples, replace=False)
            
            for idx in indices:
                im = idx // target_rank
                rr = idx % target_rank
                
                # Gradient w.r.t. U[im, rr]
                U_plus = U_rand.copy()
                U_plus[im, rr] += eps
                T_plus = current_tensor(U_plus, V_rand, W_rand)
                err_plus = np.sum((T_plus - T_target) ** 2)
                grad_U[im, rr] = (err_plus - error) / eps
                
                # Gradient w.r.t. V
                # Sample a few V entries
                num_v_samples = min(20, m * p * target_rank)
                v_indices = np.random.choice(m * p * target_rank, size=num_v_samples, replace=False)
                for v_idx in v_indices:
                    jp = v_idx // target_rank
                    rr_v = v_idx % target_rank
                    V_plus = V_rand.copy()
                    V_plus[jp, rr_v] += eps
                    T_plus_v = current_tensor(U_rand, V_plus, W_rand)
                    err_plus_v = np.sum((T_plus_v - T_target) ** 2)
                    grad_V[jp, rr_v] = (err_plus_v - error) / eps
                
                # Gradient w.r.t. W
                num_w_samples = min(20, p * n * target_rank)
                w_indices = np.random.choice(p * n * target_rank, size=num_w_samples, replace=False)
                for w_idx in w_indices:
                    kn = w_idx // target_rank
                    rr_w = w_idx % target_rank
                    W_plus = W_rand.copy()
                    W_plus[kn, rr_w] += eps
                    T_plus_w = current_tensor(U_rand, V_rand, W_plus)
                    err_plus_w = np.sum((T_plus_w - T_target) ** 2)
                    grad_W[kn, rr_w] = (err_plus_w - error) / eps
            
            # Update
            U_rand -= lr * grad_U
            V_rand -= lr * grad_V
            W_rand -= lr * grad_W
        
        # Check if we found a valid solution
        if verify(U_rand, V_rand, W_rand, tol=1e-8):
            if target_rank < best_rank:
                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                best_rank = target_rank
    
    # Also try the structured Strassen approach as a fallback
    def strassen_approach():
        """Use Strassen for 2x2x2 blocks."""
        row_splits = [2, 1]
        mid_splits = [2, 2]
        col_splits = [2, 2, 1]
        
        def get_range(splits, idx):
            start = sum(splits[:idx])
            end = start + splits[idx]
            return start, end
        
        strassen_A = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, -1],
        ]
        strassen_B = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ]
        strassen_W = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0],
        ]
        
        def schoolbook_decomp(ra, ma, mb, pa):
            assert ma == mb
            rank = ra * ma * pa
            U = []
            V = []
            W = []
            for i in range(ra):
                for j in range(ma):
                    for k in range(pa):
                        u = [0] * (ra * ma)
                        u[i * ma + j] = 1
                        U.append(u)
                        v = [0] * (ma * pa)
                        v[j * pa + k] = 1
                        V.append(v)
                        w = [0] * (pa * ra)
                        w[k * ra + i] = 1
                        W.append(w)
            return U, V, W, rank
        
        U_rows = []
        V_rows = []
        W_rows = []
        
        for i_idx in range(len(row_splits)):
            for j_idx in range(len(mid_splits)):
                for k_idx in range(len(col_splits)):
                    ra = row_splits[i_idx]
                    ma = mid_splits[j_idx]
                    mb = mid_splits[j_idx]
                    pa = col_splits[k_idx]
                    
                    i_start, i_end = get_range(row_splits, i_idx)
                    j_start, j_end = get_range(mid_splits, j_idx)
                    k_start, k_end = get_range(col_splits, k_idx)
                    
                    if ra == 2 and ma == 2 and pa == 2:
                        for r in range(7):
                            u = [0] * (n * m)
                            for a_i in range(2):
                                for a_j in range(2):
                                    u[(i_start + a_i) * m + (j_start + a_j)] = strassen_A[r][a_i * 2 + a_j]
                            U_rows.append(u)
                            
                            v = [0] * (m * p)
                            for b_j in range(2):
                                for b_k in range(2):
                                    v[(j_start + b_j) * p + (k_start + b_k)] = strassen_B[r][b_j * 2 + b_k]
                            V_rows.append(v)
                            
                            w = [0] * (p * n)
                            for c_i in range(2):
                                for c_k in range(2):
                                    w[(k_start + c_k) * n + (i_start + c_i)] = strassen_W[c_i * 2 + c_k][r]
                            W_rows.append(w)
                    else:
                        U_blk, V_blk, W_blk, blk_rank = schoolbook_decomp(ra, ma, mb, pa)
                        for r in range(blk_rank):
                            u = [0] * (n * m)
                            for idx in range(ra * ma):
                                a_i = idx // ma
                                a_j = idx % ma
                                u[(i_start + a_i) * m + (j_start + a_j)] = U_blk[r][idx]
                            U_rows.append(u)
                            
                            v = [0] * (m * p)
                            for idx in range(ma * pa):
                                b_j = idx // pa
                                b_k = idx % pa
                                v[(j_start + b_j) * p + (k_start + b_k)] = V_blk[r][idx]
                            V_rows.append(v)
                            
                            w = [0] * (p * n)
                            for idx in range(pa * ra):
                                c_k = idx // ra
                                c_i = idx % ra
                                w[(k_start + c_k) * n + (i_start + c_i)] = W_blk[r][idx]
                            W_rows.append(w)
        
        return U_rows, V_rows, W_rows
    
    # Get Strassen result
    U_str, V_str, W_str = strassen_approach()
    strassen_rank = len(U_str)
    
    if verify(np.array(U_str), np.array(V_str), np.array(W_str), tol=1e-8):
        if strassen_rank < best_rank:
            best_U = np.array(U_str)
            best_V = np.array(V_str)
            best_W = np.array(W_str)
            best_rank = strassen_rank
    
    # Final verification and return
    if not verify(best_U, best_V, best_W, tol=1e-8):
        # Fallback to schoolbook
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

