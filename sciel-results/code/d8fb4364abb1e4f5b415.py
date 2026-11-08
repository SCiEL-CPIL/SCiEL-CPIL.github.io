import time
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

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

    def validate(U, V, W, atol=1e-8):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_and_validate(U, V, W):
        # Try multiple snap granularities
        for denom in [1, 2, 3, 4, 5, 6, 10, 12]:
            scale = float(denom)
            U_s = np.round(U * scale) / scale
            V_s = np.round(V * scale) / scale
            W_s = np.round(W * scale) / scale
            if validate(U_s, V_s, W_s):
                return True, U_s, V_s, W_s
        return False, None, None

    def objective(x, T, rank):
        U = x[:U_dim * rank].reshape(U_dim, rank)
        V = x[U_dim * rank : U_dim * rank + V_dim * rank].reshape(V_dim, rank)
        W = x[U_dim * rank + V_dim * rank :].reshape(W_dim, rank)
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def get_trivial_solution():
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_sparse_v_decomposition(target_rank, time_limit=10.0):
        """
        Try to find a decomposition with rank = target_rank by fixing V to be sparse.
        We fix V to be a 25 x target_rank matrix where each row has exactly one 1.
        Then we solve for U and W using linear programming.
        """
        start = time.time()
        
        # V is 25 x target_rank
        # We want V to be "sparse" in the sense that each row has at most a few non-zeros,
        # or we fix a pattern.
        # A common technique: fix V to be a basis-like matrix.
        # But here V_dim=25, target_rank might be < 25.
        # If target_rank < 25, V cannot have each row with a 1 and be full column rank easily without overlaps.
        
        # Let's try a different approach:
        # Iterate over possible "support" patterns for V.
        # Actually, let's just use the linear structure directly.
        
        # The tensor equation is:
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # If we fix V, this is a linear system for U and W?
        # No, it's bilinear in U and W.
        # However, if we fix V, and assume U and W have small integer/half-integer values,
        # we can try to solve it.
        
        # Alternative: Fix V to be identity-like if target_rank >= 25? No, rank is small.
        
        # Let's try a specific structured V.
        # For M2,5,5, we know rank is likely around 30-40? No, trivial is 50.
        # Best known for M2,5,5 is likely lower.
        
        # Strategy:
        # 1. Fix V to be a random sparse matrix with entries in {-1, 0, 1}.
        # 2. Solve for U, W using least squares? No, that minimizes error, not exact.
        # 3. Use the fact that we want exact integer/half-integer solutions.
        
        # Let's try a "gauge fixing" approach.
        # We can multiply U by a diagonal matrix D and V by D^-1 to keep the product U*V*W invariant?
        # No, U[:,r] * V[:,r] * W[:,r] is the term.
        # If we scale U[:,r] by alpha and V[:,r] by 1/alpha, the product is unchanged.
        # This gauge freedom allows us to fix some entries of V.
        
        # Let's try to fix V to have 1s in specific positions.
        # If target_rank < 25, we can't have a full rank V with 1s in every row.
        
        # Let's try a different tactic:
        # Use the trivial solution and try to reduce rank by solving a sparse recovery problem.
        
        pass

    def try_merge_columns(U, V, W, col_a, col_b, alpha):
        """Try to merge column b into column a with coefficient alpha"""
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        U_new[:, col_a] = U_new[:, col_a] + alpha * U_new[:, col_b]
        V_new[:, col_a] = V_new[:, col_a] + alpha * V_new[:, col_b]
        W_new[:, col_a] = W_new[:, col_a] + alpha * W_new[:, col_b]
        
        # Remove column b
        U_new = np.delete(U_new, col_b, axis=1)
        V_new = np.delete(V_new, col_b, axis=1)
        W_new = np.delete(W_new, col_b, axis=1)
        
        return U_new, V_new, W_new

    def greedy_merge(U, V, W):
        """Greedily try to merge columns to reduce rank"""
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        rank = U_curr.shape[1]
        
        for _ in range(rank - 1):
            if time.time() > deadline - 5:
                break
            if U_curr.shape[1] <= 1:
                break
            
            best_merge = None
            best_error = float('inf')
            best_alpha = 0
            best_pair = None
            
            for a in range(U_curr.shape[1]):
                for b in range(a + 1, U_curr.shape[1]):
                    # Try several alpha values
                    for alpha in [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]:
                        U_new, V_new, W_new = try_merge_columns(U_curr, V_curr, W_curr, a, b, alpha)
                        prod = np.einsum('ar,br,cr->abc', U_new, V_new, W_new, optimize=True)
                        error = np.sum((prod - T) ** 2)
                        if error < best_error:
                            best_error = error
                            best_alpha = alpha
                            best_pair = (a, b)
            
            if best_pair is not None and best_error < 1e-6:
                a, b = best_pair
                U_curr, V_curr, W_curr = try_merge_columns(U_curr, V_curr, W_curr, a, b, best_alpha)
            else:
                break
        
        return U_curr, V_curr, W_curr

    def structured_init(rank, seed):
        """Create structured initialization based on matrix multiplication structure"""
        np.random.seed(seed)
        
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        # Assign each (i,j,k) triple to a rank component
        assigned = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    # Try to find an existing rank component that can cover this
                    found = False
                    for r in range(rank):
                        # Check if this rank component already has entries for i,j or j,k or k,i
                        if U[u_idx, r] != 0 and V[v_idx, r] == 0 and W[w_idx, r] == 0:
                            # Can add V and W entries
                            V[v_idx, r] = 1
                            W[w_idx, r] = 1
                            assigned[(i, j, k)] = r
                            found = True
                            break
                        elif U[u_idx, r] == 0 and V[v_idx, r] != 0 and W[w_idx, r] == 0:
                            U[u_idx, r] = 1
                            W[w_idx, r] = 1
                            assigned[(i, j, k)] = r
                            found = True
                            break
                        elif U[u_idx, r] == 0 and V[v_idx, r] == 0 and W[w_idx, r] != 0:
                            U[u_idx, r] = 1
                            V[v_idx, r] = 1
                            assigned[(i, j, k)] = r
                            found = True
                            break
                    
                    if not found:
                        # Assign to a new rank component if available
                        for r in range(rank):
                            if U[:, r].sum() == 0:
                                U[u_idx, r] = 1
                                V[v_idx, r] = 1
                                W[w_idx, r] = 1
                                assigned[(i, j, k)] = r
                                break
        
        return U, V, W

    U50, V50, W50 = get_trivial_solution()
    
    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # First, try greedy merging from the trivial solution
    U_merged, V_merged, W_merged = greedy_merge(U50, V50, W50)
    if validate(U_merged, V_merged, W_merged):
        valid, U_s, V_s, W_s = snap_and_validate(U_merged, V_merged, W_merged)
        if valid:
            best_rank = U_s.shape[1]
            best_U, best_V, best_W = U_s, V_s, W_s

    # New Strategy: Sparse V Fixing + Linear Programming
    # We fix V to be a sparse matrix with entries in {-1, 0, 1}.
    # Specifically, we try to assign each of the 25 rows of V to one of the 'rank' columns,
    # such that the resulting system is solvable for U and W with integer/half-integer entries.
    
    # Actually, a more direct approach for small ranks:
    # If we fix V, the equation T = sum_r U_r V_r W_r^T is linear in U_r and W_r?
    # No, it's a sum of outer products.
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # If V is fixed, for each r, let u_r = U[:,r], v_r = V[:,r], w_r = W[:,r].
    # This is still non-linear if we don't fix U or W.
    
    # However, if we fix V to be very sparse (e.g. 1s), we can try to solve for U and W.
    # Let's try a specific structured V:
    # V[b, r] = 1 if b % rank == r else 0. (Cyclic assignment)
    # This makes V a "permutation-like" matrix (if rank divides 25, or close).
    
    def try_cyclic_v(target_rank):
        """
        Fix V to be cyclic: V[b, r] = 1 if b % target_rank == r else 0.
        Then solve for U and W.
        Since V is fixed, T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
        For a given b, V[b,r] is non-zero for at most one r (if target_rank divides 25 or similar).
        Actually, if b % target_rank == r, then V[b,r]=1.
        So for a fixed b, the term is U[a, b%rank] * 1 * W[c, b%rank].
        Thus T[a, b, c] = U[a, b%rank] * W[c, b%rank].
        
        This implies that for all a, c, T[a, b, c] must be rank 1 in (a,c) for fixed b.
        But T is the matrix multiplication tensor.
        T[a, b, c] = 1 if a < n, b < m, c < p and a*m+b == b*p+c? No.
        T[i*m+j, j*p+k, k*n+i] = 1.
        
        Let's check if this cyclic structure works.
        For fixed b=j*p+k (where j in 0..4, k in 0..4):
        r = (j*p+k) % target_rank.
        We need U[i*m+j, r] * W[k*n+i, r] = 1 for all valid i, k?
        Wait, the indices are:
        a = i*m + j
        b = j*p + k
        c = k*n + i
        
        So for a fixed b = j*p+k, the non-zero entries of T are at (a, c) where
        a = i*m+j and c = k*n+i for i in 0..n-1.
        So for fixed b, T[:, b, :] has non-zeros at (i*m+j, k*n+i) for i=0..1.
        
        If we use cyclic V, then for this b, only column r = b % target_rank of U and W is active.
        So U[:, r] and W[:, r] must satisfy:
        U[i*m+j, r] * W[k*n+i, r] = 1 for i=0,1.
        
        This means U[i*m+j, r] and W[k*n+i, r] must be non-zero and multiply to 1.
        Since we want integer/half-integer, they could be 1 or -1.
        
        Let's see if we can assign U and W consistently for all b.
        """
        U = np.zeros((U_dim, target_rank))
        W = np.zeros((W_dim, target_rank))
        
        # For each b in 0..24:
        for b in range(V_dim):
            r = b % target_rank
            # Determine i, j, k from b
            j = b // p
            k = b % p
            
            # The non-zero entries in T for this b are at:
            # a = i*m + j
            # c = k*n + i
            # for i in 0..n-1
            
            # We need U[a, r] * W[c, r] = 1
            # Let's try to set U[a, r] = 1 and W[c, r] = 1.
            # But we must check for conflicts if multiple b's map to the same r
            # and share the same a or c index.
            
            for i in range(n):
                a = i * m + j
                c = k * n + i
                
                # If U[a, r] is already set, it must be 1 (or -1 if we allow signs)
                # If it's 0, we set it to 1.
                # If it's already 1, we are good.
                # If it's -1, we have a conflict if we want product 1.
                
                # Let's just try setting them to 1.
                if U[a, r] == 0:
                    U[a, r] = 1
                elif U[a, r] != 1:
                    return None # Conflict
                
                if W[c, r] == 0:
                    W[c, r] = 1
                elif W[c, r] != 1:
                    return None # Conflict
        
        # Check if this assignment works
        V = np.zeros((V_dim, target_rank))
        for b in range(V_dim):
            r = b % target_rank
            V[b, r] = 1
            
        if validate(U, V, W):
            valid, U_s, V_s, W_s = snap_and_validate(U, V, W)
            if valid:
                return U_s, V_s, W_s
        return None

    # Try cyclic V for various ranks
    for target_rank in range(25, 50): # Start from lower
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
            
        result = try_cyclic_v(target_rank)
        if result is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
            break # Found a valid one, try to go lower? No, we went up.
            # Actually, we should try to find the MINIMAL rank.
            # So we should iterate target_rank from low to high.
            
    # Let's redo the loop correctly: from low to high
    # But we already have a trivial solution of 50.
    # We want to find rank < 50.
    
    # Reset best to trivial if not improved by cyclic (which likely failed)
    # Let's try a range of ranks from, say, 30 to 50.
    
    # Actually, the cyclic structure might be too restrictive.
    # Let's try a more general sparse V search.
    
    # For now, if cyclic didn't work, fall back to the previous optimization.
    
    # Try structured initializations and optimization for lower ranks
    for target_rank in range(49, 35, -1):
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
            
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                    break
                    
            # Try structured initialization
            U_init, V_init, W_init = structured_init(target_rank, seed)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Optimization with tighter tolerances
                res = minimize(
                    objective, 
                    x0, 
                    args=(T, target_rank), 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                
                U_opt = res.x[:U_dim * target_rank].reshape(U_dim, target_rank)
                V_opt = res.x[U_dim * target_rank : U_dim * target_rank + V_dim * target_rank].reshape(V_dim, target_rank)
                W_opt = res.x[U_dim * target_rank + V_dim * target_rank :].reshape(W_dim, target_rank)
                
                valid, U_s, V_s, W_s = snap_and_validate(U_opt, V_opt, W_opt)
                if valid:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_s, V_s, W_s
                    found = True
                    break
            except Exception:
                pass

        if found:
            continue

    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
