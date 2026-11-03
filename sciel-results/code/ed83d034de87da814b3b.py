"""Combined strategy: Rank Matching + FISTA Relaxation + Efficient Local Search"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    n = len(a)
    
    # Ensure float64 for precision in intermediate steps
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
    def compute_cost(p):
        # p[i] is location of facility i
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Vectorized
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    def rank_matching_init(seed=0):
        rng = np.random.default_rng(seed)
        # Add small noise to break ties and create diversity
        a_sums = a.sum(axis=1) + rng.uniform(0, 1e-3 * np.max(a.sum(axis=1)), n)
        b_sums = b.sum(axis=1) + rng.uniform(0, 1e-3 * np.max(b.sum(axis=1)), n)
        
        # High flow to low distance
        fac_order = np.argsort(-a_sums, kind="stable")
        loc_order = np.argsort(b_sums, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[fac_order] = loc_order
        return p

    # Phase 1: Generate diverse initial solutions
    candidates = []
    best_p = None
    best_cost = np.inf
    
    # Try a few different rank matching seeds
    for seed in [0, 1, 2, 3]:
        p = rank_matching_init(seed)
        cost = compute_cost(p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
        candidates.append((cost, p))
    
    # Also try a random initialization
    p_rand = np.random.permutation(n)
    cost_rand = compute_cost(p_rand)
    if cost_rand < best_cost:
        best_cost = cost_rand
        best_p = p_rand.copy()
    candidates.append((cost_rand, p_rand))
    
    # Phase 2: FISTA Relaxation on the best candidate
    # Objective: f(X) = tr(A_sym X B_sym X^T)
    # Gradient: G = A_sym X B_sym + B_sym X A_sym
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    
    # Start from the best integer solution
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), best_p] = 1.0
    
    # FISTA Parameters
    # Lipschitz constant estimate for gradient G(X) = A X B + B X A
    # ||G||_L <= 2 * ||A|| * ||B||
    # We use a small step size to be safe, or backtracking.
    # For n=150, a fixed step size often works well if normalized.
    # Let's normalize A and B to have unit Frobenius norm to stabilize step size
    A_norm = A_sym / np.linalg.norm(A_sym, 'fro')
    B_norm = B_sym / np.linalg.norm(B_sym, 'fro')
    
    # Rescale objective? No, we just need the gradient direction.
    # The actual cost is computed on the integer solution.
    # We use the relaxed gradient to guide the projection.
    
    alpha = 1.0 # Step size for normalized matrices
    t_k = 1.0
    X_prev = X.copy()
    
    # Precompute for gradient
    # G = A_norm X B_norm + B_norm X A_norm
    
    max_fista_iters = 500
    for k in range(max_fista_iters):
        if time.time() - start_time > deadline - 30.0:
            break
            
        # Gradient
        AX = A_norm @ X
        G = AX @ B_norm + B_norm @ AX
        
        # FISTA Update
        t_prev = t_k
        t_k = (1 + np.sqrt(1 + 4 * t_prev * t_prev)) / 2
        Y = X + (t_prev - 1) / t_k * (X - X_prev)
        
        # Gradient Step
        X_new = Y - alpha * G
        
        # Project onto Birkhoff Polytope
        # Use Hungarian algorithm on -X_new to find the closest permutation matrix
        row_ind, col_ind = linear_sum_assignment(-X_new)
        X_proj = np.zeros((n, n), dtype=np.float64)
        X_proj[row_ind, col_ind] = 1.0
        
        # Update best solution from relaxation
        p_relax = np.empty(n, dtype=np.int64)
        p_relax[row_ind] = col_ind
        cost_relax = compute_cost(p_relax)
        
        if cost_relax < best_cost:
            best_cost = cost_relax
            best_p = p_relax.copy()
            candidates.append((cost_relax, p_relax.copy()))
        
        X_prev = X
        X = X_proj
        
        # Early stopping if no improvement for a while?
        # Or just rely on time limit.
        
    # Phase 3: Local Search (2-Swap)
    # We perform local search on the best candidate found so far.
    p = best_p.copy()
    current_cost = best_cost
    
    # Precompute for local search efficiency
    # We will use the delta formula derived above.
    # To make it faster, we can precompute row sums of b? No, we need specific pairs.
    
    # Let's implement a vectorized 2-swap pass.
    # For each i, we want to find j that minimizes delta.
    # Delta(i, j) = T1(i, j) + T2(i, j) + T3(i, j)
    # T1(i, j) = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j], k] - b[p[i], k])
    # T2(i, j) = sum_{k != i,j} (a[k,i] - a[k,j]) * (b[k, p[j]] - b[k, p[i]])
    # T3(i, j) = (a[i,j] + a[j,i]) * (b[p[i], p[j]] - b[p[j], p[i]])
    
    # Note: T1 and T2 can be computed as:
    # T1(i, j) = (a[i] - a[j]) @ (b[p[j]] - b[p[i]]) - (a[i,j] - a[j,j])*(b[p[j],j] - b[p[i],j]) - (a[i,i] - a[j,i])*(b[p[j],i] - b[p[i],i])
    # This is getting complicated to exclude i and j exactly in a vectorized way without masking.
    # Given n=150, a simple loop over i, and vectorized j is O(n^3) but with low constant.
    # 150 * 150 * 150 = 3.3M operations. In NumPy, this is very fast (<0.1s).
    
    improved = True
    max_ls_iters = 1000
    for ls_iter in range(max_ls_iters):
        if time.time() - start_time > deadline - 5.0:
            break
        if not improved:
            break
            
        improved = False
        best_delta_in_pass = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            li = p[i]
            # For all j > i to avoid duplicates
            js = np.arange(i + 1, n)
            if len(js) == 0: continue
            lj = p[js]
            
            # T3: Cross terms
            # b[li, lj] - b[lj, li]
            diff_b = b[li, lj] - b[lj, li]
            t3 = (a[i, js] + a[js, i]) * diff_b
            
            # T1: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[lj, k] - b[li, k])
            # Let v_i = a[i, :], v_j = a[js, :] (shape n-1, n)
            # Let u_lj = b[lj, :] (shape n-1, n)
            # Let u_li = b[li, :] (shape n,)
            
            # We need to exclude k=i and k=j from the sum.
            # It's easier to compute the full sum and subtract the i and j terms.
            
            # Full sum for T1:
            # S1 = sum_k (a[i,k] - a[j,k]) * (b[lj, k] - b[li, k])
            # = (a[i] - a[j]) @ (b[lj] - b[li])
            
            v_diff = a[i, None, :] - a[js, :] # (n-1, n)
            u_diff = b[lj, None, :] - b[li, None, :] # (n-1, n)
            s1_full = np.sum(v_diff * u_diff, axis=1) # (n-1,)
            
            # Subtract terms where k=i
            # Term k=i: (a[i,i] - a[j,i]) * (b[lj, i] - b[li, i])
            t_ki = (a[i, i] - a[js, i]) * (b[lj, i] - b[li, i])
            
            # Subtract terms where k=j (which is js[m])
            # Term k=j: (a[i, j] - a[j, j]) * (b[lj, j] - b[li, j])
            # Note: b[lj, j] is b[p[j], j]. b[li, j] is b[p[i], j].
            # a[js, js] is not element-wise diagonal of a, it's a[j, j].
            # We need a[j, j] for each j.
            a_jj = np.diag(a[js, :]) # This is wrong, diag of a[js,:] is not a[j,j]
            # a[j, j] is a[js[m], js[m]]
            a_jj_vals = a[js, js]
            
            t_kj = (a[i, js] - a_jj_vals) * (b[lj, js] - b[li, js])
            
            t1 = s1_full - t_ki - t_kj
            
            # T2: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[k, lj] - b[k, li])
            # Let w_i = a[:, i], w_j = a[:, js] (shape n, n-1)
            # Let z_lj = b[:, lj] (shape n, n-1)
            # Let z_li = b[:, li] (shape n,)
            
            w_diff = a[:, i, None] - a[:, js] # (n, n-1)
            z_diff = b[:, lj, None] - b[:, li, None] # (n, n-1)
            s2_full = np.sum(w_diff * z_diff, axis=0) # (n-1,)
            
            # Subtract k=i
            # (a[i,i] - a[i,j]) * (b[i, lj] - b[i, li])
            t2_ki = (a[i, i] - a[i, js]) * (b[i, lj] - b[i, li])
            
            # Subtract k=j
            # (a[j,i] - a[j,j]) * (b[j, lj] - b[j, li])
            t2_kj = (a[js, i] - a_jj_vals) * (b[js, lj] - b[js, li])
            
            t2 = s2_full - t2_ki - t2_kj
            
            deltas = t1 + t2 + t3
            
            # Find best improving swap
            min_idx = np.argmin(deltas)
            min_val = deltas[min_idx]
            
            if min_val < best_delta_in_pass:
                best_delta_in_pass = min_val
                best_i = i
                best_j = js[min_idx]
                
        if best_delta_in_pass < -1e-10:
            # Apply swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += int(best_delta_in_pass)
            improved = True
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
