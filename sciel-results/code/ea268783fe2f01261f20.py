# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact QAP cost for a permutation p."""
    # p[i] is the location assigned to facility i
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    return int(np.sum(a * b[np.ix_(p, p)]))

def _solve(a, b):
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s kill
    
    # Ensure contiguous memory for speed
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    
    # 1. Constructive Start: Rank Matching
    # Facilities with high total flow should go to locations with low total distance (central)
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    # Sort facilities by flow (descending) and locations by distance (ascending)
    fac_order = np.argsort(-flow_sums, kind='stable')
    loc_order = np.argsort(dist_sums, kind='stable')
    
    p_init = np.empty(n, dtype=np.int64)
    p_init[fac_order] = loc_order
    
    # 2. Continuous Relaxation (Burer-Monteiro / Sinkhorn)
    # We optimize X in the Birkhoff Polytope (doubly stochastic matrices)
    # Objective: f(X) = sum_{i,j} a[i,j] * (X^T B X)[i,j] = sum_{i,j,k,l} a[i,j] X[i,k] B[k,l] X[j,l]
    # Gradient w.r.t X: G = 2 * A * (X^T B X) ... wait, let's derive carefully.
    # Cost C(X) = sum_{i,j} a_{ij} sum_{k,l} X_{ik} b_{kl} X_{jl}
    # dC/dX_{pq} = sum_{j} a_{pj} sum_{l} b_{ql} X_{jl} + sum_{i} a_{ip} sum_{k} b_{kq} X_{ik}
    #            = sum_{j,l} a_{pj} b_{ql} X_{jl} + sum_{i,k} a_{ip} b_{kq} X_{ik}
    # Let M = X^T B X. Then M_{jl} = sum_k X_{kj} B_{kl} X_{kl}? No.
    # M = X^T B X => M_{jl} = sum_k (X^T)_{jk} B_{kl} X_{lm} ... indices are tricky.
    # Let's use matrix calculus identity: d/dX tr(X^T A X B) = A X B + A^T X B^T.
    # Here C(X) = sum_{i,j} a_{ij} (X^T B X)_{ji} = tr(A (X^T B X)^T) = tr(A X^T B^T X) = tr(B^T X A X^T).
    # Let Y = X. C = tr(B^T Y A Y^T).
    # Gradient = B^T Y A^T + B Y A.
    # Since A and B are symmetric (QAP standard), Gradient = B^T Y A + B Y A = B (Y A + Y A) = 2 B Y A.
    # Wait, let's check indices.
    # C = sum_{i,j,k,l} a_{ij} X_{ik} b_{kl} X_{jl}
    # dC/dX_{pq} = sum_{j,l} a_{pj} b_{ql} X_{jl} + sum_{i,k} a_{ip} b_{kq} X_{ik}
    # Term 1: sum_j a_{pj} (sum_l b_{ql} X_{jl}) = (A^T X B^T)_{pq}? 
    # (X B^T)_{j l} = sum_k X_{j k} B_{k l} = sum_k X_{j k} B_{l k}.
    # Sum_j a_{p j} (X B^T)_{j l} ... this is getting messy.
    
    # Let's just compute the gradient numerically or via simple vectorization for n=100.
    # Actually, for n=100, we can afford a slightly less optimized gradient if it's robust.
    # Gradient G_{pq} = sum_{j,l} a_{pj} b_{ql} X_{jl} + sum_{i,k} a_{ip} b_{kq} X_{ik}
    # G = A @ X @ B.T + A.T @ X @ B  (assuming A, B symmetric, G = 2 * A @ X @ B)
    # Let's verify:
    # (A X B^T)_{pq} = sum_{j,l} A_{pj} X_{jl} (B^T)_{lq} = sum_{j,l} A_{pj} X_{jl} B_{ql}. Matches Term 1.
    # (A^T X B)_{pq} = sum_{i,k} (A^T)_{pi} X_{ik} B_{kq} = sum_{i,k} A_{ip} X_{ik} B_{kq}. Matches Term 2.
    # So if A, B symmetric, G = 2 * A @ X @ B.
    
    # We use projected gradient descent with Sinkhorn-Knopp normalization.
    # X_{t+1} = Sinkhorn(X_t - lr * G_t)
    
    # Initialize X from p_init
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p_init] = 1.0
    
    # Precompute A and B as float64 for linear algebra
    A = a
    B = b
    
    # Parameters
    lr = 0.01
    # Sinkhorn iterations
    sinkhorn_iters = 100
    
    def sinkhorn(K, iters=100):
        # Normalize rows and cols to sum to 1
        # K is positive (we add a small epsilon to X - lr*G to keep it positive? 
        # Actually, X - lr*G might have negative entries. 
        # Standard Burer-Monteiro uses exponential mapping or projection.
        # Simple approach: X_new = X - lr * G. Then project onto Birkhoff polytope.
        # Projecting onto Birkhoff polytope is hard (Sinkhorn is for positive matrices).
        # If X - lr*G has negatives, Sinkhorn fails.
        # Alternative: Use a reparameterization X = exp(Y) / sum(exp(Y)) (Softmax per row/col).
        # Or simply: X_new = X - lr * G, then clip negatives to 0, then Sinkhorn.
        # Clipping is a heuristic projection.
        
        K = K.copy()
        K[K < 0] = 0.0
        if np.sum(K) == 0:
            return np.ones((n, n)) / n
        for _ in range(iters):
            # Normalize rows
            row_sums = K.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            K = K / row_sums
            # Normalize cols
            col_sums = K.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1.0
            K = K / col_sums
        return K

    # Main loop
    best_cost = _qap_cost(a, b, p_init)
    best_p = p_init.copy()
    
    current_X = X.copy()
    current_cost = float('inf') # We don't track continuous cost easily, just discrete recovery
    
    # We will run the relaxation for a while, then recover permutation, then local search.
    # To avoid getting stuck in a bad local minimum of the continuous relaxation,
    # we can restart with different initial X or perturbations.
    
    # Strategy: 
    # 1. Run relaxation for T1 time.
    # 2. Recover permutation via Hungarian.
    # 3. Local Search (2-opt) on the permutation.
    # 4. If time remains, perturb and repeat.
    
    # Helper: Local Search 2-opt
    def local_search(a_int, b_int, p, deadline):
        n = len(p)
        cur = _qap_cost(a_int, b_int, p)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() >= deadline:
                    break
                pi = p[i]
                for j in range(i + 1, n):
                    if time.time() >= deadline:
                        break
                    pj = p[j]
                    # Compute delta for swap i, j
                    # S1 = sum_k a[i,k] * (b[pj,pk] - b[pi,pk])
                    # S2 = sum_k a[j,k] * (b[pi,pk] - b[pj,pk])
                    # S3 = sum_k a[k,i] * (b[pk,pj] - b[pk,pi])
                    # S4 = sum_k a[k,j] * (b[pk,pi] - b[pk,pj])
                    diff_row = b_int[pj, p] - b_int[pi, p]
                    diff_col = b_int[p, pj] - b_int[p, pi]
                    
                    S1 = np.dot(a_int[i, :], diff_row)
                    S2 = np.dot(a_int[j, :], -diff_row)
                    S3 = np.dot(a_int[:, i], diff_col)
                    S4 = np.dot(a_int[:, j], -diff_col)
                    
                    total = S1 + S2 + S3 + S4
                    
                    # Correction terms for double counting
                    b_pi_pi = b_int[pi, pi]
                    b_pj_pj = b_int[pj, pj]
                    b_pi_pj = b_int[pi, pj]
                    b_pj_pi = b_int[pj, pi]
                    
                    Rem_ii = a_int[i, i] * (b_pj_pi - b_pi_pi + b_pi_pj - b_pi_pi)
                    Rem_ij = a_int[i, j] * (b_pj_pj - b_pi_pj + b_pi_pi - b_pi_pj)
                    Rem_ji = a_int[j, i] * (b_pi_pi - b_pj_pi + b_pj_pj - b_pj_pi)
                    Rem_jj = a_int[j, j] * (b_pi_pj - b_pj_pj + b_pj_pi - b_pj_pj)
                    Rem = Rem_ii + Rem_ij + Rem_ji + Rem_jj
                    
                    Direct = a_int[i, j] * (b_pj_pi - b_pi_pj) + a_int[j, i] * (b_pi_pj - b_pj_pi)
                    
                    delta = total - Rem + Direct
                    
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cur += delta
                        improved = True
                        break
        return cur, p

    # Helper: Recover permutation from X
    def recover_permutation(X):
        # Minimize sum X[i,j] * C[i,j] where C is the cost matrix?
        # No, we want to find the permutation p that minimizes QAP cost.
        # The continuous relaxation minimizes the quadratic form.
        # The best discrete approximation is often found by taking the "closest" permutation.
        # One way: Linearize the objective around X?
        # Simpler: Use the Hungarian algorithm on a surrogate matrix.
        # Surrogate: The gradient G = 2 A X B. The "effective cost" of assigning i to j is roughly G[i,j]?
        # Or simply use the rows of X to find the best match?
        # Actually, the standard approach is to solve the assignment problem with weights derived from the current state.
        # Let's use the cost matrix C where C[i,j] = sum_k A[i,k] X[k,j] B[j,k]? No.
        # Let's use the "linearization" of the quadratic cost at X.
        # The gradient G_{ij} indicates the marginal cost.
        # Let's just use the Hungarian algorithm on the matrix of "penalties" or simply the original problem structure?
        # A common heuristic: Solve min sum_{i,j} X_{ij} W_{ij} where W is the gradient.
        # Or, simpler: Since X is close to a permutation, just take the argmax of each row?
        # That might violate column uniqueness.
        # Let's use Hungarian on the matrix M where M[i,j] = - (A @ X @ B)[i,j].
        # This minimizes the linear approximation of the cost.
        
        grad = 2.0 * (A @ X @ B)
        # We want to minimize the cost, so we maximize the "gain" or minimize the "cost".
        # The gradient points in the direction of increasing cost.
        # So we want to assign i to j where grad[i,j] is small (low marginal cost).
        # Hungarian minimizes sum C[i,p[i]].
        # So let C = grad.
        row_ind, col_ind = linear_sum_assignment(grad)
        p = np.empty(n, dtype=np.int64)
        for i, j in zip(row_ind, col_ind):
            p[i] = j
        return p

    # Main Search Loop
    while time.time() < deadline - 5.0:
        # Run Relaxation
        start_time = time.time()
        # Dynamic learning rate? Start high, decay?
        # For now, fixed lr.
        for step in range(500):
            if time.time() >= deadline - 5.0:
                break
            G = 2.0 * (A @ current_X @ B)
            current_X = current_X - lr * G
            current_X = sinkhorn(current_X, iters=20) # Fewer iters for speed, more steps
            
        # Recover Permutation
        p_rec = recover_permutation(current_X)
        
        # Local Search
        # Use int64 matrices for exact cost and delta
        a_int = a.astype(np.int64)
        b_int = b.astype(np.int64)
        
        cur_cost, p_refined = local_search(a_int, b_int, p_rec, deadline - 5.0)
        
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p_refined.copy()
            # Update current_X to be the one-hot of the new best?
            # This helps the relaxation start from a good point.
            current_X = np.zeros((n, n), dtype=np.float64)
            current_X[np.arange(n), best_p] = 1.0
            
            # If we improved, we can try to perturb and restart relaxation
            # to escape local minima.
            if best_cost < 273038: # If we beat the known best (unlikely but possible)
                 # Do more aggressive search
                 pass
                 
        else:
            # If no improvement, perturb the starting point for relaxation
            # to explore different basins.
            if rng.random() < 0.5:
                 # Random perturbation of best_p
                 p_pert = best_p.copy()
                 for _ in range(5):
                     i = rng.integers(n)
                     j = rng.integers(n)
                     if i != j:
                         p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
                 current_X = np.zeros((n, n), dtype=np.float64)
                 current_X[np.arange(n), p_pert] = 1.0
            else:
                 # Random start
                 p_rand = rng.permutation(n)
                 current_X = np.zeros((n, n), dtype=np.float64)
                 current_X[np.arange(n), p_rand] = 1.0
                 
        # Add some noise to X to avoid getting stuck in exact permutation vertices
        # which makes gradient zero.
        if time.time() < deadline - 10.0:
             current_X += rng.normal(0, 0.01, (n, n))
             current_X = sinkhorn(current_X, iters=50)

    return best_p.tolist()

# Global RNG for reproducibility
rng = np.random.default_rng(42)

def solve(a, b):
    return _solve(a, b)
# EVOLVE-BLOCK-END
