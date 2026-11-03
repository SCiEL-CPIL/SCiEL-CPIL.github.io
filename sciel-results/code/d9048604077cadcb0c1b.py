"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])


def _burer_monteiro(a, b, time_budget=100.0):
    """
    Solve QAP via Burer-Monteiro factorization: minimize ||B V V^T A||_F^2
    where X = V V^T approximates the doubly stochastic assignment matrix.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Symmetrize matrices (QAP cost is invariant to symmetrization)
    A = 0.5 * (a + a.T)
    B = 0.5 * (b + b.T)
    
    # Convert to float64 for numerical stability
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    
    # Initial V: use SVD of a rank-matching heuristic solution
    # Start with diagonal initialization scaled appropriately
    V = np.eye(n, dtype=np.float64) * np.sqrt(1.0 / n)
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    V += 0.01 * np.random.randn(n, n)
    
    # Precompute B @ V for efficient updates
    # We'll use a simple gradient descent with adaptive step size
    
    best_cost = np.inf
    best_V = V.copy()
    
    # Precompute some useful quantities
    # Gradient of f(V) = ||B V V^T A||_F^2 w.r.t. V
    # = 2 * B^T B V V^T A^T A + 2 * B V V^T A A^T B^T (by chain rule)
    # Actually: f(V) = tr((B V V^T A)(B V V^T A)^T)
    # df/dV = 2 * B^T (B V V^T A) A^T + 2 * B (B V V^T A) A^T B^T
    # Let's compute this more carefully:
    # f(V) = ||M||_F^2 where M = B V V^T A
    # df/dM = 2M
    # dM/dV = B (dV V^T + V dV^T) A
    # So df/dV = 2 * B^T M A^T V^T + 2 * B^T M^T A V^T
    # Wait, let me use the standard formula:
    # f(V) = tr(A^T B V V^T B^T A V V^T) -- no this is getting complex
    # 
    # Simpler: let C = B @ A (n x n)
    # f(V) = ||V^T C V||_F^2 (since tr((B V V^T A)^2) = tr(V^T A^T B^T B V A^T A V^T)... 
    # Actually for symmetric A,B: f(V) = tr(V^T B A V V^T B A V) = ||V^T (BA) V||_F^2
    
    C = B @ A  # n x n
    
    # f(V) = ||V^T C V||_F^2
    # Gradient: 2 * C V (V^T C V) + 2 * C^T V (V^T C^T V)^T
    # For symmetric C: 2 * C V (V^T C V) + 2 * C V (V^T C V)^T
    # = 2 * (C V (V^T C V) + C V (V^T C V)^T)
    
    step_size = 1.0 / (2 * np.max(np.abs(C)))
    best_step = step_size
    
    iteration = 0
    no_improve_count = 0
    patience = 200
    
    while time.time() < deadline - 5.0:  # Leave 5s for finalization
        iteration += 1
        
        # Compute current X = V V^T
        X = V @ V.T
        
        # Compute cost
        M = B @ X @ A
        cost = np.sum(M * M)
        
        if cost < best_cost:
            best_cost = cost
            best_V = V.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Compute gradient
        # f(V) = ||V^T C V||_F^2 where C = B @ A
        # grad = 2 * C @ V @ (V.T @ C @ V) + 2 * C.T @ V @ (V.T @ C.T @ V).T
        # Since C may not be symmetric, use general formula:
        # Let Y = V.T @ C @ V (n x n)
        # grad = 2 * (C @ V @ Y + C.T @ V @ Y.T)
        
        Y = V.T @ C @ V
        grad = 2.0 * (C @ V @ Y + C.T @ V @ Y.T)
        
        # Gradient descent step
        V_new = V - step_size * grad
        
        # Check if step improves (simple line search if not)
        X_new = V_new @ V_new.T
        M_new = B @ X_new @ A
        cost_new = np.sum(M_new * M_new)
        
        if cost_new < cost:
            V = V_new
        else:
            # Reduce step size
            step_size *= 0.5
            # Try again with smaller step
            V_new = V - step_size * grad
            X_new = V_new @ V_new.T
            M_new = B @ X_new @ A
            cost_new = np.sum(M_new * M_new)
            if cost_new < cost:
                V = V_new
                best_step = step_size
            else:
                # Restore step size if it was too small
                step_size = max(step_size * 2.0, best_step * 0.5)
        
        # Periodically try to increase step size
        if no_improve_count > patience:
            # Reset with a different initialization
            np.random.seed(42 + iteration)
            V = np.eye(n, dtype=np.float64) * np.sqrt(1.0 / n)
            V += 0.01 * np.random.randn(n, n)
            step_size = best_step
            no_improve_count = 0
        
        # Early exit if we've been stuck for too long
        if no_improve_count > patience * 5:
            break
    
    return best_V


def _extract_permutation_from_V(V):
    """
    Extract a permutation from the Burer-Monteiro factorization V.
    X = V V^T should approximate a permutation matrix.
    Use the Hungarian algorithm via scipy to find the best assignment.
    """
    from scipy.optimize import linear_sum_assignment
    
    X = V @ V.T
    
    # The QAP cost for a permutation p is sum_{i,j} a[i,j] b[p[i],p[j]]
    # This equals tr(P^T B P A) where P is the permutation matrix
    # We want to minimize this over permutation matrices P
    # 
    # From the relaxed solution X ≈ V V^T, we can get an initial assignment
    # by finding the closest permutation matrix to X
    # 
    # But we need to be careful: the cost function in terms of P is tr(P^T B P A)
    # which is NOT a standard assignment problem (it's quadratic in P)
    # 
    # Strategy: use X to get an initial permutation, then do local search
    # 
    # For the initial permutation: assign facility i to location argmax_j X[i,j]
    # But we need to ensure it's a valid permutation (each location used once)
    # Use Hungarian on -X to maximize sum of X[i, p[i]]
    
    # Actually, let's think about this differently.
    # The relaxed objective is min_X ||B X A||_F^2 s.t. X is doubly stochastic
    # The gradient of the QAP objective tr(P^T B P A) at a permutation P is:
    # g[i,j] = sum_k (B[i,k] A[k,j] + B[j,k] A[k,i]) -- this isn't quite right
    # 
    # Let's just use a greedy approach: for each facility i, assign to the 
    # location j that minimizes the marginal cost
    # 
    # Better: use the structure of the problem. The cost is:
    # sum_i sum_j a[i,j] b[p[i], p[j]]
    # 
    # For a given assignment p, the cost can be written as:
    # sum_i b[p[i], p[i]] * sum_j a[i,j] + sum_{i != j} a[i,j] b[p[i], p[j]]
    # 
    # Let's use a simple approach: start from the rank-matching heuristic,
    # then use the continuous relaxation to guide local search
    
    return None


def _local_search(a, b, p, time_budget=30.0):
    """
    2-opt local search with efficient delta computation.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + time_budget
    
    p = np.array(p, dtype=np.int64)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute row sums of a for faster delta computation
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    def swap_delta(i, j):
        """
        Compute cost change when swapping facilities at positions i and j.
        p[i] and p[j] are the current locations.
        """
        pi = p[i]
        pj = p[j]
        
        # The change in cost comes from all pairs involving i or j
        # Old contribution: sum_k a[i,k]*b[pi,p[k]] + sum_k a[k,i]*b[p[k],pi] 
        #                    + sum_k a[j,k]*b[pj,p[k]] + sum_k a[k,j]*b[p[k],pj]
        #                    - a[i,j]*b[pi,pj] - a[j,i]*b[pj,pi] (to avoid double counting)
        # 
        # New contribution: sum_k a[i,k]*b[pj,p[k]] + sum_k a[k,i]*b[p[k],pj]
        #                    + sum_k a[j,k]*b[pi,p[k]] + sum_k a[k,j]*b[p[k],pi]
        #                    - a[i,j]*b[pj,pi] - a[j,i]*b[pi,pj]
        
        # delta = new - old
        # = sum_k a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + sum_k a[k,i]*(b[p[k],pj] - b[p[k],pi])
        #   + sum_k a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) + sum_k a[k,j]*(b[p[k],pi] - b[p[k],pj])
        #   - a[i,j]*(b[pj,pi] - b[pi,pj]) - a[j,i]*(b[pi,pj] - b[pj,pi])
        # 
        # Note: b[pj,pi] - b[pi,pj] = -(b[pi,pj] - b[pj,pi])
        # So the last two terms: -a[i,j]*(b[pj,pi] - b[pi,pj]) - a[j,i]*(b[pi,pj] - b[pj,pi])
        # = (a[i,j] - a[j,i])*(b[pj,pi] - b[pi,pj])
        
        # For efficiency, precompute vectors
        # Let's compute directly for now, optimize if needed
        
        delta = 0
        for k in range(n):
            if k == i or k == j:
                continue
            pk = p[k]
            delta += a[i, k] * (b[pj, pk] - b[pi, pk])
            delta += a[k, i] * (b[pk, pj] - b[pk, pi])
            delta += a[j, k] * (b[pi, pk] - b[pj, pk])
            delta += a[k, j] * (b[pk, pi] - b[pk, pj])
        
        # Handle the i-j pair
        delta += (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
        
        return delta
    
    improving = True
    while improving and time.time() < deadline - 2.0:
        improving = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(n):
            for j in range(i + 1, n):
                d = swap_delta(i, j)
                if d < best_delta:
                    best_delta = d
                    best_i, best_j = i, j
        
        if best_i >= 0:
            # Perform the swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improving = True
    
    return best_p.tolist(), best_cost


def solve(a, b):
    """
    Solve QAP using Burer-Monteiro continuous relaxation + local search.
    """
    n = len(a)
    total_time_budget = 120.0
    start_time = time.time()
    
    # Phase 1: Continuous relaxation (Burer-Monteiro)
    bm_time = min(60.0, total_time_budget * 0.5)
    V = _burer_monteiro(a, b, time_budget=bm_time)
    
    # Extract initial permutation from V
    X = V @ V.T
    
    # Get initial assignment: for each row i, find the best column
    # Use Hungarian algorithm to get a valid permutation
    from scipy.optimize import linear_sum_assignment
    
    # We want to maximize sum X[i, p[i]] to get close to the relaxed solution
    # But we also need to consider the QAP cost structure
    # Simple approach: use the diagonal of X to guide initial assignment
    # Then refine with local search
    
    # Initial permutation: sort by row sums of X (facilities with high X[i,*] 
    # should go to locations with high X[*,j])
    row_sums_X = X.sum(axis=1)
    col_sums_X = X.sum(axis=0)
    
    # Rank-matching based on X
    facility_order = np.argsort(-row_sums_X, kind='stable')
    location_order = np.argsort(col_sums_X, kind='stable')
    
    p_init = np.empty(n, dtype=np.int64)
    p_init[facility_order] = location_order
    
    # Also try the rank-matching heuristic as an alternative start
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_heuristic = np.empty(n, dtype=np.int64)
    p_heuristic[heavy] = central
    
    # Choose the better starting point
    cost_bm = _qap_cost(a, b, p_init)
    cost_heur = _qap_cost(a, b, p_heuristic)
    
    if cost_bm <= cost_heur:
        p_start = p_init.copy()
        best_cost = cost_bm
    else:
        p_start = p_heuristic.copy()
        best_cost = cost_heur
    
    # Phase 2: Local search to refine
    remaining_time = total_time_budget - (time.time() - start_time) - 5.0  # 5s safety margin
    remaining_time = max(remaining_time, 5.0)
    
    p_best, cost_best = _local_search(a, b, p_start, time_budget=remaining_time)
    
    return p_best
# EVOLVE-BLOCK-END
