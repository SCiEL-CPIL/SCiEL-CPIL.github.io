"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh


def solve(a, b):
    """
    Solve the Quadratic Assignment Problem for tai256c.
    Uses spectral bisection for initialization, vectorized 2-opt local search,
    and iterated local search with adaptive restarts.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin within 120s limit

    n = len(a)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    def compute_cost(p):
        """Compute the exact QAP cost for assignment p."""
        B_p = b[np.ix_(p, p)]
        return int(np.sum(a * B_p))

    def spectral_init():
        """
        Spectral bisection-based initialization:
        1. Build flow graph Laplacian from a
        2. Build distance graph Laplacian from b
        3. Use eigenvectors to create a low-cost initial assignment
        """
        # Flow graph: edge weights = a[i,j]
        # Distance graph: edge weights = b[i,j]
        
        # For spectral initialization, we want to align high-flow pairs 
        # with low-distance pairs
        
        # Compute row sums to identify "heavy" facilities and "central" locations
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        
        # Rank matching as baseline
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        
        # Try to improve with spectral information
        # Compute a few eigenvectors of the flow Laplacian
        try:
            # Create sparse Laplacian from flow matrix
            A_sparse = a.copy()
            A_sparse -= A_sparse.sum(axis=1)[:, None]  # This isn't quite right for Laplacian
            # Actually, let's just use the eigenvector approach differently
            
            # Use the fact that for QAP, we want to minimize sum a[i,j]*b[p[i],p[j]]
            # This is equivalent to finding a permutation that aligns the two matrices
            # in a certain sense
            
            # Simple spectral approach: use SVD-like alignment
            # Center the matrices
            a_centered = a - a.mean()
            b_centered = b - b.mean()
            
            # Get principal components
            Ua, sa, Va = np.linalg.svd(a_centered, full_matrices=False)
            Ub, sb, Vb = np.linalg.svd(b_centered, full_matrices=False)
            
            # Align using the top components
            # This gives us a soft assignment that we can refine
            M = Ua @ Vb.T
            
            # Convert soft assignment to hard permutation using Hungarian
            # Negate because we want to maximize alignment
            cost_matrix = -M
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            p_spectral = np.empty(n, dtype=np.int64)
            p_spectral[row_ind] = col_ind
            
            # Compare costs and pick better
            cost_spectral = compute_cost(p_spectral)
            cost_rank = compute_cost(p)
            
            if cost_spectral < cost_rank:
                p = p_spectral
                best_cost = cost_spectral
            else:
                best_cost = cost_rank
                
        except Exception:
            best_cost = compute_cost(p)
        
        return p, best_cost

    def vectorized_swap_delta(p, i, j):
        """
        Compute the cost change for swapping facilities i and j.
        Returns the delta (new_cost - old_cost).
        """
        pi = p[i]
        pj = p[j]
        
        # For each k != i, j, the change is:
        # a[i,k]*b[pj,p[k]] + a[j,k]*b[pi,p[k]] - a[i,k]*b[pi,p[k]] - a[j,k]*b[pj,p[k]]
        # + a[k,i]*b[p[k],pj] + a[k,j]*b[p[k],pi] - a[k,i]*b[p[k],pi] - a[k,j]*b[p[k],pj]
        # 
        # Vectorized form:
        # Let delta_a_row = a[i,:] - a[j,:]
        # Let delta_a_col = a[:,i] - a[:,j]
        # Let delta_b_row = b[pj,:] - b[pi,:]
        # Let delta_b_col = b[:,pj] - b[:,pi]
        # 
        # The change is:
        # sum_k delta_a_row[k] * delta_b_row[p[k]] + sum_k delta_a_col[k] * delta_b_col[p[k]]
        # But we need to exclude k=i and k=j, and handle the self-interaction terms
        
        delta_a_row = a[i, :] - a[j, :]
        delta_a_col = a[:, i] - a[:, j]
        delta_b_row = b[pj, :] - b[pi, :]
        delta_b_col = b[:, pj] - b[:, pi]
        
        # Full sum including i and j
        term1 = np.dot(delta_a_row, delta_b_row[p])
        term2 = np.dot(delta_a_col, delta_b_col[p])
        
        # Subtract the contributions from k=i and k=j
        # When k=i: delta_a_row[i] * delta_b_row[p[i]] + delta_a_col[i] * delta_b_col[p[i]]
        # When k=j: delta_a_row[j] * delta_b_row[p[j]] + delta_a_col[j] * delta_b_col[p[j]]
        
        # But actually, the formula above double-counts the (i,j) and (j,i) terms
        # Let's use a cleaner approach
        
        # The correct formula for swap delta:
        # delta = (a[i,:]-a[j,:]) @ (b[pj,:]-b[pi,:])[p] 
        #       + (a[:,i]-a[:,j]) @ (b[:,pj]-b[:,pi])[p]
        #       - 2*(a[i,j]*b[pi,pj] - a[i,j]*b[pi,pi] - a[i,j]*b[pj,pj] + a[i,j]*b[pj,pi])
        # 
        # Actually, let's just compute it directly for correctness:
        
        # Create a copy of p with the swap
        p_new = p.copy()
        p_new[i], p_new[j] = p_new[j], p_new[i]
        
        # The change in cost comes from:
        # 1. Terms involving i or j in the first index
        # 2. Terms involving i or j in the second index
        
        # For term (i,k): a[i,k]*b[pi,p[k]] -> a[i,k]*b[pj,p[k]] for k!=j, and a[i,j]*b[pi,pi] -> a[i,j]*b[pj,pj]
        # For term (j,k): a[j,k]*b[pj,p[k]] -> a[j,k]*b[pi,p[k]] for k!=i, and a[j,i]*b[pj,pj] -> a[j,i]*b[pi,pi]
        # For term (k,i): a[k,i]*b[p[k],pi] -> a[k,i]*b[p[k],pj] for k!=j, and a[j,i]*b[pj,pi] -> a[j,i]*b[pj,pj]
        # For term (k,j): a[k,j]*b[p[k],pj] -> a[k,j]*b[p[k],pi] for k!=i, and a[i,j]*b[pi,pj] -> a[i,j]*b[pi,pi]
        
        # This is getting complex. Let's use the efficient O(n) formula:
        
        # The change in cost when swapping i and j:
        # = sum_{k != i,j} [a[i,k]*b[pj,p[k]] + a[j,k]*b[pi,p[k]] + a[k,i]*b[p[k],pj] + a[k,j]*b[p[k],pi]
        #                    - a[i,k]*b[pi,p[k]] - a[j,k]*b[pj,p[k]] - a[k,i]*b[p[k],pi] - a[k,j]*b[p[k],pj]]
        # + [a[i,j]*b[pj,pj] + a[j,i]*b[pi,pi] + a[j,i]*b[pj,pj] + a[i,j]*b[pi,pi]
        #     - a[i,j]*b[pi,pi] - a[j,i]*b[pj,pj] - a[j,i]*b[pi,pi] - a[i,j]*b[pj,pj]]
        
        # The last bracket simplifies to 0, so we only need the sum over k != i,j
        
        # Vectorized:
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        p_masked = p[mask]
        
        delta = 0
        for k_idx in range(n):
            if k_idx == i or k_idx == j:
                continue
            pk = p[k_idx]
            delta += (a[i, k_idx] * b[pj, pk] + a[j, k_idx] * b[pi, pk] + 
                      a[k_idx, i] * b[pk, pj] + a[k_idx, j] * b[pk, pi] -
                      a[i, k_idx] * b[pi, pk] - a[j, k_idx] * b[pj, pk] -
                      a[k_idx, i] * b[pk, pi] - a[k_idx, j] * b[pk, pj])
        
        return delta

    def swap_delta_fast(p, i, j):
        """
        Efficient O(n) swap delta calculation.
        """
        pi = p[i]
        pj = p[j]
        
        # Use the vectorized formula
        # delta = sum_{k != i,j} (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]]) 
        #       + sum_{k != i,j} (a[k,i]-a[k,j])*(b[p[k],pj]-b[p[k],pi])
        
        # But we need to be careful about the (i,j) and (j,i) terms
        
        # Let's compute it as:
        # For each k, the contribution change is:
        # a[i,k]*b[pj,p[k]] - a[i,k]*b[pi,p[k]] + a[j,k]*b[pi,p[k]] - a[j,k]*b[pj,p[k]]
        # + a[k,i]*b[p[k],pj] - a[k,i]*b[p[k],pi] + a[k,j]*b[p[k],pi] - a[k,j]*b[p[k],pj]
        
        # = (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],pj]-b[p[k],pi])
        
        # Sum over all k, but when k=i or k=j, we need to handle specially
        
        # For k=i:
        # (a[i,i]-a[j,i])*(b[pj,p[i]]-b[pi,p[i]]) + (a[i,i]-a[i,j])*(b[p[i],pj]-b[p[i],pi])
        # = (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi]) + (a[i,i]-a[i,j])*(b[pi,pj]-b[pi,pi])
        
        # For k=j:
        # (a[i,j]-a[j,j])*(b[pj,p[j]]-b[pi,p[j]]) + (a[j,i]-a[j,j])*(b[p[j],pj]-b[p[j],pi])
        # = (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj]) + (a[j,i]-a[j,j])*(b[pj,pj]-b[pj,pi])
        
        # For k != i,j:
        # (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],pj]-b[p[k],pi])
        
        # Let's compute the full sum and then adjust for k=i and k=j
        
        da_row = a[i, :] - a[j, :]
        da_col = a[:, i] - a[:, j]
        db_row = b[pj, :] - b[pi, :]
        db_col = b[:, pj] - b[:, pi]
        
        # Full sum
        total = np.dot(da_row, db_row[p]) + np.dot(da_col, db_col[p])
        
        # Subtract the k=i and k=j terms and add back the correct ones
        # When k=i in the first sum: da_row[i] * db_row[p[i]] = (a[i,i]-a[j,i]) * (b[pj,pi]-b[pi,pi])
        # When k=j in the first sum: da_row[j] * db_row[p[j]] = (a[i,j]-a[j,j]) * (b[pj,pj]-b[pi,pj])
        
        # When k=i in the second sum: da_col[i] * db_col[p[i]] = (a[i,i]-a[i,j]) * (b[pi,pj]-b[pi,pi])
        # When k=j in the second sum: da_col[j] * db_col[p[j]] = (a[j,i]-a[j,j]) * (b[pj,pj]-b[pj,pi])
        
        # But in the new configuration, when we swap i and j:
        # The term for k=i becomes: a[i,i]*b[pj,pj] + a[i,j]*b[pj,pi] (wait, this is getting confusing)
        
        # Let's just use the direct computation for now and optimize later
        # For efficiency, let's precompute some things
        
        # Actually, let's use a simpler correct formula:
        # The change in cost is:
        # sum_k (a[i,k]*b[pj,p[k]] + a[j,k]*b[pi,p[k]] - a[i,k]*b[pi,p[k]] - a[j,k]*b[pj,p[k]])
        # + sum_k (a[k,i]*b[p[k],pj] + a[k,j]*b[p[k],pi] - a[k,i]*b[p[k],pi] - a[k,j]*b[p[k],pj])
        
        # = sum_k (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]])
        # + sum_k (a[k,i]-a[k,j])*(b[p[k],pj]-b[p[k],pi])
        
        # This is exactly what I had. The issue is that when k=i or k=j, 
        # p[k] changes, so we need to be careful.
        
        # Let me just compute it directly:
        p_new = p.copy()
        p_new[i], p_new[j] = p_new[j], p_new[i]
        
        # Compute the difference
        # Old cost contribution involving i or j:
        old = 0
        new = 0
        
        for k in range(n):
            if k == i or k == j:
                continue
            pk = p[k]
            pk_new = p_new[k]  # same as pk since k != i,j
            
            # Terms where i or j is first index
            old += a[i, k] * b[pi, pk] + a[j, k] * b[pj, pk]
            new += a[i, k] * b[pj, pk] + a[j, k] * b[pi, pk]
            
            # Terms where i or j is second index
            old += a[k, i] * b[pk, pi] + a[k, j] * b[pk, pj]
            new += a[k, i] * b[pk, pj] + a[k, j] * b[pk, pi]
        
        # Terms where both i and j are involved
        old += a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi]
        new += a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj]
        
        return new - old

    def local_search(p, max_passes=100, time_limit=None):
        """2-opt local search with best-improvement strategy."""
        if time_limit is None:
            time_limit = deadline
        p = p.copy()
        cost = compute_cost(p)
        n_local = len(p)

        for _pass in range(max_passes):
            if time.time() > time_limit:
                break

            best_delta = 0.0
            best_i, best_j = -1, -1

            for i in range(n_local):
                for j in range(i + 1, n_local):
                    delta = swap_delta_fast(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
                if time.time() > time_limit:
                    return p, cost

            if best_i == -1:
                break

            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta

        return p, cost

    def perturb(p, num_swaps):
        """Apply random multi-swap perturbation."""
        p = p.copy()
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        return p

    # Initial solution via spectral bisection
    best_p, best_cost = spectral_init()

    # First local search to convergence
    best_p, best_cost = local_search(best_p, max_passes=50)

    # Iterated local search with adaptive perturbation
    stall_count = 0
    stall_threshold = 5

    while time.time() < deadline - 5.0:
        # Adaptive perturbation
        if stall_count < stall_threshold:
            num_swaps = 2
        elif stall_count < 2 * stall_threshold:
            num_swaps = 4
        else:
            num_swaps = 6

        p_pert = perturb(best_p, num_swaps)
        p_new, cost_new = local_search(p_pert, max_passes=50)

        if cost_new < best_cost:
            best_p = p_new
            best_cost = cost_new
            stall_count = 0
        else:
            stall_count += 1
            # Occasionally accept worse solutions to escape local optima
            if stall_count > 3 * stall_threshold and np.random.random() < 0.1:
                best_p = p_new
                best_cost = cost_new
                stall_count = 0

        # If heavily stalled, try fresh initialization
        if stall_count > 15:
            p_fresh, cost_fresh = spectral_init()
            p_new, cost_new = local_search(p_fresh, max_passes=50)
            if cost_new < best_cost:
                best_p = p_new
                best_cost = cost_new
            stall_count = 0

    # Final polish
    if time.time() < deadline - 2.0:
        best_p, best_cost = local_search(best_p, max_passes=20)

    return best_p.tolist()

# EVOLVE-BLOCK-END
