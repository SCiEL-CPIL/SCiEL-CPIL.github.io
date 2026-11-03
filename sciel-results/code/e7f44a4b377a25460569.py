"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

DEADLINE_FRACTION = 0.92  # safety margin


def _compute_cost(a, b, p):
    """Exact integer cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _compute_swap_delta(a, b, p, i, j):
    """
    Delta in cost if we swap facilities i and j (i.e., p[i] and p[j] are exchanged).
    We want to minimize cost, so a negative delta is an improvement.
    """
    pi = p[i]
    pj = p[j]

    # Indices not in S
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    others = np.where(mask)[0]
    
    # S x S terms
    delta = (a[i, i] * (b[pj, pj] - b[pi, pi]) +
             a[i, j] * (b[pj, pi] - b[pi, pj]) +
             a[j, i] * (b[pi, pj] - b[pj, pi]) +
             a[j, j] * (b[pi, pi] - b[pj, pj]))
    
    # S x others
    # i row, others cols: a[i, others] * (b[pj, p[others]] - b[pi, p[others]])
    delta += np.sum(a[i, others] * (b[pj, p[others]] - b[pi, p[others]]))
    # j row, others cols: a[j, others] * (b[pi, p[others]] - b[pj, p[others]])
    delta += np.sum(a[j, others] * (b[pi, p[others]] - b[pj, p[others]]))
    
    # others x S
    # others rows, i col: a[others, i] * (b[p[others], pi] - b[p[others], pj])
    delta += np.sum(a[others, i] * (b[p[others], pi] - b[p[others], pj]))
    # others rows, j col: a[others, j] * (b[p[others], pj] - b[p[others], pi])
    delta += np.sum(a[others, j] * (b[p[others], pj] - b[p[others], pi]))
    
    return int(delta)


def _local_search(a, b, p, deadline):
    """
    First-improvement swap-based local search.
    p is a numpy array (int64). Returns the improved p (numpy array) and the cost.
    """
    n = len(p)
    current_cost = _compute_cost(a, b, p)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize the order of pairs to escape symmetry? Or just fixed order.
        # Fixed order is deterministic but might get stuck in specific patterns.
        # Let's shuffle the list of indices to try random pairs.
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        for i_idx in range(n):
            if time.time() >= deadline:
                break
            i = indices[i_idx]
            for j_idx in range(i_idx + 1, n):
                if time.time() >= deadline:
                    break
                j = indices[j_idx]
                delta = _compute_swap_delta(a, b, p, i, j)
                if delta < 0:
                    # Swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    # After a swap, the set of improving swaps has changed.
                    # In first-improvement, we typically restart the scan or continue.
                    # Restarting is safer but slower. Let's restart for correctness of "first improvement".
                    break
            if improved:
                break
    
    return p, current_cost


def solve(a, b):
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION
    
    n = len(a)
    
    # 1. Generate initial solution using rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    best_p = p_rank.copy()
    best_cost = _compute_cost(a, b, p_rank)
    
    # 2. Try Linear Assignment Heuristic
    # This is a structural change: using the linear relaxation of the QAP.
    # Minimize sum a[i,j] x[i,j] y[i,j] by fixing one matrix and solving assignment on the other.
    # Specifically, we can approximate the interaction by projecting onto the dominant eigenvector or using a matrix product.
    # A common heuristic is to solve the assignment problem for the matrix M = A @ B.T (or similar).
    # Let's try M = A @ B.T. We want to minimize sum_{i,j} M[i,j] * x[i,j]? 
    # Actually, if we fix y (locations), the cost is sum_i (sum_j a[i,j] b[p[i], p[j]]).
    # This is not linear in x.
    # However, a known heuristic is to compute the "gravity" of each facility towards each location.
    # Let's try the standard linear assignment heuristic:
    # C = A @ B.T. Then solve min sum C[i, p[i]]? No, that ignores interactions between facilities.
    # But it captures the "best match" between facility rows and location columns.
    
    if time.time() < deadline - 5:
        # Compute interaction matrix
        # A common choice is A @ B.T
        # We want to minimize sum_{i,j} a[i,j] b[p[i], p[j]]
        # If we assume b is the cost matrix and we assign i to p[i],
        # the term is sum_i sum_j a[i,j] b[p[i], p[j]].
        # This is quadratic.
        # Linearization: Let's try to match the "profile" of A with the "profile" of B.
        # One simple heuristic: Solve LAP on - (A @ B.T) or similar.
        
        # Let's try the heuristic from Karmarkar and Karp or similar spectral/linear methods.
        # A robust one is:
        # 1. Compute S = A.sum(axis=1) and T = B.sum(axis=1)
        # 2. This is rank matching (already done).
        
        # Let's try the "linear assignment" heuristic based on the matrix product.
        # We want to assign facilities to locations.
        # Let M = A @ B.T.
        # If we assign facility i to location k, we incur cost related to M[i,k]?
        # Not directly.
        
        # Let's try a different structural approach:
        # Randomized Greedy Construction (GRASP-like)
        # Or, since the prompt mentions "spectral initialization", let's try that.
        
        # Spectral Initialization:
        # Use the top eigenvectors of A and B to align them.
        # 1. Compute eigenvectors of A (symmetric part) and B (symmetric part).
        # 2. Project facilities onto the subspace spanned by top eigenvectors of A.
        # 3. Project locations onto the subspace spanned by top eigenvectors of B.
        # 4. Solve the assignment problem on the projected coordinates.
        
        # Since A and B are not necessarily symmetric, we use (A+A.T)/2.
        A_sym = (a + a.T) / 2.0
        B_sym = (b + b.T) / 2.0
        
        # Compute top k eigenvectors (k=10 or so)
        k = 10
        if n > k:
            eigvals_A, eigvecs_A = np.linalg.eigh(A_sym)
            eigvals_B, eigvecs_B = np.linalg.eigh(B_sym)
            
            # Take top k eigenvectors (largest eigenvalues)
            # eigh returns in ascending order, so take the last k
            top_A = eigvecs_A[:, -k:]
            top_B = eigvecs_B[:, -k:]
            
            # Project facilities to k-dim space
            P_fac = top_A.T @ a  # Or just use the coordinates?
            # Actually, we want to map facility i to a point in R^k.
            # The coordinates are the projections onto the eigenvectors.
            # coords_fac[i] = [v1[i], v2[i], ..., vk[i]]
            coords_fac = top_A.T # shape (k, n) -> transpose to (n, k)
            coords_loc = top_B.T # shape (k, n) -> transpose to (n, k)
            
            # Now we have n facilities in R^k and n locations in R^k.
            # We want to assign facility i to location j to minimize some distance?
            # Or maximize similarity?
            # The QAP cost is sum a[i,j] b[p[i], p[j]].
            # If we use the spectral embedding, we can solve the assignment problem on the projected coordinates.
            # A common metric is the Euclidean distance squared: sum_i ||coords_fac[i] - coords_loc[p[i]]||^2.
            # This is equivalent to maximizing sum_i <coords_fac[i], coords_loc[p[i]]>.
            
            # Let's compute the cost matrix for the assignment problem.
            # Cost[i, j] = - <coords_fac[i], coords_loc[j]>
            # We want to minimize this cost (i.e., maximize the inner product).
            
            # coords_fac is (n, k), coords_loc is (n, k)
            # C = - coords_fac @ coords_loc.T
            C = - (coords_fac @ coords_loc.T)
            
            # Solve LAP
            row_ind, col_ind = linear_sum_assignment(C)
            p_spec = np.empty(n, dtype=np.int64)
            p_spec[row_ind] = col_ind
            
            cost_spec = _compute_cost(a, b, p_spec)
            if cost_spec < best_cost:
                best_p = p_spec.copy()
                best_cost = cost_spec
    
    # 3. Local Search from best initial solution
    # We have time for local search.
    # We should run local search on the best initial solution found so far.
    # To diversify, we could run local search on multiple initial solutions if time permits.
    # But for now, let's just refine the best one.
    
    # If we have more time, we can try more initializations or more aggressive search.
    # For now, refine best_p.
    
    p_current = best_p.copy()
    p_current, cost_current = _local_search(a, b, p_current, deadline)
    
    if cost_current < best_cost:
        best_p = p_current
        best_cost = cost_current
    
    # If we still have time, we can try another initialization and search.
    # Let's try a random initialization and search.
    if time.time() < deadline - 10:
        p_rand = np.random.permutation(n).astype(np.int64)
        p_rand, cost_rand = _local_search(a, b, p_rand, deadline)
        if cost_rand < best_cost:
            best_p = p_rand
            best_cost = cost_rand
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
