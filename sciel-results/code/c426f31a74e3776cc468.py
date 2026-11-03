"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # 1. Spectral Initialization
    # We want to align high-flow facilities with high-distance locations.
    # The QAP cost is sum a[i,j] b[p[i], p[j]].
    # If we decompose a and b into eigenbases, we can try to match principal components.
    # For symmetric positive semi-definite matrices, the largest eigenvectors correspond to the "directions" of highest activity.
    # Let a = V_a Lambda_a V_a^T and b = V_b Lambda_b V_b^T.
    # The optimal assignment often correlates the principal eigenvectors.
    # Specifically, if we sort facilities by their projection onto the largest eigenvector of A,
    # and locations by their projection onto the largest eigenvector of B, and match them,
    # we get a geometrically informed initial solution.
    
    try:
        # Compute eigenvalues and eigenvectors
        # We only need the top few (or just the top one) for a strong heuristic signal.
        # eigh returns eigenvalues in ascending order.
        eigvals_a, eigvecs_a = eigh(a)
        eigvals_b, eigvecs_b = eigh(b)
        
        # Take the eigenvector corresponding to the largest eigenvalue
        # (last index in ascending order)
        v_a = eigvecs_a[:, -1]
        v_b = eigvecs_b[:, -1]
        
        # Sort facilities by their score on v_a (descending)
        # Sort locations by their score on v_b (descending)
        # This aligns the "most active" facilities with the "most connected" locations.
        facility_scores = v_a
        location_scores = v_b
        
        heavy_facilities = np.argsort(-facility_scores, kind="stable")
        central_locations = np.argsort(-location_scores, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations
        
    except Exception:
        # Fallback to rank matching if spectral fails
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        heavy_facilities = np.argsort(-flow_sums, kind="stable")
        central_locations = np.argsort(dist_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations

    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Helper to compute delta for swapping facilities i and j
    # p[i] = u, p[j] = v
    # Delta = 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    # Note: We assume a and b are symmetric. If not, the formula is more complex.
    # QAPLIB instances typically have symmetric matrices.
    def get_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        # Vectorized computation for speed
        # Create a mask for k != i, j
        # We want sum over k of (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        # Then subtract the k=i and k=j terms if they were included, or just exclude them.
        
        # Extract rows
        ai = a[i]
        aj = a[j]
        bv = b[v]
        bu = b[u]
        
        # p is the permutation array
        # b[v, p[k]] is bv[p[k]]
        # b[u, p[k]] is bu[p[k]]
        
        diff_a = ai - aj
        # b_diff_k = bv[p[k]] - bu[p[k]]
        
        # We need to exclude k=i and k=j
        # Let's compute the full sum and subtract the i and j terms
        # Term for k: diff_a[k] * (bv[p[k]] - bu[p[k]])
        
        # Full sum
        sum_all = np.sum(diff_a * (bv[p] - bu[p]))
        
        # Subtract k=i
        # k=i: diff_a[i] * (bv[p[i]] - bu[p[i]]) = (a[i,i]-a[j,i]) * (bv[u] - bu[u])
        term_i = (ai[i] - aj[i]) * (bv[u] - bu[u])
        
        # Subtract k=j
        # k=j: diff_a[j] * (bv[p[j]] - bu[p[j]]) = (a[i,j]-a[j,j]) * (bv[v] - bu[v])
        term_j = (ai[j] - aj[j]) * (bv[v] - bu[v])
        
        delta = 2 * (sum_all - term_i - term_j)
        return int(delta)

    def local_search(p, current_cost):
        improved = True
        # Limit iterations to prevent infinite loops if oscillating (though strict improvement prevents this)
        max_iters = 1000
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            # Iterate over all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    delta = get_delta(i, j, p)
                    if delta < 0:
                        # Accept
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        return p, current_cost

    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    # ILS Loop
    while time.time() - start_time < timeout:
        # Local Search
        p, current_cost = local_search(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Check time
        if time.time() - start_time > timeout:
            break
            
        # Perturbation
        # Perform a few random swaps to escape local optimum
        # Scale perturbation size with n
        num_swaps = max(2, n // 10)
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            # Calculate delta to update cost correctly even if we don't accept
            # Actually, for perturbation, we just swap and update cost
            delta = get_delta(i, j, p)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
