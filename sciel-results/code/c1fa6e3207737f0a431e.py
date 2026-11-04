"""QAP solver: Spectral Init + VND with Block Reversal moves."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    
    start_time = time.time()
    deadline = 118.0  # Safety margin before 120s/150s kill

    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_swap_delta(p, i, j):
        """
        Calculate delta cost for swapping facilities i and j in permutation p.
        """
        u = p[i]
        v = p[j]
        
        # Direct pair term
        delta = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        # Diagonal terms
        delta += a[i, i] * (b[v, v] - b[u, u])
        delta += a[j, j] * (b[u, u] - b[v, v])
        
        # Cross terms with all other k
        diff_a_row = a[i] - a[j]
        diff_a_col = a[:, i] - a[:, j]
        
        diff_b_row = b[v, p] - b[u, p]
        diff_b_col = b[p, v] - b[p, u]
        
        sum_T1 = np.dot(diff_a_row, diff_b_row)
        sum_T2 = np.dot(diff_a_col, diff_b_col)
        
        # Subtract k=i terms
        sum_T1 -= (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        sum_T2 -= (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
        
        # Subtract k=j terms
        sum_T1 -= (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        sum_T2 -= (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
        
        delta += sum_T1 + sum_T2
        return int(delta)

    def get_block_reversal_delta(p, i, j):
        """
        Calculate delta cost for reversing the segment p[i..j].
        This is a powerful move that changes the relative order of many facilities.
        """
        if i == j:
            return 0
        if i > j:
            i, j = j, i
            
        # The move reverses the segment p[i], p[i+1], ..., p[j].
        # Let L = j - i + 1 (length of segment)
        # The new permutation p' is identical to p except in [i, j].
        # p'[i+k] = p[j-k] for k=0..L-1.
        
        # Cost = sum_{u,v} a[u,v] b[p[u], p[v]]
        # Delta = Cost(p') - Cost(p)
        
        # Terms involving indices outside [i, j] remain unchanged in value?
        # No, the values of b[p[u], p[v]] change if u or v is in [i, j].
        
        # Let S be the set of indices in [i, j].
        # Let R be the rest.
        
        # Delta = (Sum_{u,v in S} a[u,v] (b[p'[u], p'[v]] - b[p[u], p[v]]))
        #       + (Sum_{u in S, v in R} a[u,v] (b[p'[u], p[v]] - b[p[u], p[v]]))
        #       + (Sum_{u in R, v in S} a[u,v] (b[p[u], p'[v]] - b[p[u], p[v]]))
        #       + (Sum_{u in S, v in S, u!=v} ... already covered if we sum all u,v in S)
        
        # Actually, it is easier to compute the change in terms of the facilities involved.
        # Let F_S be the set of facilities in the segment.
        # Let F_R be the rest.
        
        # The positions of facilities in F_R do not change.
        # The positions of facilities in F_S are permuted by reversal.
        
        # Let's compute the cost contribution of the segment to itself and to the rest.
        
        # 1. Internal cost of the segment
        # Old: sum_{u,v in S} a[u,v] b[p[u], p[v]]
        # New: sum_{u,v in S} a[u,v] b[p'[u], p'[v]]
        # Note that p'[u] is a permutation of the same set of locations {p[k] for k in S}.
        
        # 2. External cost: interaction between S and R
        # Old: sum_{u in S, v in R} a[u,v] b[p[u], p[v]] + sum_{u in R, v in S} a[u,v] b[p[u], p[v]]
        # New: sum_{u in S, v in R} a[u,v] b[p'[u], p[v]] + sum_{u in R, v in S} a[u,v] b[p[u], p'[v]]
        
        # This calculation is O(L * n) if done naively for each candidate.
        # For L up to ~10-20, this is manageable.
        
        L = j - i + 1
        if L > 20: # Limit block size to keep delta computation fast
             return 0 # Skip large blocks in delta calc, maybe handle separately or ignore
        
        # Extract relevant sub-matrices/vectors
        # Indices in segment
        seg_idx = np.arange(i, j+1)
        rest_idx = np.setdiff1d(np.arange(n), seg_idx)
        
        # Facilities in segment
        fac_seg = p[seg_idx]
        # Facilities in rest
        fac_rest = p[rest_idx]
        
        # Old internal cost
        A_seg = a[np.ix_(seg_idx, seg_idx)]
        B_seg_old = b[np.ix_(fac_seg, fac_seg)]
        old_internal = np.sum(A_seg * B_seg_old)
        
        # New internal cost
        # Reversal of fac_seg
        fac_seg_new = fac_seg[::-1]
        B_seg_new = b[np.ix_(fac_seg_new, fac_seg_new)]
        new_internal = np.sum(A_seg * B_seg_new)
        
        # Old external cost
        # sum_{u in S, v in R} a[u,v] b[p[u], p[v]]
        A_ext_old_1 = a[np.ix_(seg_idx, rest_idx)]
        B_ext_old_1 = b[np.ix_(fac_seg, fac_rest)]
        old_ext_1 = np.sum(A_ext_old_1 * B_ext_old_1)
        
        # sum_{u in R, v in S} a[u,v] b[p[u], p[v]]
        A_ext_old_2 = a[np.ix_(rest_idx, seg_idx)]
        B_ext_old_2 = b[np.ix_(fac_rest, fac_seg)]
        old_ext_2 = np.sum(A_ext_old_2 * B_ext_old_2)
        
        # New external cost
        # sum_{u in S, v in R} a[u,v] b[p'[u], p[v]]
        # p'[u] for u in seg_idx corresponds to fac_seg_new
        A_ext_new_1 = a[np.ix_(seg_idx, rest_idx)] # Same A
        B_ext_new_1 = b[np.ix_(fac_seg_new, fac_rest)]
        new_ext_1 = np.sum(A_ext_new_1 * B_ext_new_1)
        
        # sum_{u in R, v in S} a[u,v] b[p[u], p'[v]]
        A_ext_new_2 = a[np.ix_(rest_idx, seg_idx)] # Same A
        B_ext_new_2 = b[np.ix_(fac_rest, fac_seg_new)]
        new_ext_2 = np.sum(A_ext_new_2 * B_ext_new_2)
        
        delta = (new_internal - old_internal) + (new_ext_1 - old_ext_1) + (new_ext_2 - old_ext_2)
        return int(delta)

    def local_search_swap(p, current_cost):
        """Best-improvement swap local search."""
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > deadline - 2.0:
                break
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Sample a subset of swaps if n is large? No, n=100, 4950 swaps.
            # Vectorized delta calc is fast.
            for i in range(n):
                if time.time() - start_time > deadline - 2.0:
                    break
                for j in range(i + 1, n):
                    delta = get_swap_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
        return p, current_cost

    def local_search_block(p, current_cost):
        """Best-improvement block reversal local search."""
        improved = True
        max_block_len = 10 # Limit to keep it fast
        while improved:
            improved = False
            if time.time() - start_time > deadline - 2.0:
                break
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Scan for best block reversal
            # Number of blocks is ~ n * max_block_len = 1000
            for i in range(n):
                if time.time() - start_time > deadline - 2.0:
                    break
                for L in range(2, max_block_len + 1):
                    j = i + L - 1
                    if j >= n:
                        break
                    delta = get_block_reversal_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i != -1:
                # Apply reversal
                p[best_i:best_j+1] = p[best_i:best_j+1][::-1]
                current_cost += best_delta
                improved = True
        return p, current_cost

    def vnd(p, current_cost):
        """Variable Neighborhood Descent."""
        # Neighborhood 1: Swaps
        p, current_cost = local_search_swap(p, current_cost)
        # Neighborhood 2: Block Reversals
        p, current_cost = local_search_block(p, current_cost)
        # Check if we need another round? Usually one pass of each is enough for VND step.
        # If we want to be thorough, we can loop until no improvement in either.
        # But for speed in ILS, single pass per neighborhood is standard.
        return p, current_cost

    def spectral_init():
        """
        Initialize using spectral embedding.
        We want to find a permutation p that minimizes sum a[i,j] b[p[i], p[j]].
        This is similar to finding a low-rank factorization.
        
        A common heuristic:
        1. Compute eigenvectors of a and b.
        2. Sort facilities by eigenvector components and match to locations sorted by eigenvector components.
        
        Alternatively, use the "Surrogate" method:
        C[i, j] = - sum_k a[i, k] b_row_sums[k]? No.
        
        Let's use the standard spectral approach for QAP:
        Let U_a, S_a, V_a = SVD(a)
        Let U_b, S_b, V_b = SVD(b)
        
        Actually, a simpler and effective spectral init for QAP:
        Project the problem onto the first few principal components.
        
        Let's try:
        1. Center a and b?
        2. Compute leading eigenvectors.
        3. Sort facilities by projection on first eigenvector of a.
        4. Sort locations by projection on first eigenvector of b.
        5. Match them.
        
        This works well if the matrices have strong low-rank structure.
        """
        # Compute eigenvectors of a
        # a is symmetric? Not necessarily, but often close.
        # Use SVD to be safe.
        try:
            U_a, s_a, Vt_a = np.linalg.svd(a, full_matrices=False)
            U_b, s_b, Vt_b = np.linalg.svd(b, full_matrices=False)
            
            # Use the first right singular vector (Vt_a[0]) to rank facilities
            # and first right singular vector (Vt_b[0]) to rank locations
            # Actually, for QAP, we want to align the "shapes".
            # Let's use the first left singular vectors as coordinates.
            coords_a = U_a[:, 0]
            coords_b = U_b[:, 0]
            
            # Sort facilities by coords_a
            order_a = np.argsort(coords_a)
            # Sort locations by coords_b
            order_b = np.argsort(coords_b)
            
            # Match i-th smallest facility coord to i-th smallest location coord
            p = np.empty(n, dtype=np.int64)
            for k in range(n):
                fac = order_a[k]
                loc = order_b[k]
                p[fac] = loc
                
            return p
        except:
            # Fallback to random or Hungarian if SVD fails
            return np.random.permutation(n)

    def hungarian_init():
        """
        Initialize using Hungarian algorithm on surrogate cost.
        C[i, j] = -a_row_sums[i] * b_row_sums[j]
        """
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        
        C = -np.outer(a_row_sums, b_row_sums)
        
        row_ind, col_ind = linear_sum_assignment(C)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    # 1. Initialization
    # Try both Spectral and Hungarian, pick the better one after a quick local search?
    # Or just use Spectral as it's structurally different.
    # Let's run a few quick local searches on both to see which is better start.
    
    p_spec = spectral_init()
    p_hung = hungarian_init()
    
    # Quick polish for both
    p_spec, cost_spec = local_search_swap(p_spec, calc_cost(p_spec))
    p_hung, cost_hung = local_search_swap(p_hung, calc_cost(p_hung))
    
    if cost_spec < cost_hung:
        p = p_spec
        current_cost = cost_spec
    else:
        p = p_hung
        current_cost = cost_hung
        
    best_p = p.copy()
    best_cost = current_cost

    # 2. Initial VND
    p, current_cost = vnd(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()

    # 3. ILS Loop with VND
    no_improve_count = 0
    rng = np.random.default_rng()
    
    while time.time() - start_time < deadline - 2.0:
        # Adaptive perturbation
        if no_improve_count < 5:
            num_swaps = 2
        elif no_improve_count < 10:
            num_swaps = 5
        elif no_improve_count < 20:
            num_swaps = 10
        else:
            num_swaps = 20
        
        # Perturb
        p_perturbed = best_p.copy()
        for _ in range(num_swaps):
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j: continue
            p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
            
        # VND Local Search
        p_perturbed, cost_perturbed = vnd(p_perturbed, calc_cost(p_perturbed))
        
        if cost_perturbed < best_cost:
            best_cost = cost_perturbed
            best_p = p_perturbed.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

    return best_p.tolist()
# EVOLVE-BLOCK-END
