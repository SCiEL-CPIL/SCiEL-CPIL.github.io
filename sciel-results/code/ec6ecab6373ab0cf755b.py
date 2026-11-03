"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _generate_supports(max_size=4):
    """Generate all supports of size 2 to max_size."""
    supports = []
    sizes = []
    for k in range(2, max_size + 1):
        for comb in combinations(range(D), k):
            supports.append(comb)
            sizes.append(k)
    return supports, np.array(sizes)

def _construct_vectors_from_supports(supports, sizes, selected_mask):
    """Construct all sign variants for selected supports."""
    vecs = []
    for i, mask in enumerate(selected_mask):
        if not mask:
            continue
        s = supports[i]
        k = sizes[i]
        # Generate all 2^k sign patterns
        for pattern in range(2**k):
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(s):
                if (pattern >> j) & 1:
                    v[idx] = 1
                else:
                    v[idx] = -1
            vecs.append(v)
    
    # Add size-1 supports for free indices
    # Determine which indices are covered by selected supports
    covered = np.zeros(D, dtype=bool)
    for i, mask in enumerate(selected_mask):
        if mask:
            for idx in supports[i]:
                covered[idx] = True
    
    for i in range(D):
        if not covered[i]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = 1
            vecs.append(v)
            
    if not vecs:
        return np.array([[1]] + [0]*(D-1), dtype=np.int64)
        
    return np.array(vecs, dtype=np.int64)

def _solve_sa(supports, sizes, max_norm_sq, time_limit=10):
    """
    Solve support selection using Simulated Annealing.
    Objective: Maximize sum(2^size * x) + count_free_indices
    Constraint: For all selected i, j, |S_i n S_j| <= max_int[i][j]
    max_int depends on max_norm_sq.
    """
    n_sup = len(supports)
    if n_sup == 0:
        return np.zeros(0, dtype=bool)
    
    # Precompute intersection sizes
    # Supports are tuples of indices
    # Create a matrix of intersection sizes
    # n_sup is at most C(11,2)+C(11,3)+C(11,4) = 55 + 165 + 330 = 550
    # 550x550 matrix is small.
    
    # Convert supports to sets for faster intersection? Or just use sets directly.
    sup_sets = [set(s) for s in supports]
    
    inter_matrix = np.zeros((n_sup, n_sup), dtype=np.int8)
    for i in range(n_sup):
        for j in range(i, n_sup):
            k = len(sup_sets[i].intersection(sup_sets[j]))
            inter_matrix[i][j] = k
            inter_matrix[j][i] = k
            
    # Determine max allowed intersection for each pair based on max_norm_sq
    # If max_norm_sq = 2 (only size 2 allowed effectively, or we restrict pool)
    # Actually, we run SA on the full pool of size 2,3,4 but the constraint
    # uses the GLOBAL max_norm_sq.
    
    # Calculate max allowed intersection C(i, j)
    # dist^2 = w_i^2 + w_j^2 - 2 * dot
    # We assume worst case dot = intersection size.
    # Need w_i^2 + w_j^2 - 2 * inter >= max_norm_sq
    # 2 * inter <= w_i^2 + w_j^2 - max_norm_sq
    # inter <= (w_i^2 + w_j^2 - max_norm_sq) / 2
    
    w_sq = sizes * sizes
    max_int_matrix = np.zeros((n_sup, n_sup), dtype=np.int8)
    for i in range(n_sup):
        for j in range(n_sup):
            if i == j:
                # Same support: we take all signs.
                # Min dist^2 for same support size w is 4 (for w>=2).
                # Need 4 >= max_norm_sq.
                # If max_norm_sq > 4, we cannot take all signs.
                # But our pool is size 2,3,4. Max w^2=4.
                # So for max_norm_sq <= 4, same support is always ok for w>=2?
                # Wait, if w=2, min dist^2=4. If max_norm_sq=4, 4>=4 OK.
                # If w=3, min dist^2=4. If max_norm_sq=4, 4>=4 OK.
                # If w=4, min dist^2=4. If max_norm_sq=4, 4>=4 OK.
                # So same support is always compatible if max_norm_sq <= 4.
                max_int_matrix[i][j] = 100 # Effectively no constraint
            else:
                val = (w_sq[i] + w_sq[j] - max_norm_sq) / 2.0
                # inter must be <= val
                # Also inter cannot exceed min(w_i, w_j)
                max_allowed = int(np.floor(val))
                max_allowed = min(max_allowed, min(sizes[i], sizes[j]))
                max_int_matrix[i][j] = max_allowed
                
    # Initial state: Random or Greedy?
    # Let's start with all False and add greedily, then SA.
    x = np.zeros(n_sup, dtype=bool)
    
    # Greedy init: Sort by size descending, add if compatible
    order = np.argsort(-sizes)
    for i in order:
        if time.time() > time_limit + 2: break
        # Check compatibility with all selected
        compatible = True
        for j in np.where(x)[0]:
            if inter_matrix[i][j] > max_int_matrix[i][j]:
                compatible = False
                break
        if compatible:
            x[i] = True
            
    # SA Loop
    start_time = time.time()
    current_score = 0
    
    def calc_score(x):
        # Sum 2^size
        score = np.sum(2**sizes[x])
        # Free indices
        covered = np.zeros(D, dtype=bool)
        for i in np.where(x)[0]:
            for idx in supports[i]:
                covered[idx] = True
        score += np.sum(~covered)
        return score
        
    current_score = calc_score(x)
    best_x = x.copy()
    best_score = current_score
    
    T = 10.0
    cooling = 0.99
    
    while time.time() < start_time + time_limit:
        # Pick a random support to flip
        i = np.random.randint(n_sup)
        
        # Calculate delta score
        # Delta from flipping x[i]
        # 1. Change in 2^size term
        delta_weight = 2**sizes[i] if not x[i] else -2**sizes[i]
        
        # 2. Change in free indices
        # If adding i: free indices decrease by (number of indices in S_i not already covered)
        # If removing i: free indices increase by (number of indices in S_i not covered by OTHERS)
        
        covered_by_others = np.zeros(D, dtype=bool)
        # This is expensive to compute every step.
        # Optimization: Maintain a count array of how many selected supports cover each index.
        
        # Let's refactor to use counts for speed.
        
        break # Re-implement below with counts
        
    # Re-implement SA with maintained counts for efficiency
    x = np.zeros(n_sup, dtype=bool)
    count_cover = np.zeros(D, dtype=np.int32)
    
    def get_score(x, count_cover):
        return np.sum(2**sizes[x]) + np.sum(count_cover == 0)
        
    current_score = get_score(x, count_cover)
    best_x = x.copy()
    best_score = current_score
    
    T = 5.0
    cooling = 0.995
    
    # Greedy init again
    order = np.argsort(-sizes)
    for i in order:
        # Check compatibility
        ok = True
        for j in np.where(x)[0]:
            if inter_matrix[i][j] > max_int_matrix[i][j]:
                ok = False
                break
        if ok:
            x[i] = True
            for idx in supports[i]:
                count_cover[idx] += 1
            current_score = get_score(x, count_cover)
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
                
    while time.time() < start_time + time_limit:
        i = np.random.randint(n_sup)
        
        if x[i]:
            # Try to remove
            # Check if removal is beneficial? SA always tries.
            # Delta score: -2^size[i] + (indices in S_i that become free)
            # Indices in S_i that become free are those where count_cover[idx] == 1
            # And we are removing i.
            
            # Check compatibility of REMOVAL? No, removal always restores compatibility.
            # So we just calculate delta.
            
            delta_weight = -2**sizes[i]
            delta_free = 0
            for idx in supports[i]:
                if count_cover[idx] == 1:
                    delta_free += 1
            
            delta = delta_weight + delta_free
            
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                # Accept
                x[i] = False
                for idx in supports[i]:
                    count_cover[idx] -= 1
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_x = x.copy()
        else:
            # Try to add
            # Check compatibility with all selected
            ok = True
            selected_indices = np.where(x)[0]
            if len(selected_indices) > 0:
                # Vectorized check?
                # inter_matrix[i, selected_indices] vs max_int_matrix[i, selected_indices]
                if np.any(inter_matrix[i, selected_indices] > max_int_matrix[i, selected_indices]):
                    ok = False
            
            if not ok:
                T *= cooling
                continue
                
            # Calculate delta
            delta_weight = 2**sizes[i]
            delta_free = 0
            for idx in supports[i]:
                if count_cover[idx] == 0:
                    delta_free -= 1 # Becomes covered
            
            delta = delta_weight + delta_free
            
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                # Accept
                x[i] = True
                for idx in supports[i]:
                    count_cover[idx] += 1
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_x = x.copy()
            
        T *= cooling
        
    return best_x

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    supports, sizes = _generate_supports(4)
    
    best_result = None
    best_k = 0
    
    np.random.seed(42)
    
    # Try different max_norm_sq values
    for max_norm_sq in [2, 3, 4]:
        if time.time() > deadline - 10:
            break
            
        # Allocate time for SA
        sa_time = 15
        
        selected_mask = _solve_sa(supports, sizes, max_norm_sq, time_limit=sa_time)
        
        if np.sum(selected_mask) == 0:
            continue
            
        vecs = _construct_vectors_from_supports(supports, sizes, selected_mask)
        
        # Verify validity
        # Min dist >= max norm
        if len(vecs) < 2:
            continue
            
        # Compute norms
        norms_sq = np.sum(vecs * vecs, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        # Compute min dist
        # For large k, this is O(k^2). k can be ~200. 40000 ops. Fast.
        min_dist_sq = np.inf
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                d = vecs[i] - vecs[j]
                dist_sq = np.dot(d, d)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    if min_dist_sq < max_norm_sq:
                        break
            if min_dist_sq < max_norm_sq:
                break
                
        if min_dist_sq >= max_norm_sq:
            k = len(vecs)
            if k > best_k:
                best_k = k
                best_result = vecs
        
    if best_result is None:
        # Fallback: just return some valid set
        # e.g. all size 2 supports
        supports_2, sizes_2 = _generate_supports(2)
        mask = np.ones(len(supports_2), dtype=bool)
        best_result = _construct_vectors_from_supports(supports_2, sizes_2, mask)
        
    return best_result

# EVOLVE-BLOCK-END
