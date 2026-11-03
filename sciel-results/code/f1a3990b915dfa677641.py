"""Rewritten program for kissing number in dimension 11."""
import time
import itertools
import numpy as np

D = 11

# EVOLVE-BLOCK-START
def _generate_shell(m, D):
    """Generate all nonzero integer vectors in R^D with exactly m non-zero entries,
    each entry in {1, -1}. Returns a list of tuples (for speed) or we build np array."""
    # Choose m positions out of D
    positions = itertools.combinations(range(D), m)
    # For each position, choose sign (+1 or -1)
    # Total count: C(D, m) * 2^m
    # For m=2, D=11: 55 * 4 = 220
    # For m=3, D=11: 165 * 8 = 1320
    # For m=4, D=11: 330 * 16 = 5280
    # For m=5, D=11: 462 * 32 = 14784
    # For m=6, D=11: 462 * 64 = 29568
    # For m=7, D=11: 330 * 128 = 42240
    # For m=8, D=11: 165 * 256 = 42240
    # For m=9, D=11: 55 * 512 = 28160
    # For m=10, D=11: 11 * 1024 = 11264
    # For m=11, D=11: 1 * 2048 = 2048
    
    vectors = []
    for pos in positions:
        # Generate all 2^m sign combinations
        for signs in itertools.product([1, -1], repeat=m):
            v = [0] * D
            for p, s in zip(pos, signs):
                v[p] = s
            vectors.append(tuple(v))
    return vectors


def _to_array(vectors):
    """Convert list of tuples to numpy int64 array."""
    if not vectors:
        return np.empty((0, D), dtype=np.int64)
    return np.array(vectors, dtype=np.int64)


def _check_validity(arr):
    """Check if arr is a valid kissing configuration.
    Minimum pairwise distance >= maximum norm."""
    if len(arr) == 0:
        return True
    norms_sq = np.sum(arr ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Compute all pairwise squared distances
    # dist_sq(i,j) = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    # We need min_{i!=j} dist_sq >= max_norm_sq
    
    # Use matrix multiplication for inner products
    inner = arr @ arr.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * inner
    
    # Set diagonal to inf
    np.fill_diagonal(dist_sq, np.inf)
    
    if len(arr) > 1:
        min_dist_sq = np.min(dist_sq)
    else:
        min_dist_sq = np.inf
    
    return min_dist_sq >= max_norm_sq


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Strategy: Start with m=2 shell (220 vectors), then try to add from higher shells
    # using greedy selection with vectorized inner product updates.
    
    # Step 1: Generate base set from m=2 shell
    shell2 = _generate_shell(2, D)
    base_arr = _to_array(shell2)
    
    # Verify base set is valid
    # For m=2 vectors with entries in {1,-1}, norm^2 = 2
    # Inner product between two m=2 vectors: 
    #   If they share 0 positions: 0
    #   If they share 1 position: ±1
    #   If they share 2 positions: ±2
    # Distance squared = 2 + 2 - 2*ip = 4 - 2*ip
    # Min distance squared = 4 - 2*2 = 0 if ip=2 (same vector up to sign? No, different vectors)
    # Wait, if two vectors share both positions and have same signs, they're the same vector.
    # In our generation, each vector is unique. Two different m=2 vectors can share at most 2 positions.
    # If they share 2 positions, they must have the same signs to be distinct? No.
    # Example: (1,1,0,...) and (1,-1,0,...) share position 0 with same sign, position 1 with different signs.
    # Inner product = 1*1 + (-1)*1 = 0. Wait no: v1[0]=1, v2[0]=1, v1[1]=1, v2[1]=-1.
    # ip = 1*1 + 1*(-1) = 0.
    # If v1=(1,1,0,...) and v2=(-1,-1,0,...): ip = -1-1 = -2.
    # If v1=(1,1,0,...) and v2=(1,1,0,...): same vector, not in set.
    # So max |ip| for distinct m=2 vectors sharing 2 positions: 
    #   They must differ in at least one sign. If they differ in one sign: ip = 1-1=0 or -1+1=0.
    #   If they differ in both signs: ip = -1-1 = -2 or 1+1=2? No, if v1=(1,1) and v2=(-1,-1), ip=-2.
    #   v1=(1,-1) and v2=(-1,1): ip = -1-1 = -2.
    #   v1=(1,1) and v2=(1,-1): ip = 1-1 = 0.
    # So max ip = 0 for distinct vectors? Let me recheck.
    # v1=(1,1,0,...), v2=(1,1,0,...) is same. 
    # v1=(1,1,0,...), v2=(-1,1,0,...): ip = -1+1 = 0.
    # v1=(1,1,0,...), v2=(1,-1,0,...): ip = 1-1 = 0.
    # v1=(1,1,0,...), v2=(-1,-1,0,...): ip = -1-1 = -2.
    # So max ip = 0, min ip = -2.
    # dist_sq = 4 - 2*ip, so min dist_sq = 4 - 2*0 = 4, max dist_sq = 4 - 2*(-2) = 8.
    # max_norm_sq = 2.
    # So min_dist_sq = 4 >= 2. Valid!
    
    best_set = base_arr
    best_score = len(base_arr)
    
    # Step 2: Try to add vectors from higher shells greedily
    # We'll try shells m=3,4,5,... and add vectors that maintain validity.
    
    # For greedy addition, we need to track for each candidate, the maximum inner product
    # with any vector already in the set. A new vector v can be added iff:
    #   ||v||^2 + ||u||^2 - 2*v.u >= max_norm_sq for all u in set
    # Since all vectors in the set have norm^2 = 2 (for m=2) or higher, this is tricky.
    
    # Actually, let's reconsider. The condition is:
    #   min_{i!=j} ||v_i - v_j||^2 >= max_i ||v_i||^2
    # 
    # If we have a set S with max norm squared = M, and we want to add v with norm squared N,
    # then we need:
    #   For all u in S: ||v - u||^2 = N + ||u||^2 - 2*v.u >= M
    #   And also N <= M (otherwise max norm increases, making it harder)
    #   Actually if N > M, then new max is N, and we need ||v-u||^2 >= N for all u in S.
    
    # Let's try a different approach: generate a large candidate pool and use greedy selection.
    
    # Generate candidates from shells m=2 to m=5
    candidates = []
    for m in [2, 3, 4, 5]:
        if time.time() > deadline - 30:
            break
        shell = _generate_shell(m, D)
        candidates.extend(shell)
    
    if not candidates:
        return best_set
    
    cand_arr = _to_array(candidates)
    n_cand = len(cand_arr)
    
    # Greedy selection with vectorized updates
    # Start with empty set, add vectors one by one if valid
    # But this is O(n^2) which might be slow for 20000+ candidates.
    
    # Better: Start with the m=2 set (which we know is valid), then try to add from m=3,4,5.
    
    # Let's use the vectorized greedy approach mentioned in the knowledge.
    # Maintain: chosen_indices, and for each candidate, track max_ip with chosen set.
    
    # Actually, let me think about this more carefully.
    # The m=2 set has 220 vectors, all with norm^2 = 2.
    # min pairwise dist^2 = 4 (as computed above).
    # So the set is valid with margin: 4 >= 2.
    
    # If we add a vector v with norm^2 = N:
    # - If N <= 2: max_norm stays 2. Need ||v-u||^2 >= 2 for all u in set.
    #   ||v-u||^2 = N + 2 - 2*v.u >= 2 => N - 2*v.u >= 0 => v.u <= N/2
    #   For N=2 (m=1 vectors, but we don't have those), v.u <= 1.
    # - If N > 2: max_norm becomes N. Need ||v-u||^2 >= N for all u in set.
    #   N + 2 - 2*v.u >= N => 2 - 2*v.u >= 0 => v.u <= 1
    
    # So for adding to the m=2 set:
    # - m=3 vectors have norm^2 = 3. Need v.u <= 1 for all u in m=2 set.
    # - m=4 vectors have norm^2 = 4. Need v.u <= 1 for all u in m=2 set.
    # - m=5 vectors have norm^2 = 5. Need v.u <= 1 for all u in m=2 set.
    
    # So the condition is simply: max_{u in set} v.u <= 1
    
    # Let's compute this efficiently.
    
    # Build the current set (m=2 vectors)
    current_set = base_arr.copy()
    n_current = len(current_set)
    
    # For each candidate, compute max inner product with current set
    # We'll do this in batches to save memory
    
    # Actually, let's just compute all inner products at once if feasible
    # 20000 candidates x 220 current = 4.4M operations, fine
    
    # Let's separate candidates by shell to handle norm correctly
    # But the condition v.u <= 1 works for all N >= 2 as shown above.
    
    # Wait, I need to double-check. If N > 2, max_norm becomes N.
    # Condition: ||v-u||^2 >= N for all u in set.
    # ||v-u||^2 = N + ||u||^2 - 2*v.u = N + 2 - 2*v.u >= N
    # => 2 - 2*v.u >= 0 => v.u <= 1. Yes, correct.
    
    # So we just need max_{u in set} v.u <= 1.
    
    # Compute inner products
    ip_matrix = cand_arr @ current_set.T  # shape (n_cand, n_current)
    max_ip = np.max(ip_matrix, axis=1)  # shape (n_cand,)
    
    # Candidates with max_ip <= 1 can be added
    valid_mask = max_ip <= 1
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        # Try to add as many as possible. But adding one changes the set,
        # so we need to re-check.
        # For now, let's just add all valid ones at once and see if the result is valid.
        # This might not be valid because the new vectors might conflict with each other.
        
        # Let's be more careful: add them one by one, checking validity.
        # But this is slow. Let's try a batch approach first.
        
        new_vectors = cand_arr[valid_indices]
        combined = np.vstack([current_set, new_vectors])
        
        if _check_validity(combined):
            current_set = combined
            best_score = len(current_set)
        else:
            # Try adding them one by one
            for idx in valid_indices:
                if time.time() > deadline - 10:
                    break
                v = cand_arr[idx:idx+1]
                test_set = np.vstack([current_set, v])
                if _check_validity(test_set):
                    current_set = test_set
                    best_score = len(current_set)
    
    return current_set

# EVOLVE-BLOCK-END
