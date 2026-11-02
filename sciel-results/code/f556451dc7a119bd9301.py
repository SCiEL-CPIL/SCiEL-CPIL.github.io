"""Kissing number in dimension 11 - constructive approach using D_11 lattice structure."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # ============================================================
    # Strategy: Use the known D_11 lattice shortest vector structure
    # D_n lattice shortest vectors are weight-4 vectors with even
    # number of minus signs, plus the 2n weight-1 vectors (±2e_i)
    # ============================================================
    
    # Generate weight-1 vectors: ±2e_i, norm^2 = 4
    w1_vectors = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            w1_vectors.append(v)
    
    # Generate weight-4 vectors with even number of minus signs (D_11 structure)
    # A weight-4 vector has 4 nonzero entries, each ±1
    # Even parity means the number of -1 entries is even (0, 2, or 4)
    w4_vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            minus_count = sum(1 for s in signs if s == -1)
            if minus_count % 2 == 0:  # even number of minus signs
                v = [0] * D
                for pos, s in zip(positions, signs):
                    v[pos] = s
                w4_vectors.append(v)
    
    # Also generate weight-2 vectors: two entries ±1, rest 0, norm^2 = 2
    # These might help fill gaps
    w2_vectors = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            w2_vectors.append(v)
    
    # Build the candidate set: start with D_11 structure (w1 + even-parity w4)
    # Then consider adding w2 vectors if they don't conflict
    all_candidates = w1_vectors + w4_vectors + w2_vectors
    vectors_all = np.array(all_candidates, dtype=np.int64)
    
    N = len(vectors_all)
    
    # Compute norms squared
    norms_sq = np.sum(vectors_all ** 2, axis=1)
    
    # Compute dot product matrix
    dots = vectors_all @ vectors_all.T
    
    # Conflict condition: for vectors i, j with norms n_i, n_j
    # dist^2 = n_i^2 + n_j^2 - 2*dot >= max_norm^2 = 4
    # So: dot <= (n_i^2 + n_j^2 - 4) / 2
    threshold = (norms_sq[:, None] + norms_sq[None, :] - 4) / 2.0
    conflicts = dots > threshold
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # ============================================================
    # Phase 1: Constructive initialization using D_11 structure
    # Start with the even-parity w4 vectors + w1 vectors
    # ============================================================
    
    # Identify indices of w1 and even-parity w4 vectors
    n_w1 = len(w1_vectors)
    n_w4 = len(w4_vectors)
    
    # Initial set: all w1 + all even-parity w4
    init_indices = list(range(n_w1 + n_w4))
    
    # Check if this initial set is valid (no conflicts among selected)
    init_set = np.zeros(N, dtype=bool)
    init_set[init_indices] = True
    
    # Verify no conflicts within the initial set
    conflict_in_init = False
    for i in init_indices:
        if np.any(conflicts[i] & init_set):
            conflict_in_init = True
            break
    
    if not conflict_in_init:
        # Great! The D_11 structure alone gives a valid kissing configuration
        # Now try to add w2 vectors that don't conflict
        current_set = init_set.copy()
        current_size = int(current_set.sum())
        
        # Try adding w2 vectors one by one (greedy, pick those with fewest conflicts)
        w2_indices = list(range(n_w1 + n_w4, N))
        w2_sorted = sorted(w2_indices, key=lambda i: neighbor_counts[i])
        
        for idx in w2_sorted:
            if time.time() > deadline - 20:
                break
            if not current_set[idx]:
                # Check if we can add idx without removing others
                conflicts_with_set = np.where(conflicts[idx] & current_set)[0]
                if len(conflicts_with_set) == 0:
                    current_set[idx] = True
                    current_size += 1
        
        best_size = current_size
        best_in_set = current_set.copy()
    else:
        # D_11 structure has internal conflicts; fall back to search
        best_size = 0
        best_in_set = np.zeros(N, dtype=bool)
    
    # ============================================================
    # Phase 2: Local search to improve
    # ============================================================
    
    rng = np.random.default_rng(42)
    
    def local_polish(in_set_init, max_iters=20000):
        """Exact local search: try all single add/remove moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        # Precompute conflict counts
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # Try adding: find vertex with best gain (1 - conflicts)
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                gains = 1 - count_conflicts[not_in_set]
                best_gain_idx = np.argmax(gains)
                best_gain = gains[best_gain_idx]
                
                if best_gain > 0:
                    v = not_in_set[best_gain_idx]
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += best_gain
                    count_conflicts[neighbors[v]] += 1
                    improved = True
            
            # Try removing: find vertex whose removal allows adding more
            in_set_indices = np.where(in_set)[0]
            if len(in_set_indices) > 2:
                best_remove_gain = 0
                best_remove_v = -1
                for v in in_set_indices:
                    # If we remove v, which vertices can we add?
                    freed = np.where(conflicts[v] & ~in_set)[0]
                    # Count how many of those freed vertices have no other conflicts
                    addable = 0
                    for u in freed:
                        if count_conflicts[u] == 1:  # only conflicted with v
                            addable += 1
                    gain = addable - 1
                    if gain > best_remove_gain:
                        best_remove_gain = gain
                        best_remove_v = v
                
                if best_remove_v >= 0 and best_remove_gain > 0:
                    v = best_remove_v
                    in_set[v] = False
                    current_size -= 1
                    count_conflicts[neighbors[v]] -= 1
                    
                    # Add the freed vertices
                    freed = np.where(conflicts[v] & ~in_set)[0]
                    for u in freed:
                        if count_conflicts[u] == 0:
                            in_set[u] = True
                            current_size += 1
                            count_conflicts[neighbors[u]] += 1
                    improved = True
        
        return in_set, current_size
    
    def sa_search(in_set_init, max_time=15.0):
        """Simulated annealing with add/remove/swap moves."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        stuck = 0
        
        while time.time() < sa_deadline:
            # Choose move type
            move_type = rng.random()
            
            if move_type < 0.4:
                # Add move
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                counts = count_conflicts[not_in_set]
                min_c = np.min(counts)
                cands = not_in_set[counts <= min_c + 2]
                v = cands[rng.integers(len(cands))]
                k = count_conflicts[v]
                delta = 1 - k
            elif move_type < 0.7:
                # Remove move
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0:
                    continue
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
            else:
                # Swap move: remove one in-set, add one not-in-set
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                if len(in_set_indices) == 0 or len(not_in_set) == 0:
                    continue
                v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                counts = count_conflicts[not_in_set] - conflicts[not_in_set, v_remove].astype(int)
                min_c = np.min(counts)
                cands = not_in_set[counts <= min_c + 2]
                v_add = cands[rng.integers(len(cands))]
                k = max(0, count_conflicts[v_add] - (1 if conflicts[v_add, v_remove] else 0))
                delta = 1 - k
            
            if move_type < 0.7:
                accept = delta > 0 or (delta == 0 and rng.random() < 0.3) or (delta < 0 and rng.random() < np.exp(delta / T))
            else:
                accept = delta > 0 or (delta == 0 and rng.random() < 0.3) or (delta < 0 and rng.random() < np.exp(delta / T))
            
            if accept:
                if move_type >= 0.7:
                    # Swap
                    in_set[v_remove] = False
                    count_conflicts[neighbors[v_remove]] -= 1
                    to_remove_extra = np.where(conflicts[v_add] & in_set)[0]
                    if len(to_remove_extra) > 0:
                        in_set[to_remove_extra] = False
                        for u in to_remove_extra:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v_add] = True
                    current_size += delta
                    count_conflicts[neighbors[v_add]] += 1
                elif move_type < 0.4:
                    # Add
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                else:
                    # Remove
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck = 0
                else:
                    stuck += 1
            else:
                stuck += 1
            
            if stuck > 500:
                T = max(T, 2.0)
                stuck = 0
            T *= cooling
            if T < min_T:
                T = min_T
        
        return local_best_in_set, local_best_size
    
    # Use the constructive initialization as starting point
    if best_size > 0:
        # Polish the constructive solution
        polish_result, polish_size = local_polish(best_in_set, max_iters=15000)
        if polish_size > best_size:
            best_size = polish_size
            best_in_set = polish_result
        
        # SA from the polished solution
        remaining = deadline - 15 - time.time()
        if remaining > 5:
            sa_result, sa_size = sa_search(best_in_set, max_time=min(remaining * 0.6, 20.0))
            if sa_size > best_size:
                best_size = sa_size
                best_in_set = sa_result
            
            # Final polish
            remaining2 = deadline - 5 - time.time()
            if remaining2 > 3:
                final_result, final_size = local_polish(best_in_set, max_iters=10000)
                if final_size > best_size:
                    best_size = final_size
                    best_in_set = final_result
    else:
        # Fallback: multiple restarts with greedy + SA
        for restart in range(30):
            if time.time() > deadline - 10:
                break
            
            # Greedy min-degree initialization
            in_set = np.zeros(N, dtype=bool)
            alive = np.ones(N, dtype=bool)
            degrees = neighbor_counts.copy()
            
            while np.any(alive) and time.time() < deadline - 10:
                alive_indices = np.where(alive)[0]
                if len(alive_indices) == 0:
                    break
                min_deg = np.min(degrees[alive_indices])
                cands = alive_indices[degrees[alive_indices] == min_deg]
                chosen = cands[rng.integers(len(cands))]
                in_set[chosen] = True
                nbrs = neighbors[chosen]
                to_remove = np.unique(np.concatenate(([chosen], nbrs)))
                alive[to_remove] = False
                rem = np.where(alive)[0]
                if len(rem) > 0:
                    degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
            
            current_size = int(in_set.sum())
            if current_size > best_size:
                best_size = current_size
                best_in_set = in_set.copy()
            
            # SA
            remaining = deadline - 10 - time.time()
            if remaining > 5:
                sa_result, sa_size = sa_search(in_set, max_time=min(remaining * 0.5, 15.0))
                if sa_size > best_size:
                    best_size = sa_size
                    best_in_set = sa_result.copy()
            
            # Polish
            remaining2 = deadline - 5 - time.time()
            if remaining2 > 3:
                polish_result, polish_size = local_polish(best_in_set, max_iters=8000)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result
    
    # Final result
    final_indices = np.where(best_in_set)[0]
    result_vectors = vectors_all[final_indices]
    
    return result_vectors

# EVOLVE-BLOCK-END
