import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate all norm^2=4 integer vectors in R^11
    # Type A: weight 4, entries in {-1, 0, 1} -> C(11,4)*2^4 = 330*16 = 5280... wait
    # Actually C(11,4) = 330, times 2^4 = 16, so 5280. That's too many.
    # Wait, the kissing number in D_11 is 165. Let me reconsider.
    # 
    # The D_n lattice has 2^(n-1) + 2n vectors of norm^2 = 2 (for the standard basis).
    # But here we want norm^2 = 4.
    #
    # Actually for the kissing problem in dimension 11, the optimal configuration
    # uses vectors from the D_11 lattice scaled. The 165 vectors are:
    # - 2*11 = 22 vectors of form (±2, 0, ..., 0) [weight 1, norm^2=4]
    # - 144 vectors of weight 4 with specific sign patterns
    #
    # Wait, let me just generate all candidates and find the max independent set.
    
    # Generate weight-4 vectors (entries ±1 in 4 positions)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    # Generate weight-1 vectors (entry ±2 in 1 position)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)  # 330 * 16 = 5280
    
    # Compute conflict matrix for v4 vectors: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # The weight-1 vectors have inner product 0 or ±2 with weight-4 vectors,
    # so they're always compatible. We'll add all 22 at the end.
    
    # Use a fast greedy + local search to find a large independent set
    # among the 5280 weight-4 vectors.
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
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
        
        return in_set
    
    def local_search_fast(in_set_init, max_time=10.0, min_T=0.05, cooling=0.9995):
        """Optimized SA with O(1) not_in_set updates."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 3.0
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        # Maintain not_in_set as a list with position tracking for O(1) removal
        not_in_set = list(np.where(~in_set)[0])
        pos_in_not_in_set = {}
        for idx, v in enumerate(not_in_set):
            pos_in_not_in_set[v] = idx
        
        def remove_from_not_in_set(v):
            if v in pos_in_not_in_set:
                idx = pos_in_not_in_set[v]
                last = not_in_set[-1]
                not_in_set[idx] = last
                pos_in_not_in_set[last] = idx
                not_in_set.pop()
                del pos_in_not_in_set[v]
        
        def add_to_not_in_set(v):
            if v not in pos_in_not_in_set:
                pos_in_not_in_set[v] = len(not_in_set)
                not_in_set.append(v)
        
        while time.time() < sa_deadline:
            if len(not_in_set) == 0:
                break
            
            # Sample a subset if not_in_set is large
            if len(not_in_set) > 300:
                sample_size = 300
                sample_indices = rng.choice(len(not_in_set), size=sample_size, replace=False)
                sample = np.array([not_in_set[i] for i in sample_indices])
            else:
                sample = np.array(not_in_set)
            
            counts = count_conflicts[sample]
            min_count = int(np.min(counts))
            
            if min_count == 0:
                zero_mask = counts == 0
                if np.any(zero_mask):
                    zero_cands = sample[zero_mask]
                    v = int(zero_cands[rng.integers(len(zero_cands))])
                else:
                    v = int(not_in_set[rng.integers(len(not_in_set))])
                
                in_set[v] = True
                current_size += 1
                count_conflicts[neighbors[v]] += 1
                remove_from_not_in_set(v)
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                continue
            
            # Prioritize 1-conflict nodes
            one_mask = counts == 1
            if np.any(one_mask) and rng.random() < 0.7:
                one_cands = sample[one_mask]
                v = int(one_cands[rng.integers(len(one_cands))])
                k = 1
                delta = 1 - k
                accept = rng.random() < min(1.0, 0.3 + T * 0.5)
            else:
                min_mask = counts == min_count
                min_cands = sample[min_mask]
                v = int(min_cands[rng.integers(len(min_cands))])
                k = int(count_conflicts[v])
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    if delta < -20 * T:
                        accept = False
                    else:
                        accept = rng.random() < np.exp(delta / T)
            
            if accept:
                to_remove = np.where(conflicts[v] & in_set)[0]
                
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                        add_to_not_in_set(int(u))
                
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                remove_from_not_in_set(v)
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
            
            if stuck_counter > 800:
                T = max(T, 2.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return local_best_in_set, local_best_size
    
    # Multiple restarts with greedy + SA
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
        
        sa_time = min(remaining_time * 0.45, 14.0)
        if sa_time < 2:
            sa_time = 2.0
        
        sa_result, sa_size = local_search_fast(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
    
    # Final polish: try to add any remaining compatible vectors
    def final_polish(in_set):
        """Try to greedily add any vector that has 0 conflicts with current set."""
        in_set = in_set.copy()
        count_conflicts = np.zeros(N4, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            zero_conf = np.where((count_conflicts == 0) & (~in_set))[0]
            if len(zero_conf) > 0:
                # Add all zero-conflict vectors at once
                for v in zero_conf:
                    in_set[v] = True
                    count_conflicts[neighbors[v]] += 1
                changed = True
        
        return in_set
    
    best_in_set = final_polish(best_in_set)
    best_size = int(best_in_set.sum())
    
    # Combine with weight-1 vectors
    final_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
