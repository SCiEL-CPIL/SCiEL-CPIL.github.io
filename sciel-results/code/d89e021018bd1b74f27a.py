"""Kissing number in dimension 11 - Lattice-structured search with MIS refinement."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Generate all integer vectors in Z^11 with norm squared = 4
    # Type A: exactly one coordinate is +/-2, rest 0. Count: 11 * 2 = 22
    # Type B: exactly four coordinates are +/-1, rest 0. Count: C(11,4) * 2^4 = 330 * 16 = 5280
    # Total: 5302 vectors
    
    # The D11 lattice (odd lattice) consists of vectors in Z^11 with even sum of coordinates.
    # For Type A: one coord is +/-2, sum is +/-2 (even). All 22 are in D11.
    # For Type B: four coords are +/-1. Sum is even iff the number of negative entries is even.
    #   # negatives can be 0, 2, or 4.
    #   Count: C(4,0)*1 + C(4,2)*1 + C(4,4)*1 = 1 + 6 + 1 = 8 sign patterns per position set.
    #   So 330 * 8 = 2640 Type B vectors are in D11.
    # Total D11 vectors with norm^2=4: 22 + 2640 = 2662.
    
    # However, the kissing configuration in dimension 11 is known to have size 506.
    # The full set of 5302 vectors (or 2662 D11 vectors) is too large; we need a subset.
    # The current approach treats this as a Maximum Independent Set (MIS) problem on the conflict graph.
    # Conflict if dot product > 2.
    
    # Let's generate all 5302 vectors first, then filter if needed, or just use all.
    # Using all 5302 gives more freedom for the MIS solver.
    
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    all_candidates = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_candidates)

    if N == 0:
        return np.array([], dtype=np.int64)

    # Compute conflict matrix
    # Conflict if dot > 2
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to list of lists for faster iteration
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def greedy_init():
        """Greedy min-degree initialization with randomization."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            
            # Add some randomness: pick from bottom 10% by degree
            deg_sorted = np.sort(deg[alive_indices])
            threshold = deg_sorted[min(len(alive_indices)-1, int(len(alive_indices)*0.1))]
            cands = alive_indices[deg[alive_indices] <= threshold]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                deg[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search_2swap(in_set, max_time=5.0):
        """Local search with 1-add, 1-1 swap, and 1-2 swap (remove 1, add 2)."""
        in_set = in_set.copy()
        current_size = in_set.sum()
        best_size = current_size
        best_in_set = in_set.copy()
        
        # conflict_count[i]: number of neighbors of i that are in in_set
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1
            
        ls_deadline = time.time() + max_time
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < ls_deadline:
            improved = False
            
            # 1. Try to add a vertex with 0 conflicts
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = conflict_count[not_in_set]
                zero_conf = not_in_set[counts == 0]
                if len(zero_conf) > 0:
                    v = zero_conf[rng.integers(len(zero_conf))]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[neighbors[v]] += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                    no_improve_count = 0
                    continue
            
            # 2. Try 1-1 Swap
            if not improved:
                cands_v = np.where((~in_set) & (conflict_count == 1))[0]
                if len(cands_v) > 0:
                    v = cands_v[rng.integers(len(cands_v))]
                    nbrs_v_in_set = np.where(conflicts[v] & in_set)[0]
                    if len(nbrs_v_in_set) == 1:
                        u = nbrs_v_in_set[0]
                        in_set[u] = False
                        in_set[v] = True
                        conflict_count[neighbors[u]] -= 1
                        conflict_count[neighbors[v]] += 1
                        no_improve_count = 0
                        improved = True
                        continue

            # 3. Try 1-2 Swap (Remove 1, Add 2) -> Size increases by 1
            if not improved:
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > 0:
                    # Try multiple random u candidates
                    for _ in range(min(5, len(in_set_indices))):
                        u = in_set_indices[rng.integers(len(in_set_indices))]
                        cands_v = np.where((~in_set) & (conflict_count == 1))[0]
                        if len(cands_v) >= 2:
                            idx = rng.choice(len(cands_v), size=2, replace=False)
                            v1 = cands_v[idx[0]]
                            v2 = cands_v[idx[1]]
                            
                            nbrs_v1 = np.where(conflicts[v1] & in_set)[0]
                            nbrs_v2 = np.where(conflicts[v2] & in_set)[0]
                            
                            if len(nbrs_v1) == 1 and nbrs_v1[0] == u and \
                               len(nbrs_v2) == 1 and nbrs_v2[0] == u and \
                               not conflicts[v1, v2]:
                               
                               in_set[u] = False
                               in_set[v1] = True
                               in_set[v2] = True
                               
                               conflict_count[neighbors[u]] -= 1
                               conflict_count[neighbors[v1]] += 1
                               conflict_count[neighbors[v2]] += 1
                               
                               current_size += 1
                               improved = True
                               no_improve_count = 0
                               if current_size > best_size:
                                   best_size = current_size
                                   best_in_set = in_set.copy()
                               break

            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturbation: remove 2 random vertices to escape local optimum
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) >= 2:
                        idx = rng.choice(len(in_set_indices), size=2, replace=False)
                        for u in in_set_indices[idx]:
                            in_set[u] = False
                            current_size -= 1
                            conflict_count[neighbors[u]] -= 1
                    elif len(in_set_indices) == 1:
                        u = in_set_indices[0]
                        in_set[u] = False
                        current_size -= 1
                        conflict_count[neighbors[u]] -= 1
                    no_improve_count = 0

        return best_in_set, best_size

    # Run multiple restarts
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    # Initial Greedy
    in_set = greedy_init()
    size = in_set.sum()
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()
        
    # Local Search on initial
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 2:
        ls_time = min(remaining_time * 0.6, 15.0)
        ls_result, ls_size = local_search_2swap(best_in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
            
    # Restarts - more aggressive with time allocation
    n_restarts = 0
    while time.time() < deadline - 5 and n_restarts < 30:
        n_restarts += 1
        in_set = greedy_init()
        size = in_set.sum()
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2:
            break
            
        # Allocate more time to local search, especially for promising starts
        if size >= best_size * 0.9:
            ls_time = min(remaining_time * 0.5, 8.0)
        else:
            ls_time = min(remaining_time * 0.3, 4.0)
            
        ls_result, ls_size = local_search_2swap(in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

    # Final polish: try to add any remaining vertices with 0 conflicts
    if time.time() < deadline - 2:
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            conflict_count[neighbors[i]] += 1
        
        not_in_set = np.where(~best_in_set)[0]
        if len(not_in_set) > 0:
            counts = conflict_count[not_in_set]
            zero_conf = not_in_set[counts == 0]
            for v in zero_conf:
                best_in_set[v] = True
                conflict_count[neighbors[v]] += 1
                # Update counts for other not-in-set vertices
                remaining = np.where(~best_in_set)[0]
                if len(remaining) > 0:
                    conflict_count[remaining] += conflicts[remaining, v].astype(np.int32)
    
    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]
    return result_vectors

# EVOLVE-BLOCK-END
