"""Kissing number in dimension 11 - D11 lattice constructive + SA fallback."""
import numpy as np
import itertools
import time

D = 11

def _construct_d11_lattice():
    """Construct D11 lattice shortest vectors: even coordinate sum, weight 2 or 4.
    
    D11 = {x in Z^11 : sum(x_i) is even}
    Shortest vectors have squared norm 4:
      - Type A: weight 2, entries in {±2}, sum even (always)
      - Type B: weight 4, entries in {±1}, sum even
    Total: C(11,2)*4 + C(11,4)*8 = 220 + 165*8 = 220 + 1320 = 1540
    Wait, let me recount. Actually for D_n, the shortest vectors are:
      - e_i - e_j for i != j (weight 2, entries ±1) : C(n,2) * 2
      - e_i + e_j for i != j (weight 2, entries ±1) : C(n,2) * 2  -- but sum = ±2, even, OK
      Actually no. D_n lattice vectors with norm^2 = 2 are: ±e_i ± e_j (i≠j), that's C(n,2)*4.
      Vectors with norm^2 = 4: weight 4 with ±1 entries and even sum, OR weight 2 with ±2 entries.
      
    Let me just generate all integer vectors with even coordinate sum and norm^2 = 4.
    """
    vectors = []
    
    # Type 1: weight 2, entries ±2 (sum is ±4 or 0, always even)
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
    
    # Type 2: weight 4, entries ±1, even sum
    # Even sum means even number of -1's: 0, 2, or 4 negative signs
    for positions in itertools.combinations(range(D), 4):
        for n_neg in [0, 2, 4]:
            for neg_positions in itertools.combinations(positions, n_neg):
                v = [0] * D
                for pos in positions:
                    if pos in neg_positions:
                        v[pos] = -1
                    else:
                        v[pos] = 1
                vectors.append(v)
    
    vec_array = np.array(vectors, dtype=np.int64)
    return vec_array

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill
    
    # First: try the D11 lattice constructive approach
    d11_vectors = _construct_d11_lattice()
    n_d11 = len(d11_vectors)
    
    # Verify the D11 set is valid: min pairwise dist^2 >= max norm^2
    if n_d11 > 0:
        # Compute pairwise distances efficiently
        # For vectors with norm^2 = 4, dist^2(i,j) = 8 - 2*dot(i,j)
        # We need dist^2 >= 4 for all pairs, i.e., dot(i,j) <= 2
        dots = d11_vectors @ d11_vectors.T
        np.fill_diagonal(dots, 4)  # self-dot is 4
        max_dot = np.max(dots)
        min_dist_sq = 8 - 2 * max_dot
        max_norm_sq = np.max(np.sum(d11_vectors**2, axis=1))
        
        if min_dist_sq >= max_norm_sq:
            # Valid! Return directly
            return d11_vectors
    
    # Fallback: use the SA-based search on W4 vectors
    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)

    # Conflict matrix: two vectors conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Precompute neighbor lists as arrays for fast indexing
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # Precompute conflict rows as boolean arrays for fast AND operations
    conflict_rows = [conflicts[i] for i in range(N4)]

    rng = np.random.default_rng(42)

    def greedy_min_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            if len(cands) > 1:
                scores = np.array([np.mean(degrees[neighbors[c]]) if len(neighbors[c]) > 0 else 0.0 for c in cands])
                chosen = cands[int(np.argmin(scores))]
            else:
                chosen = cands[0]

            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        return in_set

    def polish(in_set):
        """Greedy add every zero-conflict vertex. Returns (set, size)."""
        in_set = in_set.copy()
        cc = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            cc[neighbors[i]] += 1

        changed = True
        while changed:
            changed = False
            cand = np.where((cc == 0) & (~in_set))[0]
            for v in cand:
                if cc[v] == 0 and not in_set[v]:
                    in_set[v] = True
                    cc[neighbors[v]] += 1
                    changed = True
        return in_set, int(in_set.sum())

    def sa_local_search(in_set_init, max_time=8.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        cc = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            cc[neighbors[i]] += 1

        T = 3.0
        cooling = 0.9993
        min_T = 0.03

        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline and T > min_T:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break

            r = rng.random()

            if r < 0.55 and len(not_in_set) > 0:
                # ADD: pick vertex with fewest in-set conflicts
                counts = cc[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = int(min_count)
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.4
                else:
                    accept = rng.random() < np.exp(delta / T)

                if accept:
                    # Vectorized: find all in-set neighbors of v
                    to_remove = np.where(conflict_rows[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        # Vectorized cc update
                        all_nbrs = np.concatenate([neighbors[u] for u in to_remove]) if len(to_remove) > 0 else np.array([], dtype=np.int64)
                        if len(all_nbrs) > 0:
                            unique_nbrs, counts_arr = np.unique(all_nbrs, return_counts=True)
                            cc[unique_nbrs] -= counts_arr
                    in_set[v] = True
                    current_size += delta
                    cc[neighbors[v]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()

            elif r < 0.75 and len(in_set_indices) > 0:
                # REMOVE
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                if accept:
                    in_set[v] = False
                    current_size += delta
                    cc[neighbors[v]] -= 1

            else:
                # SMART SWAP
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    if len(not_in_set) > 200:
                        sample_idx = rng.choice(len(not_in_set), size=200, replace=False)
                        sample_cands = not_in_set[sample_idx]
                    else:
                        sample_cands = not_in_set
                    
                    conflict_counts = np.array([np.sum(conflict_rows[v] & in_set) for v in sample_cands])
                    min_cc = np.min(conflict_counts)
                    best_cands = sample_cands[conflict_counts == min_cc]
                    v_add = best_cands[rng.integers(len(best_cands))]
                    
                    to_remove = np.where(conflict_rows[v_add] & in_set)[0]
                    delta = 1 - len(to_remove)
                    
                    if delta >= 0:
                        accept = True
                    else:
                        accept = rng.random() < np.exp(delta / T)
                    
                    if accept:
                        if len(to_remove) > 0:
                            in_set[to_remove] = False
                            all_nbrs = np.concatenate([neighbors[u] for u in to_remove])
                            if len(all_nbrs) > 0:
                                unique_nbrs, counts_arr = np.unique(all_nbrs, return_counts=True)
                                cc[unique_nbrs] -= counts_arr
                        in_set[v_add] = True
                        current_size += delta
                        cc[neighbors[v_add]] += 1
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()

            T *= cooling

        # Final exact polish on the best set found during SA
        best_set, best_size = polish(local_best_in_set)
        return best_set, best_size

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    n_restarts = 0
    max_restarts = 50
    while time.time() < deadline - 12 and n_restarts < max_restarts:
        n_restarts += 1

        in_set = greedy_min_degree()
        if time.time() > deadline - 12:
            break

        in_set, cur_size = sa_local_search(in_set)
        if cur_size > best_size:
            best_size = cur_size
            best_in_set = in_set.copy()

        # Perturbation restart from the current best
        if n_restarts > 1 and time.time() < deadline - 15:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            if len(indices) > 4:
                to_remove = rng.choice(indices, size=int(rng.integers(3, 6)), replace=False)
                perturbed[to_remove] = False
            elif len(indices) > 2:
                to_remove = rng.choice(indices, size=2, replace=False)
                perturbed[to_remove] = False
            perturbed, psize = sa_local_search(perturbed)
            if psize > best_size:
                best_size = psize
                best_in_set = perturbed.copy()

    # Final result: selected W4 vectors + all W1 (norm-4) vectors
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]

    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
