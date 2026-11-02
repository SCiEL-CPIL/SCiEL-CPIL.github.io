"""Kissing number in dimension 11 - combined design with improved SA and polish."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

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

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    # W1 vectors are always compatible with each other and with W4 vectors
    # (dot product between W1 and W4 is at most 2, between W1 and W1 is at most 4
    #  but two W1 vectors with same position and sign have dot=4, opposite sign dot=-4,
    #  different positions dot=0. So W1-W1 conflicts exist only for same-position pairs.)
    # Actually: W1 vector e_i * 2 and e_j * 2: dot = 4 if i==j, 0 if i!=j.
    # So W1 vectors at the same position conflict! We need to handle this.
    # But we only have one vector per (position, sign), so no two W1 vectors
    # share the same position AND sign. Two W1 vectors at same position have
    # opposite signs, so dot = -4, which is fine (|dot| <= 2 is not required,
    # we need distance >= max_norm, i.e., 4 - 2*dot >= 4, i.e., dot <= 0).
    # dot = -4 < 0, so OK. Different positions: dot = 0, OK.
    # So all 22 W1 vectors are mutually compatible.
    # W1 vs W4: dot = 2*s where s is the sign of the W4 entry at that position.
    # |dot| = 2, so 4 - 2*dot >= 4 iff dot <= 0. If dot = 2, then 4-4=0 < 4. CONFLICT!
    # Wait, let me re-check. Distance squared = ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b = 4+4-2*a.b = 8-2*a.b
    # We need 8-2*a.b >= 4 (max norm squared), so a.b <= 2.
    # W1·W4: if W4 has entry +1 at position i, W1 has entry +2 at position i, dot = 2. OK (<=2).
    # If W4 has entry -1 at position i, W1 has entry +2 at position i, dot = -2. OK.
    # If W4 has entry +1 at position i, W1 has entry -2 at position i, dot = -2. OK.
    # So W1 and W4 are always compatible! Good.
    # W1·W1: same position, opposite signs: dot = -4. OK. Different positions: 0. OK.
    # So ALL 22 W1 vectors can be included unconditionally.

    # Conflict matrix for W4: two vectors conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

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
                scores = np.array([np.sum(neighbor_counts[neighbors[c]]) for c in cands])
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
        """Greedy add every zero-conflict vertex repeatedly until fixed point."""
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

    def polish_deep(in_set):
        """
        Try to improve by swapping: remove a small set of conflicting nodes, add one node, then greedy fill.
        """
        in_set = in_set.copy()
        current_size = int(in_set.sum())
        
        cc = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            cc[neighbors[i]] += 1

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            not_in_set = np.where(~in_set)[0]
            for v in not_in_set:
                if cc[v] == 0:
                    in_set[v] = True
                    cc[neighbors[v]] += 1
                    current_size += 1
                    improved = True
                    continue
                
                # Check swaps
                nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                
                # Try swapping 1 out, 1 in (if cc[v]==1)
                if len(nbrs_in_set) == 1:
                    u = nbrs_in_set[0]
                    in_set[u] = False
                    cc[neighbors[u]] -= 1
                    
                    in_set[v] = True
                    cc[neighbors[v]] += 1
                    
                    new_set, new_size = polish(in_set)
                    
                    if new_size > current_size:
                        in_set = new_set
                        current_size = new_size
                        cc = np.zeros(N4, dtype=np.int32)
                        for i in np.where(in_set)[0]:
                            cc[neighbors[i]] += 1
                        improved = True
                    else:
                        in_set[v] = False
                        cc[neighbors[v]] -= 1
                        in_set[u] = True
                        cc[neighbors[u]] += 1
                
                # Try swapping 2 out, 1 in (if cc[v]==2)
                elif len(nbrs_in_set) == 2:
                    u1, u2 = nbrs_in_set
                    in_set[u1] = False
                    in_set[u2] = False
                    cc[neighbors[u1]] -= 1
                    cc[neighbors[u2]] -= 1
                    
                    in_set[v] = True
                    cc[neighbors[v]] += 1
                    
                    new_set, new_size = polish(in_set)
                    
                    if new_size > current_size:
                        in_set = new_set
                        current_size = new_size
                        cc = np.zeros(N4, dtype=np.int32)
                        for i in np.where(in_set)[0]:
                            cc[neighbors[i]] += 1
                        improved = True
                    else:
                        in_set[v] = False
                        cc[neighbors[v]] -= 1
                        in_set[u1] = True
                        in_set[u2] = True
                        cc[neighbors[u1]] += 1
                        cc[neighbors[u2]] += 1

        return in_set, current_size

    def sa_local_search(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        cc = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            cc[neighbors[i]] += 1

        T = 3.0
        cooling = 0.9995
        min_T = 0.05

        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        stuck = 0

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0:
                break

            r = rng.random()

            if r < 0.55 and len(not_in_set) > 0:
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
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            cc[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    cc[neighbors[v]] += 1
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck = 0
                    else:
                        stuck += 1
                else:
                    stuck += 1

            elif r < 0.75 and len(in_set_indices) > 0:
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                if accept:
                    in_set[v] = False
                    current_size += delta
                    cc[neighbors[v]] -= 1
                    stuck += 1
                else:
                    stuck += 1

            else:
                if len(in_set_indices) > 0 and len(not_in_set) > 0:
                    v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                    v_add = not_in_set[rng.integers(len(not_in_set))]
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_remove]
                    if len(other_conflicts) == 0:
                        in_set[v_remove] = False
                        in_set[v_add] = True
                        cc[neighbors[v_remove]] -= 1
                        cc[neighbors[v_add]] += 1
                        stuck += 1
                    else:
                        to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                        delta = 1 - len(to_remove)
                        accept = delta > 0 or (rng.random() < np.exp(delta / T))
                        if accept:
                            in_set[to_remove] = False
                            for u in to_remove:
                                cc[neighbors[u]] -= 1
                            in_set[v_add] = True
                            current_size += delta
                            cc[neighbors[v_add]] += 1
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()
                                stuck = 0
                            else:
                                stuck += 1
                        else:
                            stuck += 1

            if stuck > 2000:
                T = max(T, 2.0)
                stuck = 0

            T *= cooling
            if T < min_T:
                T = min_T

        # Final polish
        best_set, best_size = polish_deep(local_best_in_set)
        return best_set, best_size

    def verify_config(v4_indices, v1_vectors):
        """Verify the full configuration is valid."""
        if len(v4_indices) == 0:
            full = v1_vectors
        else:
            full = np.vstack([vectors_v4[v4_indices], v1_vectors])
        
        n = len(full)
        if n == 0:
            return True
        
        norms_sq = np.sum(full.astype(np.float64) ** 2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        dots = full.astype(np.float64) @ full.astype(np.float64).T
        dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        np.fill_diagonal(dists_sq, np.inf)
        min_dist_sq = np.min(dists_sq)
        
        return min_dist_sq >= max_norm_sq - 1e-9

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    n_restarts = 0
    max_restarts = 50
    while time.time() < deadline - 15 and n_restarts < max_restarts:
        n_restarts += 1

        in_set = greedy_min_degree()
        if time.time() > deadline - 15:
            break

        remaining = deadline - 15 - time.time()
        sa_time = min(remaining * 0.55, 12.0)
        if sa_time < 2.0:
            sa_time = 2.0

        in_set, cur_size = sa_local_search(in_set, max_time=sa_time)
        if cur_size > best_size:
            best_size = cur_size
            best_in_set = in_set.copy()

        # Perturbation restart from best
        if n_restarts > 1 and time.time() < deadline - 20:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            n_rem = max(2, int(len(indices) * 0.08))
            n_rem = min(n_rem, max(2, len(indices) - 2))
            if len(indices) > n_rem:
                to_remove = rng.choice(indices, size=n_rem, replace=False)
                perturbed[to_remove] = False
            perturbed, psize = sa_local_search(perturbed, max_time=min(remaining * 0.4, 10.0))
            if psize > best_size:
                best_size = psize
                best_in_set = perturbed.copy()

    # Final result: selected W4 vectors + all W1 vectors
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]

    final_vectors = np.vstack([result_v4, vectors_v1])
    
    # Verify before returning
    if not verify_config(final_v4_indices, vectors_v1):
        # Fallback: return just W1 vectors which are always valid
        return vectors_v1
    
    return final_vectors
