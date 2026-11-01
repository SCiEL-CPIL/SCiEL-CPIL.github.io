<<<<<<< SEARCH
def _max_independent_set_conflict(conflicts, neighbors, neighbor_counts, rng, max_time=8.0):
    """Combined greedy + SA for max independent set on conflict graph."""
    N = len(neighbor_counts)
    
    def greedy_min_degree():
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < max_time - 5:
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
        """Greedy add every zero-conflict vertex."""
        in_set = in_set.copy()
        cc = np.zeros(N, dtype=np.int32)
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

        cc = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            cc[neighbors[i]] += 1

        T = 3.0
        cooling = 0.9995
        min_T = 0.05

        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        step = 0

        while time.time() < sa_deadline and T > min_T:
            step += 1
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break

            r = rng.random()

            if r < 0.6 and len(not_in_set) > 0:
                counts = cc[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = int(min_count)
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.35
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

            elif r < 0.8 and len(in_set_indices) > 0:
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                if accept:
                    in_set[v] = False
                    current_size += delta
                    cc[neighbors[v]] -= 1

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

            T *= cooling

        best_set, best_size = polish(local_best_in_set)
        return best_set, best_size

    # Main loop
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    n_restarts = 0
    max_restarts = 40
    start_time = time.time()
    deadline = start_time + max_time

    while time.time() < deadline - 12 and n_restarts < max_restarts:
        n_restarts += 1

        in_set = greedy_min_degree()
        if time.time() > deadline - 12:
            break

        in_set, cur_size = sa_local_search(in_set)
        if cur_size > best_size:
            best_size = cur_size
            best_in_set = in_set.copy()

        if n_restarts > 1 and time.time() < deadline - 15:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            if len(indices) > 4:
                to_remove = rng.choice(indices, size=int(rng.integers(3, 5)), replace=False)
                perturbed[to_remove] = False
            elif len(indices) > 2:
                to_remove = rng.choice(indices, size=2, replace=False)
                perturbed[to_remove] = False
            perturbed, psize = sa_local_search(perturbed)
            if psize > best_size:
                best_size = psize
                best_in_set = perturbed.copy()

    return best_in_set, best_size
=======
def _max_independent_set_conflict(conflicts, neighbors, neighbor_counts, rng, max_time=8.0):
    """Combined greedy + SA for max independent set on conflict graph."""
    N = len(neighbor_counts)
    # Precompute conflict matrix as int8 for fast summing
    conflicts_int8 = conflicts.astype(np.int8)

    def greedy_min_degree():
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.int32)

        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            if len(cands) > 1:
                # Vectorized tie-break: prefer vertices whose neighbors have lower degree
                nbr_degrees = np.array([degrees[neighbors[c]] if len(neighbors[c]) > 0 else np.array([0]) for c in cands])
                scores = np.array([np.mean(nd) if len(nd) > 0 else 0.0 for nd in nbr_degrees])
                chosen = cands[int(np.argmin(scores))]
            else:
                chosen = cands[0]

            in_set[chosen] = True
            to_remove = np.unique(np.concatenate(([chosen], neighbors[chosen])))
            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts_int8[rem][:, to_remove].sum(axis=1)

        return in_set

    def polish(in_set):
        """Vectorized greedy add every zero-conflict vertex."""
        in_set = in_set.copy()
        # Compute conflict counts for all vertices using matrix multiply
        in_set_int = in_set.astype(np.int8)
        cc = conflicts_int8 @ in_set_int  # cc[i] = number of in-set neighbors of i

        # Iteratively add vertices with zero conflicts
        changed = True
        while changed:
            changed = False
            candidates = np.where((cc == 0) & (~in_set))[0]
            if len(candidates) == 0:
                break
            # Add all zero-conflict candidates at once (they don't conflict with each other
            # only if they're mutually non-adjacent; check this)
            # For safety, add one at a time but vectorize the check
            # Actually, if two candidates both have cc==0, they might still conflict with each other
            # So we need to be careful. Let's add them greedily in order.
            for v in candidates:
                if cc[v] == 0 and not in_set[v]:
                    in_set[v] = True
                    cc[neighbors[v]] += 1
                    changed = True
        return in_set, int(in_set.sum())

    def sa_local_search(in_set_init, max_time=8.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        # Precompute conflict counts using matrix multiply
        in_set_int = in_set.astype(np.int8)
        cc = (conflicts_int8 @ in_set_int).astype(np.int32)

        T = 3.0
        cooling = 0.9995
        min_T = 0.05

        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        # Cache indices, update incrementally
        in_set_indices = np.where(in_set)[0]
        not_in_set_indices = np.where(~in_set)[0]

        while time.time() < sa_deadline and T > min_T:
            r = rng.random()

            if r < 0.6 and len(not_in_set_indices) > 0:
                counts = cc[not_in_set_indices]
                min_count = np.min(counts)
                min_cands = not_in_set_indices[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = int(cc[v])
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.35
                else:
                    accept = rng.random() < np.exp(delta / T)

                if accept:
                    # Find conflicts with current in_set
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        cc[neighbors[to_remove]] -= 1
                        # Update index caches
                        to_remove_set = set(to_remove.tolist())
                        in_set_indices = np.array([x for x in in_set_indices if x not in to_remove_set])
                        not_in_set_indices = np.append(not_in_set_indices, to_remove)
                        not_in_set_indices.sort()
                    in_set[v] = True
                    current_size += delta
                    cc[neighbors[v]] += 1
                    # Update index caches for v
                    if v in not_in_set_indices:
                        mask = not_in_set_indices != v
                        not_in_set_indices = not_in_set_indices[mask]
                    in_set_indices = np.append(in_set_indices, v)
                    in_set_indices.sort()
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()

            elif r < 0.8 and len(in_set_indices) > 0:
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                accept = rng.random() < np.exp(delta / T)
                if accept:
                    in_set[v] = False
                    current_size += delta
                    cc[neighbors[v]] -= 1
                    # Update index caches
                    mask = in_set_indices != v
                    in_set_indices = in_set_indices[mask]
                    not_in_set_indices = np.append(not_in_set_indices, v)
                    not_in_set_indices.sort()

            else:
                if len(in_set_indices) > 0 and len(not_in_set_indices) > 0:
                    v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                    v_add = not_in_set_indices[rng.integers(len(not_in_set_indices))]
                    conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = conflicts_with[conflicts_with != v_remove]
                    if len(other_conflicts) == 0:
                        in_set[v_remove] = False
                        in_set[v_add] = True
                        cc[neighbors[v_remove]] -= 1
                        cc[neighbors[v_add]] += 1
                        # Update index caches
                        mask_rm = in_set_indices != v_remove
                        in_set_indices = in_set_indices[mask_rm]
                        mask_add = not_in_set_indices != v_add
                        not_in_set_indices = not_in_set_indices[mask_add]
                        in_set_indices = np.append(in_set_indices, v_add)
                        in_set_indices.sort()
                        not_in_set_indices = np.append(not_in_set_indices, v_remove)
                        not_in_set_indices.sort()
                    else:
                        to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                        delta = 1 - len(to_remove)
                        accept = delta > 0 or (rng.random() < np.exp(delta / T))
                        if accept:
                            in_set[to_remove] = False
                            cc[neighbors[to_remove]] -= 1
                            in_set[v_add] = True
                            current_size += delta
                            cc[neighbors[v_add]] += 1
                            # Update index caches
                            to_remove_set = set(to_remove.tolist())
                            in_set_indices = np.array([x for x in in_set_indices if x not in to_remove_set])
                            not_in_set_indices = np.append(not_in_set_indices, to_remove)
                            mask_add = not_in_set_indices != v_add
                            not_in_set_indices = not_in_set_indices[mask_add]
                            in_set_indices = np.append(in_set_indices, v_add)
                            in_set_indices.sort()
                            not_in_set_indices.sort()
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()

            T *= cooling

        best_set, best_size = polish(local_best_in_set)
        return best_set, best_size

    # Main loop
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    n_restarts = 0
    max_restarts = 60
    start_time = time.time()
    deadline = start_time + max_time

    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1

        in_set = greedy_min_degree()
        if time.time() > deadline - 10:
            break

        in_set, cur_size = sa_local_search(in_set)
        if cur_size > best_size:
            best_size = cur_size
            best_in_set = in_set.copy()

        if n_restarts > 1 and time.time() < deadline - 12:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            if len(indices) > 5:
                to_remove = rng.choice(indices, size=int(rng.integers(3, 6)), replace=False)
                perturbed[to_remove] = False
            elif len(indices) > 2:
                to_remove = rng.choice(indices, size=2, replace=False)
                perturbed[to_remove] = False
            perturbed, psize = sa_local_search(perturbed)
            if psize > best_size:
                best_size = psize
                best_in_set = perturbed.copy()

    # Final polish
    best_in_set, best_size = polish(best_in_set)
    return best_in_set, best_size
>>>>>>> REPLACE

Now let me also update the `construct` function to use a tighter deadline and slightly different parameters:

