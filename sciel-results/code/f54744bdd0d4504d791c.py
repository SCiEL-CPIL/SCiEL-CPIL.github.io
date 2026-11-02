"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    # ---- Candidate generation (norm^2 = 4) ----
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

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)
    if N4 == 0:
        return vectors_v1

    # ---- Conflict / compatibility structure ----
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2            # conflict if dot > 2
    np.fill_diagonal(conflicts, False)
    compat = ~conflicts             # compatibility graph (edges = non-conflict)
    np.fill_diagonal(compat, False)

    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    # Compatibility adjacency as per-vertex bitmasks (for fast clique work)
    compat_mask = np.zeros(N4, dtype=object)
    for i in range(N4):
        compat_mask[i] = np.packbits(compat[i])

    rng = np.random.default_rng(42)

    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)

    # ===================================================================
    # 1) GREEDY MIN-DEGREE INDEPENDENT SET (initial solutions)
    # ===================================================================
    def greedy_min_degree(seed_offset=0):
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy().astype(np.float64)
        local_rng = np.random.default_rng(42 + seed_offset)

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] <= min_deg + 1e-9]
            if len(cands) > 1:
                scores = np.zeros(len(cands))
                for idx, c in enumerate(cands):
                    nbrs_c = neighbors[c]
                    alive_nbrs = nbrs_c[alive[nbrs_c]]
                    if len(alive_nbrs) > 0:
                        scores[idx] = np.sum(degrees[alive_nbrs])
                noise = local_rng.random(len(cands)) * 0.1
                scores += noise
                chosen = cands[np.argmin(scores)]
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

    # ===================================================================
    # 2) SIMULATED ANNEALING (refinement / escape local optima)
    # ===================================================================
    def sa_local_search(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        T = 4.0
        cooling = 0.9993
        min_T = 0.05

        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time

        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]
            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break

            # Kick: escape deep local optima
            if stuck_counter > 3000:
                n_remove = max(1, int(0.15 * len(in_set_indices)))
                if n_remove < len(in_set_indices):
                    remove_idx = rng.choice(len(in_set_indices), size=n_remove, replace=False)
                    for ri in remove_idx:
                        v = in_set_indices[ri]
                        in_set[v] = False
                        count_conflicts[neighbors[v]] -= 1
                    current_size -= n_remove
                    stuck_counter = 0
                    for v in not_in_set:
                        if count_conflicts[v] == 0:
                            in_set[v] = True
                            count_conflicts[neighbors[v]] += 1
                            current_size += 1
                    not_in_set = np.where(~in_set)[0]
                    in_set_indices = np.where(in_set)[0]
                    continue

            move_type = rng.random()
            if move_type < 0.45 and len(not_in_set) > 0:
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                k = min_count
                delta = 1 - k
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.70 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                in_degrees = count_conflicts[in_set_indices]
                weights = 1.0 / (1.0 + in_degrees.astype(np.float64))
                weights /= weights.sum()
                v_remove = in_set_indices[rng.choice(len(in_set_indices), p=weights)]
                add_counts = count_conflicts[not_in_set]
                add_weights = 1.0 / (1.0 + add_counts.astype(np.float64))
                add_weights /= add_weights.sum()
                v_add = not_in_set[rng.choice(len(not_in_set), p=add_weights)]
                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]
                if len(other_conflicts) == 0:
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                else:
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = delta > 0 or (rng.random() < np.exp(delta / T))
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1
                    else:
                        stuck_counter += 1

            elif move_type < 0.85 and len(in_set_indices) >= 2 and len(not_in_set) > 0:
                idx1, idx2 = rng.choice(len(in_set_indices), size=2, replace=False)
                v_rem1 = in_set_indices[idx1]
                v_rem2 = in_set_indices[idx2]
                v_add = not_in_set[rng.integers(len(not_in_set))]
                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[~np.isin(conflicts_with, [v_rem1, v_rem2])]
                if len(other_conflicts) == 0:
                    to_remove = [v_rem1, v_rem2]
                    delta = 1 - 2
                    accept = rng.random() < np.exp(delta / T)
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            else:
                if len(in_set_indices) > 0:
                    v = in_set_indices[rng.integers(len(in_set_indices))]
                    accept = rng.random() < np.exp(-1.0 / T)
                    if accept:
                        in_set[v] = False
                        current_size -= 1
                        count_conflicts[neighbors[v]] -= 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            if stuck_counter > 1500:
                T = max(T, 3.0)
                stuck_counter = 0
            T *= cooling
            if T < min_T:
                T = min_T

        return local_best_in_set, local_best_size

    # ===================================================================
    # 3) MAX-CLIQUE BRANCH-AND-BOUND (certify / improve the W4 selection)
    #    Independent set on W4 == maximum clique on compatibility graph.
    #    Greedy coloring gives a tight upper bound (Tomita style).
    # ===================================================================
    def max_clique_bnb(initial_clique, max_time=20.0):
        """Branch-and-bound max clique on the compatibility graph.
        Returns (best_clique_list, best_size)."""
        bnb_deadline = time.time() + max_time
        nonlocal best_size, best_in_set

        # order vertices by descending compatibility degree (good for B&B)
        deg = compat.sum(axis=1)
        order = np.argsort(-deg)

        # greedy coloring of the whole graph -> per-vertex color (upper bound)
        colors = np.zeros(N4, dtype=np.int32)
        used = np.zeros(N4, dtype=bool)
        for v in order:
            c = 1
            while True:
                ok = True
                for w in order:
                    if used[w] and colors[w] == c and compat[v, w]:
                        ok = False
                        break
                if ok:
                    break
                c += 1
            colors[v] = c
            used[v] = True
        color_max = int(colors.max()) if N4 > 0 else 0

        # greedy clique heuristic for a strong initial lower bound
        def greedy_clique():
            clique = []
            cand = set(range(N4))
            while cand:
                best_v = None
                best_score = -1
                for v in cand:
                    s = len([w for w in cand if w != v and compat[v, w]])
                    if s > best_score:
                        best_score = s
                        best_v = v
                clique.append(best_v)
                cand = {w for w in cand if w != best_v and compat[best_v, w]}
            return clique

        # build initial clique from the SA independent set (a valid clique in compat)
        sa_clique = [int(i) for i in np.where(initial_clique)[0]]
        # ensure it is actually a clique (SA guarantees it, but be safe)
        for a in range(len(sa_clique)):
            for b in range(a + 1, len(sa_clique)):
                if not compat[sa_clique[a], sa_clique[b]]:
                    sa_clique = []
                    break
            if not sa_clique:
                break
        if not sa_clique:
            sa_clique = greedy_clique()

        best_clique = list(sa_clique)
        best_csize = len(best_clique)
        if best_csize > best_size:
            best_size = best_csize
            tmp = np.zeros(N4, dtype=bool)
            tmp[best_clique] = True
            best_in_set = tmp.copy()

        # branch-and-bound
        # P = candidate set (list), C = current clique (list)
        # bound = |C| + greedy-coloring(|P|)
        def color_bound(P):
            """Greedy sequential coloring of P; returns number of colors (upper bound)."""
            if not P:
                return 0
            # order P by degree within P
            degP = {v: sum(1 for w in P if w != v and compat[v, w]) for v in P}
            Pord = sorted(P, key=lambda v: -degP[v])
            col = {}
            ncol = 0
            for v in Pord:
                used_colors = set()
                for w in Pord:
                    if w == v:
                        break
                    if w in col and compat[v, w]:
                        used_colors.add(col[w])
                c = 1
                while c in used_colors:
                    c += 1
                col[v] = c
                ncol = max(ncol, c)
            return ncol

        # recursive search with explicit stack to avoid recursion limits
        # state: (P_tuple, C_list)
        stack = [(tuple(range(N4)), [])]
        # start from a good ordering: process the initial clique region first
        # We'll do a depth-first using recursion with manual bound checks.

        import sys
        sys.setrecursionlimit(100000)

        def bnb_rec(P, C):
            nonlocal best_clique, best_csize
            if len(C) + len(P) <= best_csize:
                return
            if not P:
                if len(C) > best_csize:
                    best_csize = len(C)
                    best_clique = list(C)
                return
            # bound via coloring
            cb = color_bound(P)
            if len(C) + cb <= best_csize:
                return
            # expand: pick vertices from P in color order (high color first)
            degP = {v: sum(1 for w in P if w != v and compat[v, w]) for v in P}
            Pord = sorted(P, key=lambda v: -degP[v])
            # greedy color Pord for ordering
            col = {}
            ncol = 0
            for v in Pord:
                used_colors = set()
                for w in Pord:
                    if w == v:
                        break
                    if w in col and compat[v, w]:
                        used_colors.add(col[w])
                c = 1
                while c in used_colors:
                    c += 1
                col[v] = c
                ncol = max(ncol, c)
            # iterate in decreasing color (Tomita)
            for v in sorted(Pord, key=lambda x: -col[x]):
                if time.time() > bnb_deadline:
                    return
                if len(C) + 1 + col[v] <= best_csize:
                    continue
                # new candidate set = vertices in P before v that are compatible with v
                newP = [w for w in Pord if w != v and col[w] <= col[v] and compat[v, w]]
                C.append(v)
                bnb_rec(newP, C)
                C.pop()
                if len(C) + len(newP) <= best_csize:
                    return

        bnb_rec(list(range(N4)), [])

        # final update
        if best_csize > best_size:
            best_size = best_csize
            tmp = np.zeros(N4, dtype=bool)
            tmp[best_clique] = True
            best_in_set = tmp.copy()

        return best_clique, best_csize

    # ===================================================================
    # 4) POLISH: greedily add non-conflicting vertices
    # ===================================================================
    def polish_greedy_add(in_set_init, max_time=5.0):
        polish_set = in_set_init.copy()
        polish_size = int(polish_set.sum())
        polish_deadline = time.time() + max_time
        for trial in range(3):
            if time.time() >= polish_deadline:
                break
            not_in = np.where(~polish_set)[0]
            if len(not_in) == 0:
                break
            order = rng.permutation(len(not_in))
            for idx in order:
                if time.time() >= polish_deadline:
                    break
                v = not_in[idx]
                if np.any(conflicts[v] & polish_set):
                    continue
                polish_set[v] = True
                polish_size += 1
        return polish_set, polish_size

    # ===================================================================
    # MAIN SEARCH LOOP
    # ===================================================================
    # Initialize with a few diverse greedy solutions
    for i in range(4):
        if time.time() >= deadline - 20:
            break
        in_set = greedy_min_degree(seed_offset=i)
        sz = int(in_set.sum())
        if sz > best_size:
            best_size = sz
            best_in_set = in_set.copy()

    # Phase A: SA refinement rounds (each round SA + polish)
    sa_rounds = 0
    while time.time() < deadline - 25:
        sa_rounds += 1
        remaining = deadline - 25 - time.time()
        sa_time = min(remaining * 0.5, 8.0)
        if sa_time < 1.5:
            break
        sa_result, sa_size = sa_local_search(best_in_set, max_time=sa_time)
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        # polish
        if time.time() < deadline - 25:
            pol_time = min((deadline - 25 - time.time()) * 0.4, 3.0)
            if pol_time > 0.5:
                pol_result, pol_size = polish_greedy_add(best_in_set, max_time=pol_time)
                if pol_size > best_size:
                    best_size = pol_size
                    best_in_set = pol_result.copy()

    # Phase B: Max-clique branch-and-bound to certify/improve the selection
    if time.time() < deadline - 3:
        bnb_time = min(deadline - 3 - time.time(), 30.0)
        if bnb_time > 3:
            _, bnb_size = max_clique_bnb(best_in_set, max_time=bnb_time)

    # Phase C: final polish
    if time.time() < deadline - 1:
        pol_time = min(deadline - 1 - time.time(), 3.0)
        if pol_time > 0.5:
            final_polish, final_size = polish_greedy_add(best_in_set, max_time=pol_time)
            if final_size > best_size:
                best_size = final_size
                best_in_set = final_polish.copy()

    # ---- Assemble final vector set ----
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors

# EVOLVE-BLOCK-END
