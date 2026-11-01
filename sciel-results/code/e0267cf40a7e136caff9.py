"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # ---------- Generate candidate vectors (squared norm 4) ----------
    # W4: weight 4, entries in {-1,0,1}
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    # W1: weight 1, entries in {-2,0,2}
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    N4 = len(vectors_v4)

    # ---------- Group W4 vectors by support ----------
    support_to_indices = {}
    for i in range(N4):
        support = tuple(np.nonzero(vectors_v4[i])[0])
        if support not in support_to_indices:
            support_to_indices[support] = []
        support_to_indices[support].append(i)

    supports = list(support_to_indices.keys())
    n_supports = len(supports)
    vectors_per_support = 16

    # ---------- Support-level conflict graph ----------
    # Two supports conflict if they share >= 3 positions.
    conflict = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            sj = set(supports[j])
            if len(si & sj) >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    neighbors = [np.where(conflict[i])[0] for i in range(n_supports)]
    degree = np.array([len(n) for n in neighbors], dtype=np.int64)

    rng = np.random.default_rng(12345)

    # ---------- Greedy independent set (min-degree) ----------
    def greedy_init(seed):
        r = np.random.default_rng(seed)
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        deg = degree.copy()
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0:
                break
            min_deg = deg[alive_idx].min()
            cands = alive_idx[deg[alive_idx] == min_deg]
            chosen = cands[r.integers(len(cands))]
            in_set[chosen] = True
            to_remove = np.unique(np.concatenate(([chosen], neighbors[chosen])))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                deg[rem] -= conflict[rem][:, to_remove].sum(axis=1)
        return in_set

    # ---------- Min-conflict local search ----------
    def local_search(in_set_init, max_time):
        in_set = in_set_init.copy()
        cur = in_set.sum()
        best = cur
        best_set = in_set.copy()

        # count_conflicts[s] = number of in-set neighbors of s
        count_conflicts = np.zeros(n_supports, dtype=np.int64)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        T = 2.0
        cooling = 0.999
        min_T = 0.05
        stuck = 0
        dl = time.time() + max_time

        while time.time() < dl:
            if rng.random() < 0.6:
                # Single insert
                not_in = np.where(~in_set)[0]
                if len(not_in) == 0:
                    break
                counts = count_conflicts[not_in]
                mc = counts.min()
                cands = not_in[counts == mc]
                s = cands[rng.integers(len(cands))]
                k = counts[s]
                delta = 1 - k
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                if accept:
                    to_remove = np.where(conflict[s] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[s] = True
                    cur += delta
                    count_conflicts[neighbors[s]] += 1
                    if cur > best:
                        best = cur
                        best_set = in_set.copy()
                        stuck = 0
                    else:
                        stuck += 1
            else:
                # 2-for-1 swap
                not_in = np.where(~in_set)[0]
                if len(not_in) == 0:
                    break
                n_sample = min(6, len(not_in))
                sample = not_in[rng.choice(len(not_in), size=n_sample, replace=False)]
                done = False
                for s in sample:
                    conf_in = np.where(conflict[s] & in_set)[0]
                    k = len(conf_in)
                    if k < 2:
                        continue
                    delta = 1 - k
                    if delta >= 0:
                        accept = True
                    else:
                        accept = rng.random() < np.exp(delta / T)
                    if accept:
                        in_set[conf_in] = False
                        for u in conf_in:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[s] = True
                        cur += delta
                        count_conflicts[neighbors[s]] += 1
                        if cur > best:
                            best = cur
                            best_set = in_set.copy()
                            stuck = 0
                        else:
                            stuck += 1
                        done = True
                        break
                if not done:
                    stuck += 1

            if stuck > 1200:
                T = max(T, 2.0)
                stuck = 0
            T *= cooling
            if T < min_T:
                T = min_T

        return best_set, best

    # ---------- Exact local polish (swap-based, exact feasibility) ----------
    def exact_polish(in_set_init, max_time):
        in_set = in_set_init.copy()
        cur = in_set.sum()
        best = cur
        best_set = in_set.copy()
        dl = time.time() + max_time
        improved = True
        while improved and time.time() < dl:
            improved = False
            in_set_idx = np.where(in_set)[0]
            # For each in-set node, its conflicts that are in-set
            for i in in_set_idx:
                # try removing i and adding a non-in node that conflicts only with i
                not_in = np.where(~in_set)[0]
                if len(not_in) == 0:
                    break
                # candidates s not in set with conflict[s] & in_set == {i}
                # compute count of in-set conflicts for each not_in
                counts = conflict[not_in][:, in_set_idx].sum(axis=1)
                cands = not_in[counts == 1]
                for s in cands:
                    # s conflicts with exactly one in-set node; if that node is i, we can swap
                    if conflict[s, i]:
                        in_set[i] = False
                        in_set[s] = True
                        cur += 0  # swap keeps size
                        improved = True
                        break
                if improved:
                    break
            if cur > best:
                best = cur
                best_set = in_set.copy()
        return best_set, best

    # ---------- Branch and bound with color-based upper bound ----------
    def upper_bound(alive):
        # Greedy coloring of the conflict subgraph on 'alive' nodes.
        # Returns the number of colors = upper bound on independence number.
        n_alive = alive.sum()
        if n_alive == 0:
            return 0
        alive_idx = np.where(alive)[0]
        # Order by descending degree within alive (approx)
        degs = conflict[alive_idx][:, alive_idx].sum(axis=1)
        order = alive_idx[np.argsort(-degs)]
        color = {}
        for v in order:
            # find smallest color not used by colored neighbors
            used = set()
            for u in neighbors[v]:
                if u in color:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
        return max(color.values()) + 1 if color else 0

    def branch_and_bound(initial_set, max_time):
        best = initial_set.sum()
        best_set = initial_set.copy()
        dl = time.time() + max_time
        # Stack of (alive_mask, current_set_size)
        # Start: alive = all nodes not in initial_set
        initial_alive = ~initial_set
        stack = [(initial_alive, initial_set.sum())]
        # We want to find a larger independent set; treat as maximizing.
        # Use best-first: prioritize nodes with higher potential.
        # For simplicity, DFS with upper-bound pruning.
        # Reorder stack to explore promising branches first.
        while stack and time.time() < dl:
            alive, cur_size = stack.pop()
            ub = cur_size + upper_bound(alive)
            if ub <= best:
                continue
            if not np.any(alive):
                if cur_size > best:
                    best = cur_size
                    best_set = np.zeros(n_supports, dtype=bool)
                continue
            # Pick a node to branch on: highest degree in alive
            alive_idx = np.where(alive)[0]
            degs = conflict[alive_idx][:, alive_idx].sum(axis=1)
            v = alive_idx[np.argmax(degs)]
            # Branch 1: include v
            alive_include = alive.copy()
            alive_include[v] = False
            alive_include[neighbors[v]] = False
            stack.append((alive_include, cur_size + 1))
            # Branch 2: exclude v
            alive_exclude = alive.copy()
            alive_exclude[v] = False
            stack.append((alive_exclude, cur_size))

        return best, best_set

    # ---------- Main search loop ----------
    best_size = 0
    best_support_set = np.zeros(n_supports, dtype=bool)

    # Phase 1: Greedy + local search (multiple restarts)
    n_restarts = 0
    max_restarts = 50
    while time.time() < deadline - 60 and n_restarts < max_restarts:
        n_restarts += 1
        in_set = greedy_init(1000 + n_restarts)
        cur = in_set.sum()
        if cur > best_size:
            best_size = cur
            best_support_set = in_set.copy()

        remaining = deadline - 60 - time.time()
        if remaining < 2:
            break
        sa_time = min(remaining * 0.4, 10.0)
        if sa_time < 1:
            sa_time = 1.0
        sa_set, sa_size = local_search(in_set, sa_time)
        if sa_size > best_size:
            best_size = sa_size
            best_support_set = sa_set.copy()

    # Phase 2: Exact local polish on best
    remaining = deadline - 20 - time.time()
    if remaining > 2:
        polish_time = min(remaining * 0.5, 15.0)
        pol_set, pol_size = exact_polish(best_support_set, polish_time)
        if pol_size > best_size:
            best_size = pol_size
            best_support_set = pol_set.copy()

    # Phase 3: Branch and bound to try to certify/improve
    remaining = deadline - 10 - time.time()
    if remaining > 5:
        bb_time = min(remaining * 0.8, 40.0)
        bb_set, bb_size = branch_and_bound(best_support_set, bb_time)
        if bb_size > best_size:
            best_size = bb_size
            best_support_set = bb_set.copy()

    # ---------- Build final vector list ----------
    selected_support_indices = np.where(best_support_set)[0]
    final_vectors = []
    for si in selected_support_indices:
        support = supports[si]
        for idx in support_to_indices[support]:
            final_vectors.append(vectors_v4[idx])
    final_vectors.extend(v1_candidates)

    result = np.array(final_vectors, dtype=np.int64)

    if len(result) == 0:
        # Fallback
        all_supports = list(itertools.combinations(range(D), 4))
        selected_supports = []
        for s in all_supports:
            valid = True
            for t in selected_supports:
                if len(set(s).intersection(t)) > 2:
                    valid = False
                    break
            if valid:
                selected_supports.append(s)
        fallback_vectors = []
        for s in selected_supports:
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for pos, sign in zip(s, signs):
                    v[pos] = sign
                fallback_vectors.append(v)
        for pos in range(D):
            for sign in [1, -1]:
                v = [0] * D
                v[pos] = 2 * sign
                fallback_vectors.append(v)
        result = np.array(fallback_vectors, dtype=np.int64)

    return result

# EVOLVE-BLOCK-END
