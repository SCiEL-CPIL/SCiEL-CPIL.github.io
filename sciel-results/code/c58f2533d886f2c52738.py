"""Kissing number in dimension 11: maximum independent set in conflict graph of norm-4 integer vectors."""
import numpy as np
import itertools
import time
from collections import deque

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 120s

    # Generate all candidate vectors with norm^2 = 4
    # Type A: weight 4, entries in {-1,0,1}
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    n4 = len(v4_list)

    # Type B: weight 1, entry ±2
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    n1 = len(v1_list)

    n_total = n4 + n1
    all_vectors = np.array(v4_list + v1_list, dtype=np.int64)

    # Compute conflict matrix: conflict if dot > 2 (since norm^2=4, threshold = 4/2 = 2)
    dots = all_vectors @ all_vectors.T
    conflicts = (dots > 2).astype(np.int32)
    np.fill_diagonal(conflicts, 0)

    # Precompute neighbor lists for faster operations
    neighbors = [np.where(conflicts[i] == 1)[0] for i in range(n_total)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    # --- Phase 1: Solve W4-W4 independent set via bipartite matching (if bipartite) ---
    w4_conf = conflicts[:n4, :n4]

    # Check if W4-W4 graph is bipartite using BFS 2-coloring
    color = np.full(n4, -1, dtype=np.int32)
    is_bipartite = True
    for start in range(n4):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            nbrs = np.where(w4_conf[u] == 1)[0]
            for v in nbrs:
                if color[v] == -1:
                    color[v] = color[u] ^ 1
                    queue.append(v)
                elif color[v] == color[u]:
                    is_bipartite = False
                    break
            if not is_bipartite:
                break
        if not is_bipartite:
            break

    w4_indep_set = set()
    w4_indep_size = 0

    if is_bipartite:
        left_nodes = np.where(color == 0)[0]
        right_nodes = np.where(color == 1)[0]
        n_left = len(left_nodes)
        n_right = len(right_nodes)

        # Build adjacency: for each left node, which right nodes it connects to
        left_to_right = []
        for i in range(n_left):
            u = left_nodes[i]
            nbrs = np.where(w4_conf[u] == 1)[0]
            right_nbrs = []
            for v in nbrs:
                if color[v] == 1:
                    idx = np.where(right_nodes == v)[0]
                    if len(idx) > 0:
                        right_nbrs.append(idx[0])
            left_to_right.append(right_nbrs)

        # Hopcroft-Karp
        def hopcroft_karp(adj, n_l, n_r):
            pair_u = [-1] * n_l
            pair_v = [-1] * n_r
            dist = [0] * n_l

            def bfs():
                q = deque()
                for u in range(n_l):
                    if pair_u[u] == -1:
                        dist[u] = 0
                        q.append(u)
                    else:
                        dist[u] = -1
                found = False
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        pu = pair_v[v]
                        if pu == -1:
                            found = True
                        elif dist[pu] == -1:
                            dist[pu] = dist[u] + 1
                            q.append(pu)
                return found

            def dfs(u):
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                        pair_u[u] = v
                        pair_v[v] = u
                        return True
                dist[u] = -1
                return False

            matching = 0
            while bfs():
                for u in range(n_l):
                    if pair_u[u] == -1:
                        if dfs(u):
                            matching += 1
            return matching, pair_u, pair_v

        matching, pair_u, pair_v = hopcroft_karp(left_to_right, n_left, n_right)

        # Find Z: vertices reachable from unmatched left vertices via alternating paths
        visited_left = [False] * n_left
        visited_right = [False] * n_right
        q = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                visited_left[u] = True
                q.append(('L', u))
        while q:
            typ, idx = q.popleft()
            if typ == 'L':
                for v in left_to_right[idx]:
                    if not visited_right[v]:
                        visited_right[v] = True
                        q.append(('R', v))
            else:
                if pair_v[idx] != -1:
                    u = pair_v[idx]
                    if not visited_left[u]:
                        visited_left[u] = True
                        q.append(('L', u))

        # Independent set = (Left ∩ Z) U (Right \ Z)
        indep_left = []
        for u in range(n_left):
            if visited_left[u]:
                indep_left.append(left_nodes[u])
        indep_right = []
        for v in range(n_right):
            if not visited_right[v]:
                indep_right.append(right_nodes[v])

        w4_indep_set = set(indep_left + indep_right)
        w4_indep_size = len(w4_indep_set)

    # --- Phase 2: Simulated Annealing Local Search ---
    rng = np.random.default_rng(42)

    def sa_local_search(in_set_init, max_time=8.0, seed=42):
        """Simulated annealing local search with add/remove/swap moves."""
        rng_local = np.random.default_rng(seed)
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())

        T = 5.0
        cooling = 0.9995
        min_T = 0.05

        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(n_total, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1

        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        step = 0

        while time.time() < sa_deadline:
            step += 1
            not_in_set = np.where(~in_set)[0]
            in_set_indices = np.where(in_set)[0]

            if len(not_in_set) == 0 and len(in_set_indices) == 0:
                break

            move_type = rng_local.random()

            if move_type < 0.5 and len(not_in_set) > 0:
                # ADD move: try to add a vertex, removing any conflicts
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng_local.integers(len(min_cands))]

                k = min_count
                delta = 1 - k

                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng_local.random() < 0.3
                else:
                    accept = rng_local.random() < np.exp(delta / T)

                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1

                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.7 and len(in_set_indices) > 0:
                # REMOVE move: remove a random vertex to escape local optima
                v = in_set_indices[rng_local.integers(len(in_set_indices))]
                delta = -1
                accept = rng_local.random() < np.exp(delta / T)

                if accept:
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                    stuck_counter += 1
                else:
                    stuck_counter += 1

            elif move_type < 0.85 and len(in_set_indices) > 0 and len(not_in_set) > 0:
                # SWAP move: remove one in-set vertex, add one not-in-set vertex
                v_remove = in_set_indices[rng_local.integers(len(in_set_indices))]
                v_add_candidates = not_in_set[count_conflicts[not_in_set] <= 1]
                if len(v_add_candidates) == 0:
                    v_add_candidates = not_in_set
                v_add = v_add_candidates[rng_local.integers(len(v_add_candidates))]

                conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                other_conflicts = conflicts_with[conflicts_with != v_remove]

                if len(other_conflicts) == 0:
                    # Clean swap
                    in_set[v_remove] = False
                    in_set[v_add] = True
                    count_conflicts[neighbors[v_remove]] -= 1
                    count_conflicts[neighbors[v_add]] += 1
                    stuck_counter += 1
                else:
                    # Complex swap: remove v_remove and all other conflicts, add v_add
                    to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                    delta = 1 - len(to_remove)
                    accept = delta > 0 or (rng_local.random() < np.exp(delta / T))
                    if accept:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size += delta
                        count_conflicts[neighbors[v_add]] += 1

                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                    else:
                        stuck_counter += 1

            else:
                # 2-for-1 move: remove 2 vertices, add 1
                if len(in_set_indices) >= 2 and len(not_in_set) > 0:
                    v1 = in_set_indices[rng_local.integers(len(in_set_indices))]
                    v2_idx = np.delete(in_set_indices, np.where(in_set_indices == v1)[0][0])
                    v2 = v2_idx[rng_local.integers(len(v2_idx))]
                    v_add = not_in_set[rng_local.integers(len(not_in_set))]

                    to_remove = np.array([v1, v2])
                    other_conflicts = np.where(conflicts[v_add] & in_set)[0]
                    other_conflicts = other_conflicts[~np.isin(other_conflicts, to_remove)]

                    if len(other_conflicts) == 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        in_set[v_add] = True
                        current_size -= 1
                        count_conflicts[neighbors[v_add]] += 1
                        stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1

            T *= cooling

            # If stuck for too long, perturb
            if stuck_counter > 50:
                # Remove 2-3 random vertices
                if len(in_set_indices) >= 2:
                    n_remove = min(3, len(in_set_indices))
                    to_remove_idx = rng_local.choice(len(in_set_indices), size=n_remove, replace=False)
                    for idx in to_remove_idx:
                        v = in_set_indices[idx]
                        in_set[v] = False
                        count_conflicts[neighbors[v]] -= 1
                        current_size -= 1
                    stuck_counter = 0

        # Final polish: add all vertices with 0 conflicts
        not_in_set = np.where(~local_best_in_set)[0]
        for v in not_in_set:
            if not np.any(conflicts[v] & local_best_in_set):
                local_best_in_set[v] = True
                local_best_size += 1

        return local_best_in_set, local_best_size

    def greedy_init(seed=42):
        """Greedy min-degree initialization."""
        rng_local = np.random.default_rng(seed)
        in_set = np.zeros(n_total, dtype=bool)
        alive = np.ones(n_total, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]

            if len(cands) > 1:
                # Tie-break: prefer vertex whose neighbors have lower average degree
                scores = np.array([
                    np.mean(degrees[neighbors[c]]) if len(neighbors[c]) > 0 else 0
                    for c in cands
                ])
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

    # --- Main search loop ---
    best_in_set = np.zeros(n_total, dtype=bool)
    best_size = 0

    # Start with the bipartite solution as initial best
    if w4_indep_size > 0:
        init_in_set = np.zeros(n_total, dtype=bool)
        for v in w4_indep_set:
            init_in_set[v] = True
        # Try to add W1 vectors
        current = init_in_set.copy()
        current_size = w4_indep_size
        for v in range(n4, n_total):
            if not np.any(conflicts[v] & current):
                current[v] = True
                current_size += 1
        if current_size > best_size:
            best_size = current_size
            best_in_set = current

    # Multi-start local search
    n_restarts = 0
    max_restarts = 500

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1

        # Greedy initialization
        init_in_set = greedy_init(seed=42 + n_restarts)
        init_size = int(init_in_set.sum())

        if init_size > best_size:
            best_size = init_size
            best_in_set = init_in_set

        # Local search
        remaining = deadline - 8 - time.time()
        if remaining < 1.0:
            break

        ls_time = min(remaining * 0.3, 2.5)
        if ls_time < 0.3:
            ls_time = 0.3

        ls_result, ls_size = sa_local_search(init_in_set, max_time=ls_time, seed=42 + n_restarts)

        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result

        if time.time() > deadline - 10:
            break

    # Also try local search starting from the bipartite solution
    if w4_indep_size > 0 and time.time() < deadline - 5:
        init_in_set = np.zeros(n_total, dtype=bool)
        for v in w4_indep_set:
            init_in_set[v] = True
        # Add W1 vectors greedily
        current = init_in_set.copy()
        for v in range(n4, n_total):
            if not np.any(conflicts[v] & current):
                current[v] = True
        ls_result, ls_size = sa_local_search(current, max_time=min(2.0, deadline - 5 - time.time()), seed=999)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result

    # Convert best_in_set to index array
    final_indices = np.where(best_in_set)[0]

    if len(final_indices) == 0:
        return np.array([], dtype=np.int64)

    result_vectors = all_vectors[final_indices]
    return result_vectors
