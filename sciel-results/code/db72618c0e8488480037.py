import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)

    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)

    # Precompute conflict and adjacency
    conflict = np.zeros((N_S, N_S), dtype=bool)
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                conflict[i, j] = True
                conflict[j, i] = True
                adj[i].add(j)
                adj[j].add(i)

    adj_arrays = [np.array(list(adj[i]), dtype=np.int64) for i in range(N_S)]
    adj_counts = np.array([len(adj[i]) for i in range(N_S)], dtype=np.int64)

    # Precompute coordinate incidence: for each coordinate, which supports contain it
    coord_to_supports = [[] for _ in range(D)]
    for i, s in enumerate(supports):
        for c in s:
            coord_to_supports[c].append(i)
    coord_to_supports = [np.array(lst, dtype=np.int64) for lst in coord_to_supports]

    rng = np.random.RandomState(42)

    def compute_blocked(x):
        blocked = np.zeros(N_S, dtype=bool)
        in_set = np.where(x)[0]
        for v in in_set:
            neighbors = adj_arrays[v]
            if len(neighbors) > 0:
                blocked[neighbors] = True
        return blocked

    def greedy_fill(x, blocked):
        changed = True
        while changed:
            changed = False
            candidates = np.where(~x & ~blocked)[0]
            for v in candidates:
                if not x[v] and not blocked[v]:
                    x[v] = True
                    neighbors = adj_arrays[v]
                    if len(neighbors) > 0:
                        blocked[neighbors] = True
                    changed = True
        return x, blocked

    def ilp_cutting_plane(x_init, time_limit):
        """
        ILP-inspired heuristic: maximize sum(x) subject to
        - x is binary
        - for each pair (i,j) with conflict: x[i] + x[j] <= 1
        
        Use a simple LP relaxation with cutting planes.
        Since scipy.optimize.linprog is available, we use it.
        """
        from scipy.optimize import linprog

        # Objective: maximize sum(x) => minimize -sum(x)
        c_obj = -np.ones(N_S)

        # Constraints: x[i] + x[j] <= 1 for all conflicting pairs
        # Build constraint matrix
        rows = []
        for i in range(N_S):
            for j in adj_arrays[i]:
                if j > i:  # avoid duplicates
                    row = np.zeros(N_S)
                    row[i] = 1.0
                    row[j] = 1.0
                    rows.append(row)

        if len(rows) == 0:
            A_ub = np.zeros((0, N_S))
            b_ub = np.zeros(0)
        else:
            A_ub = np.array(rows)
            b_ub = np.ones(len(rows))

        bounds = [(0, 1)] * N_S

        try:
            result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.success:
                x_lp = result.x
                # Round with a greedy approach guided by LP solution
                x_rounded = np.zeros(N_S, dtype=bool)
                # Sort by LP value descending, greedily add if no conflict
                order = np.argsort(-x_lp)
                blocked = np.zeros(N_S, dtype=bool)
                for v in order:
                    if not x_rounded[v] and not blocked[v]:
                        x_rounded[v] = True
                        neighbors = adj_arrays[v]
                        if len(neighbors) > 0:
                            blocked[neighbors] = True
                # Greedy fill
                x_rounded, blocked = greedy_fill(x_rounded, blocked)
                return x_rounded
        except Exception:
            pass

        return x_init

    def local_search(x, max_iter=20000, time_limit=None):
        current_x = x.copy()
        blocked = compute_blocked(current_x)
        current_x, blocked = greedy_fill(current_x, blocked)
        current_score = int(np.sum(current_x))

        best_x = current_x.copy()
        best_score = current_score

        T = 3.0
        cooling = 0.9997

        for it in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break

            r = rng.random()

            if r < 0.25:
                in_set = np.where(current_x)[0]
                if len(in_set) < 1:
                    continue
                n_remove = 1 + rng.randint(0, 2)
                n_remove = min(n_remove, len(in_set))
                to_remove = in_set[rng.choice(len(in_set), size=n_remove, replace=False)]
                for v in to_remove:
                    current_x[v] = False
                blocked = compute_blocked(current_x)
                current_x, blocked = greedy_fill(current_x, blocked)
                current_score = int(np.sum(current_x))
                if current_score > best_score:
                    best_score = current_score
                    best_x = current_x.copy()

            elif r < 0.55:
                in_set = np.where(current_x)[0]
                if len(in_set) == 0:
                    continue
                u = in_set[rng.randint(len(in_set))]
                temp_x = current_x.copy()
                temp_x[u] = False
                temp_blocked = compute_blocked(temp_x)
                candidates = np.where(~current_x & ~temp_blocked)[0]
                if len(candidates) > 0:
                    v = candidates[rng.randint(len(candidates))]
                    current_x[u] = False
                    current_x[v] = True
                    blocked = compute_blocked(current_x)
                    current_x, blocked = greedy_fill(current_x, blocked)
                    current_score = int(np.sum(current_x))
                    if current_score > best_score:
                        best_score = current_score
                        best_x = current_x.copy()

            elif r < 0.80:
                in_set = np.where(current_x)[0]
                if len(in_set) < 2:
                    continue
                u1_idx, u2_idx = rng.choice(len(in_set), size=2, replace=False)
                u1, u2 = in_set[u1_idx], in_set[u2_idx]
                temp_x = current_x.copy()
                temp_x[u1] = False
                temp_x[u2] = False
                temp_blocked = compute_blocked(temp_x)
                candidates = np.where(~current_x & ~temp_blocked)[0]
                if len(candidates) >= 2:
                    v1_idx, v2_idx = rng.choice(len(candidates), size=2, replace=False)
                    v1, v2 = candidates[v1_idx], candidates[v2_idx]
                    if not conflict[v1, v2]:
                        current_x[u1] = False
                        current_x[u2] = False
                        current_x[v1] = True
                        current_x[v2] = True
                        blocked = compute_blocked(current_x)
                        current_x, blocked = greedy_fill(current_x, blocked)
                        current_score = int(np.sum(current_x))
                        if current_score > best_score:
                            best_score = current_score
                            best_x = current_x.copy()
            else:
                in_set = np.where(current_x)[0]
                if len(in_set) < 3:
                    continue
                n_remove = 3 + rng.randint(0, 2)
                n_remove = min(n_remove, len(in_set))
                to_remove = in_set[rng.choice(len(in_set), size=n_remove, replace=False)]
                for v in to_remove:
                    current_x[v] = False
                blocked = compute_blocked(current_x)
                current_x, blocked = greedy_fill(current_x, blocked)
                current_score = int(np.sum(current_x))
                if current_score > best_score:
                    best_score = current_score
                    best_x = current_x.copy()

            T *= cooling

        return best_x

    best_support_set = set()
    best_size = 0

    # First, try ILP cutting plane approach
    if time.time() < deadline - 30.0:
        x_ilp = ilp_cutting_plane(np.zeros(N_S, dtype=bool), deadline - 25.0)
        size_ilp = int(np.sum(x_ilp))
        if size_ilp > best_size:
            best_size = size_ilp
            best_support_set = set(np.where(x_ilp)[0].tolist())

    num_restarts = 80
    for r in range(num_restarts):
        if time.time() > deadline - 8.0:
            break

        if r < 20:
            x0 = np.zeros(N_S, dtype=bool)
            order = rng.permutation(N_S)
            blocked = np.zeros(N_S, dtype=bool)
            for v in order:
                if not x0[v] and not blocked[v]:
                    x0[v] = True
                    neighbors = adj_arrays[v]
                    if len(neighbors) > 0:
                        blocked[neighbors] = True
        elif r < 40:
            x0 = np.zeros(N_S, dtype=bool)
            degrees = adj_counts.copy()
            jittered = degrees + rng.randint(0, 10, size=N_S)
            order = np.argsort(jittered)
            blocked = np.zeros(N_S, dtype=bool)
            for v in order:
                if not x0[v] and not blocked[v]:
                    x0[v] = True
                    neighbors = adj_arrays[v]
                    if len(neighbors) > 0:
                        blocked[neighbors] = True
        elif r < 60:
            x0 = np.zeros(N_S, dtype=bool)
            n_random = rng.randint(5, 20)
            random_nodes = rng.choice(N_S, size=n_random, replace=False)
            blocked = np.zeros(N_S, dtype=bool)
            for v in random_nodes:
                if not blocked[v]:
                    x0[v] = True
                    neighbors = adj_arrays[v]
                    if len(neighbors) > 0:
                        blocked[neighbors] = True
            x0, blocked = greedy_fill(x0, blocked)
        else:
            if best_size > 0:
                x0 = np.zeros(N_S, dtype=bool)
                best_nodes = list(best_support_set)
                for v in best_nodes:
                    x0[v] = True
                n_remove = rng.randint(2, 5)
                to_remove = rng.choice(len(best_nodes), size=min(n_remove, len(best_nodes)), replace=False)
                for idx in to_remove:
                    x0[best_nodes[idx]] = False
                blocked = compute_blocked(x0)
                x0, blocked = greedy_fill(x0, blocked)
            else:
                x0 = np.zeros(N_S, dtype=bool)
                order = rng.permutation(N_S)
                blocked = np.zeros(N_S, dtype=bool)
                for v in order:
                    if not x0[v] and not blocked[v]:
                        x0[v] = True
                        neighbors = adj_arrays[v]
                        if len(neighbors) > 0:
                            blocked[neighbors] = True

        remaining_time = deadline - time.time()
        max_iter = min(20000, max(5000, int(remaining_time * 300)))

        x_best = local_search(x0, max_iter=max_iter, time_limit=deadline - 4.0)
        size = int(np.sum(x_best))

        if size > best_size:
            best_size = size
            best_support_set = set(np.where(x_best)[0].tolist())

    if time.time() < deadline - 2.0 and best_support_set:
        polish_x = np.zeros(N_S, dtype=bool)
        for v in best_support_set:
            polish_x[v] = True
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            blocked = compute_blocked(polish_x)
            candidates = np.where(~polish_x & ~blocked)[0]
            for v in candidates:
                if not polish_x[v] and not blocked[v]:
                    polish_x[v] = True
                    changed = True
        final_size = int(np.sum(polish_x))
        if final_size > best_size:
            best_size = final_size
            best_support_set = set(np.where(polish_x)[0].tolist())

    vectors = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors.append(v2)

    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)

    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)

    return np.array(vectors, dtype=np.int8)
