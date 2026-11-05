import random
import time
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    d = 7
    n = 2 ** d  # 128 vertices

    # Precompute all edges: (u, v, dim) where u < v and dim is the bit that differs
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))

    total_edges = len(all_edges)  # 448

    # Build a mapping from edge tuple to index
    edge_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_to_idx[(u, v)] = idx

    # For each edge, precompute the "forbidden triples" - the other 3 edges that would complete a C4
    edge_to_forbidden = {}
    for idx, (u, v, i) in enumerate(all_edges):
        forbidden = []
        for j in range(d):
            if j != i:
                c = u ^ (1 << j)
                e = v ^ (1 << j)
                e1 = (min(u, c), max(u, c))
                e2 = (min(v, e), max(v, e))
                e3 = (min(c, e), max(c, e))
                forbidden.append((e1, e2, e3))
        edge_to_forbidden[idx] = forbidden

    # Convert forbidden to index lists for faster lookup
    edge_to_forbidden_idx = {}
    for idx in range(total_edges):
        forbidden = []
        for (e1, e2, e3) in edge_to_forbidden[idx]:
            i1 = edge_to_idx.get(e1)
            i2 = edge_to_idx.get(e2)
            i3 = edge_to_idx.get(e3)
            if i1 is not None and i2 is not None and i3 is not None:
                forbidden.append((i1, i2, i3))
        edge_to_forbidden_idx[idx] = forbidden

    # numpy array of forbidden triples: shape (total_edges, 6, 3) with -1 padding
    max_triples = max(len(v) for v in edge_to_forbidden_idx.values())
    forb_arr = np.full((total_edges, max_triples, 3), -1, dtype=np.int64)
    for idx in range(total_edges):
        triples = edge_to_forbidden_idx[idx]
        for t in range(len(triples)):
            forb_arr[idx, t] = triples[t]

    def is_addable(idx, chosen_arr):
        """Check if edge idx can be added (numpy bool array) without creating a C4."""
        triples = forb_arr[idx]
        # mask out padded rows
        m = triples >= 0
        cols = triples[m]
        if cols.size == 0:
            return True
        # For each triple, are all 3 chosen?
        sel = chosen_arr[cols]  # shape (k,3)
        return not np.all(sel, axis=1).any()

    def blocking_count(idx, chosen_arr):
        """Number of forbidden triples of idx that are fully chosen."""
        triples = forb_arr[idx]
        m = triples >= 0
        cols = triples[m]
        if cols.size == 0:
            return 0
        sel = chosen_arr[cols]
        return int(np.all(sel, axis=1).sum())

    def greedy_order(order, chosen_arr=None):
        """Run greedy over a fixed order of edge indices."""
        if chosen_arr is None:
            chosen_arr = np.zeros(total_edges, dtype=bool)
        for idx in order:
            if not chosen_arr[idx] and is_addable(idx, chosen_arr):
                chosen_arr[idx] = True
        return chosen_arr

    def greedy_mcs(rng, chosen_arr=None):
        """MCS-style: iteratively pick the least-constrained addable edge
        (fewest fully-chosen forbidden triples), breaking ties randomly."""
        if chosen_arr is None:
            chosen_arr = np.zeros(total_edges, dtype=bool)
        remaining = np.ones(total_edges, dtype=bool)
        for _ in range(total_edges):
            if not remaining.any():
                break
            rem_idx = np.where(remaining)[0]
            # compute constraint score for each remaining edge
            best = None
            best_score = None
            # sample to keep it fast if many remain
            if len(rem_idx) > 200:
                sample = rem_idx[rng.choice(len(rem_idx), size=200, replace=False)]
            else:
                sample = rem_idx
            for idx in sample:
                if chosen_arr[idx]:
                    continue
                if not is_addable(idx, chosen_arr):
                    continue
                sc = blocking_count(idx, chosen_arr)
                if best_score is None or sc < best_score or (sc == best_score and rng.random() < 0.5):
                    best_score = sc
                    best = int(idx)
            if best is None:
                # no addable in sample; try full remaining
                for idx in rem_idx:
                    if not chosen_arr[idx] and is_addable(idx, chosen_arr):
                        best = int(idx)
                        break
            if best is None:
                break
            chosen_arr[best] = True
            remaining[best] = False
        return chosen_arr

    def local_search(chosen_arr, max_iters=10000, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        chosen_arr = chosen_arr.copy()
        current_size = int(chosen_arr.sum())
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            unchosen = np.where(~chosen_arr)[0]
            if len(unchosen) == 0:
                break
            idx = int(unchosen[rng.randint(0, len(unchosen) - 1)])
            if is_addable(idx, chosen_arr):
                chosen_arr[idx] = True
                current_size += 1
            else:
                blocking = []
                for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                    if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                        blocking.append((i1, i2, i3))
                if not blocking:
                    chosen_arr[idx] = True
                    current_size += 1
                else:
                    triple = blocking[rng.randint(0, len(blocking) - 1)]
                    for rem_idx in triple:
                        chosen_arr[rem_idx] = False
                        if is_addable(idx, chosen_arr):
                            chosen_arr[idx] = True
                            break
                        else:
                            chosen_arr[rem_idx] = True
        return chosen_arr

    def simulated_annealing(chosen_arr, max_iters=20000, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        chosen_arr = chosen_arr.copy()
        current_size = int(chosen_arr.sum())
        best_arr = chosen_arr.copy()
        best_size = current_size
        T_start = 5.0
        T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iters))
        T = T_start
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            T *= cooling_rate
            unchosen = np.where(~chosen_arr)[0]
            chosen_indices = np.where(chosen_arr)[0]
            if len(unchosen) == 0:
                break
            if rng.random() < 0.7 and len(unchosen) > 0:
                idx = int(unchosen[rng.randint(0, len(unchosen) - 1)])
                if is_addable(idx, chosen_arr):
                    chosen_arr[idx] = True
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_arr = chosen_arr.copy()
                else:
                    blocking = []
                    for (i1, i2, i3) in edge_to_forbidden_idx[idx]:
                        if chosen_arr[i1] and chosen_arr[i2] and chosen_arr[i3]:
                            blocking.append((i1, i2, i3))
                    if blocking:
                        triple = blocking[rng.randint(0, len(blocking) - 1)]
                        rem_idx = triple[rng.randint(0, 2)]
                        chosen_arr[rem_idx] = False
                        if is_addable(idx, chosen_arr):
                            chosen_arr[idx] = True
                        else:
                            chosen_arr[rem_idx] = True
            else:
                if len(chosen_indices) > 0:
                    rem_idx = int(chosen_indices[rng.randint(0, len(chosen_indices) - 1)])
                    old_size = current_size
                    chosen_arr[rem_idx] = False
                    current_size -= 1
                    accept_prob = np.exp(1.0 / T) if T > 1e-6 else 0.0
                    if rng.random() < accept_prob:
                        if current_size > best_size:
                            best_size = current_size
                            best_arr = chosen_arr.copy()
                    else:
                        chosen_arr[rem_idx] = True
                        current_size = old_size
        return best_arr

    best_solution = None
    best_size = 0

    def consider(sol):
        nonlocal best_solution, best_size
        s = int(sol.sum())
        if s > best_size:
            best_size = s
            best_solution = sol

    # Phase 1a: structured initial orderings (measured improvement)
    if time.time() - start_time < time_limit - 40:
        # order by dimension
        by_dim = sorted(range(total_edges), key=lambda i: (all_edges[i][2], all_edges[i][0]))
        consider(greedy_order(by_dim))
        by_dim_rev = list(reversed(by_dim))
        consider(greedy_order(by_dim_rev))
        # order by endpoint then dim
        by_ep = sorted(range(total_edges), key=lambda i: (all_edges[i][0], all_edges[i][2]))
        consider(greedy_order(by_ep))
        by_ep_rev = list(reversed(by_ep))
        consider(greedy_order(by_ep_rev))

    # Phase 1b: randomized greedy restarts (random shuffle)
    num_restarts = 0
    while time.time() - start_time < time_limit - 40:
        rng = random.Random(num_restarts)
        order = list(range(total_edges))
        rng.shuffle(order)
        consider(greedy_order(order))
        num_restarts += 1

    # Phase 1c: MCS greedy restarts (least-constrained-first global ordering)
    num_mcs = 0
    while time.time() - start_time < time_limit - 40:
        rng = random.Random(100000 + num_mcs)
        consider(greedy_mcs(rng))
        num_mcs += 1

    # Phase 2: local search on best solution
    if best_solution is not None and time.time() - start_time < time_limit - 25:
        improved = local_search(best_solution, max_iters=15000, seed=42)
        consider(improved)

    # Phase 3: Simulated Annealing
    if best_solution is not None and time.time() - start_time < time_limit - 15:
        improved = simulated_annealing(best_solution, max_iters=25000, seed=789)
        consider(improved)

    # Phase 4: more local search + SA iterations if time permits
    if best_solution is not None and time.time() - start_time < time_limit - 5:
        for _ in range(6):
            if time.time() - start_time > time_limit - 5:
                break
            improved = local_search(best_solution, max_iters=8000, seed=random.randint(0, 100000))
            consider(improved)
            if time.time() - start_time > time_limit - 5:
                break
            improved = simulated_annealing(best_solution, max_iters=10000, seed=random.randint(0, 100000))
            consider(improved)
            if time.time() - start_time > time_limit - 5:
                break

    if best_solution is None:
        best_solution = greedy_order(list(range(total_edges)))

    # Convert to output format
    result = []
    for idx in np.where(best_solution)[0]:
        u, v, i = all_edges[idx]
        result.append([u, v])

    return result
