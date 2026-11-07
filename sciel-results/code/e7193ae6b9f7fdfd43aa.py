import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # safety margin for 90s kill

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def compute_score(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods)), sums, prods

    def marginal_cost(values, edge, sums_set, prods_set):
        """Return (new_sums_count, new_prods_count) if edge added."""
        u, v = edge
        s = values[u] + values[v]
        p = values[u] * values[v]
        ns = 0 if s in sums_set else 1
        np_ = 0 if p in prods_set else 1
        return ns, np_

    def greedy_edge_selection(values, max_time=2.0):
        """Greedy: add edges one by one choosing the one with lowest marginal cost."""
        t0 = time.time()
        # Start with empty edge set
        edges = []
        edge_set = set()
        sums_set = set()
        prods_set = set()

        # Precompute all edge costs dynamically
        for _ in range(m):
            best_edge = None
            best_cost = (float('inf'), float('inf'))
            for e in all_edges:
                if e in edge_set:
                    continue
                ns, np_ = marginal_cost(values, e, sums_set, prods_set)
                cost = (ns + np_, ns, np_)
                if cost < best_cost:
                    best_cost = cost
                    best_edge = e

            if best_edge is None:
                break

            u, v = best_edge
            sums_set.add(values[u] + values[v])
            prods_set.add(values[u] * values[v])
            edges.append(best_edge)
            edge_set.add(best_edge)

            if time.time() - t0 > max_time:
                break

        # If we didn't get all m edges, fill randomly
        while len(edges) < m:
            e = random.choice(all_edges)
            if e not in edge_set:
                edges.append(e)
                edge_set.add(e)
                sums_set.add(values[edges[-1][0]] + values[edges[-1][1]])
                prods_set.add(values[edges[-1][0]] * values[edges[-1][1]])

        return edges

    def edge_exchange_local_search(values, edges, max_iters=5000, max_time=5.0):
        """Try swapping one edge for a non-edge if it reduces score."""
        t0 = time.time()
        current_edges = list(edges)
        current_score, current_sums, current_prods = compute_score(values, current_edges)
        best_score = current_score
        best_edges = list(current_edges)
        edge_set = set(current_edges)

        for it in range(max_iters):
            if time.time() - t0 > max_time:
                break

            # Pick a random edge to remove
            e_out_idx = random.randint(0, m - 1)
            e_out = current_edges[e_out_idx]

            # Remove it temporarily
            edge_set.discard(e_out)
            # Recompute sums/prods without this edge
            sums_set = set()
            prods_set = set()
            for e in current_edges:
                if e == e_out:
                    continue
                sums_set.add(values[e[0]] + values[e[1]])
                prods_set.add(values[e[0]] * values[e[1]])

            # Try to find a non-edge that adds minimal new values
            best_new_edge = None
            best_new_score = current_score

            # Sample a subset of non-edges for speed
            candidates = []
            for e in all_edges:
                if e not in edge_set:
                    candidates.append(e)

            # If too many, sample
            if len(candidates) > 50:
                candidates = random.sample(candidates, 50)

            for e_in in candidates:
                u, v = e_in
                s = values[u] + values[v]
                p = values[u] * values[v]
                new_sums_count = len(sums_set) + (0 if s in sums_set else 1)
                new_prods_count = len(prods_set) + (0 if p in prods_set else 1)
                new_score = max(new_sums_count, new_prods_count)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_edge = e_in

            if best_new_edge is not None:
                # Perform the swap
                current_edges[e_out_idx] = best_new_edge
                edge_set.discard(e_out)
                edge_set.add(best_new_edge)

                current_score, current_sums, current_prods = compute_score(values, current_edges)
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(current_edges)

        return best_score, best_edges

    def value_local_search(values, edges, max_iters=3000, max_time=3.0):
        """Perturb values to create more collisions."""
        t0 = time.time()
        current_values = list(values)
        current_score, _, _ = compute_score(current_values, edges)
        best_score = current_score
        best_values = list(current_values)
        edge_list = list(edges)

        for it in range(max_iters):
            if time.time() - t0 > max_time:
                break

            idx = random.randint(0, n - 1)
            # Try a few candidate values
            for attempt in range(5):
                if random.random() < 0.5:
                    # Swap with another index
                    idx2 = random.randint(0, n - 1)
                    if idx2 == idx:
                        continue
                    new_values = list(current_values)
                    new_values[idx], new_values[idx2] = new_values[idx2], new_values[idx]
                else:
                    # Replace with a small value from a pool
                    pool = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]
                    new_val = random.choice(pool)
                    if new_val in current_values:
                        continue
                    new_values = list(current_values)
                    new_values[idx] = new_val

                new_score, _, _ = compute_score(new_values, edge_list)
                if new_score <= current_score:
                    current_values = new_values
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_values = list(current_values)
                    break  # Accept and move on

        return best_score, best_values

    # --- Main search ---
    random.seed(42)

    best_global_score = float('inf')
    best_global_values = None
    best_global_edges = None

    # Generate diverse value pools emphasizing small numbers and multiplicative structure
    value_pools = [
        # Pool 1: Very small numbers
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        # Pool 2: Multiplicative structure (many divisors)
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        # Pool 3: Mix of small and medium
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40],
        # Pool 4: Heavily weighted toward 1-5
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Will deduplicate
        # Pool 5: Arithmetic progression
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        # Pool 6: Random small
        random.sample(range(1, 31), n),
        # Pool 7: Random with small bias
        sorted([random.randint(1, 20) for _ in range(n)]),
        # Pool 8: Another multiplicative
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
    ]

    # Deduplicate pools and ensure 14 distinct positive integers <= 10000
    valid_pools = []
    for pool in value_pools:
        # Ensure distinct and valid
        seen = set()
        deduped = []
        for v in pool:
            if v not in seen and 1 <= v <= 10000:
                seen.add(v)
                deduped.append(v)
        # If not enough, fill
        if len(deduped) < n:
            extra = set(range(1, 101)) - seen
            for v in extra:
                if len(deduped) >= n:
                    break
                deduped.append(v)
        if len(deduped) == n:
            valid_pools.append(deduped[:n])

    # Shuffle each pool
    for pool in valid_pools:
        random.shuffle(pool)

    # Run search on each pool
    time_per_pool = (deadline - time.time()) / len(valid_pools) if valid_pools else 10.0

    for pool in valid_pools:
        if time.time() > deadline:
            break

        t_pool_start = time.time()
        time_budget = min(time_per_pool, deadline - time.time())

        # Step 1: Greedy edge selection
        t0 = time.time()
        edges = greedy_edge_selection(pool, max_time=min(1.0, time_budget * 0.3))
        t_greedy = time.time() - t0

        # Step 2: Edge exchange local search
        t0 = time.time()
        max_es_iters = int((time_budget * 0.4) * 200)
        max_es_iters = max(max_es_iters, 500)
        score, edges = edge_exchange_local_search(pool, edges, max_iters=max_es_iters,
                                                   max_time=min(3.0, time_budget * 0.4))
        t_es = time.time() - t0

        # Step 3: Value local search
        t0 = time.time()
        max_vl_iters = int((time_budget * 0.3) * 300)
        max_vl_iters = max(max_vl_iters, 300)
        score, values = value_local_search(edges, edges, max_iters=max_vl_iters,
                                            max_time=min(2.0, time_budget * 0.3))
        t_vl = time.time() - t0

        # Step 4: Another round of edge exchange with updated values
        if time.time() < deadline - 1.0:
            remaining = deadline - time.time() - 1.0
            max_es2_iters = int(remaining * 0.5 * 200)
            max_es2_iters = max(max_es2_iters, 200)
            score2, edges2 = edge_exchange_local_search(values, edges, max_iters=max_es2_iters,
                                                         max_time=min(2.0, remaining * 0.5))
            if score2 < score:
                score = score2
                edges = edges2

        if score < best_global_score:
            best_global_score = score
            best_global_values = list(values)
            best_global_edges = list(edges)

    # Final polish if time remains
    if best_global_values is not None and time.time() < deadline - 1.0:
        remaining = deadline - time.time() - 1.0
        max_iters = int(remaining * 300)
        max_iters = max(max_iters, 500)
        score, edges = edge_exchange_local_search(best_global_values, best_global_edges,
                                                   max_iters=max_iters,
                                                   max_time=min(3.0, remaining * 0.6))
        if score < best_global_score:
            best_global_score = score
            best_global_edges = edges

        if time.time() < deadline - 0.5:
            remaining2 = deadline - time.time() - 0.5
            max_vl2 = int(remaining2 * 400)
            max_vl2 = max(max_vl2, 200)
            score2, values2 = value_local_search(best_global_values, best_global_edges,
                                                  max_iters=max_vl2,
                                                  max_time=min(2.0, remaining2 * 0.8))
            if score2 < best_global_score:
                best_global_score = score2
                best_global_values = values2

    # Fallback
    if best_global_values is None:
        best_global_values = list(range(1, 15))
        best_global_edges = list(all_edges[:35])
        best_global_score, _, _ = compute_score(best_global_values, best_global_edges)

    # Ensure types: Python int, list
    final_values = [int(v) for v in best_global_values]
    final_edges = [[int(u), int(v)] for u, v in best_global_edges]

    return {
        'values': final_values,
        'edges': final_edges
    }
