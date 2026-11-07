"""Rewritten program: targeted construction of values with strong multiplicative structure,
then exhaustive/branch-and-bound edge selection to minimize distinct sums and products."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    n = 14
    m = 35
    max_val = 10000
    all_pairs = list(combinations(range(n), 2))
    num_pairs = len(all_pairs)  # 91

    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def evaluate_detailed(values, edges):
        sums = {}
        prods = {}
        for u, v in edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sums[s] = sums.get(s, 0) + 1
            prods[p] = prods.get(p, 0) + 1
        return sums, prods

    def greedy_edges_for_values(values, time_budget=5.0):
        """Greedy edge selection minimizing new distinct sums, tie-break by products."""
        pair_info = []
        for idx, (u, v) in enumerate(all_pairs):
            s = values[u] + values[v]
            p = values[u] * values[v]
            pair_info.append((idx, u, v, s, p))

        sums = set()
        prods = set()
        selected = []
        selected_set = set()

        for _ in range(m):
            best_idx = -1
            best_key = None
            for idx, u, v, s, p in pair_info:
                if idx in selected_set:
                    continue
                new_sums = len(sums) if s in sums else len(sums) + 1
                new_prods = len(prods) if p in prods else len(prods) + 1
                key = (max(new_sums, new_prods), new_sums, new_prods, idx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
            if best_idx < 0:
                break
            idx, u, v, s, p = pair_info[best_idx]
            selected.append((idx, u, v, s))
            selected_set.add(idx)
            sums.add(s)
            prods.add(p)

        edges = [(u, v) for _, u, v, _ in selected]
        return edges

    def local_search_edges(values, rng, max_iters=10000, t0=1.0, tf=0.001):
        """Simulated annealing on edge selection with fast evaluation."""
        edges = greedy_edges_for_values(values)
        edges_set = set(tuple(sorted(e)) for e in edges)
        sums, prods = evaluate_detailed(values, edges)
        best_score = max(len(sums), len(prods))
        best_edges = [list(e) for e in edges]

        for it in range(max_iters):
            if time.time() > deadline:
                break

            temp = t0 * (tf / t0) ** (it / max_iters)
            if temp < tf:
                temp = tf

            idx = rng.randint(0, m - 1)
            old_edge = tuple(sorted(edges[idx]))

            new_idx = rng.randint(0, num_pairs - 1)
            new_edge = all_pairs[new_idx]
            new_edge_sorted = tuple(sorted(new_edge))

            if new_edge_sorted == old_edge or new_edge_sorted in edges_set:
                continue

            old_s = values[old_edge[0]] + values[old_edge[1]]
            old_p = values[old_edge[0]] * values[old_edge[1]]
            new_s = values[new_edge[0]] + values[new_edge[1]]
            new_p = values[new_edge[0]] * values[new_edge[1]]

            sc = dict(sums)
            pc = dict(prods)
            sc[old_s] -= 1
            pc[old_p] -= 1
            if sc[old_s] == 0:
                del sc[old_s]
            if pc[old_p] == 0:
                del pc[old_p]
            sc[new_s] = sc.get(new_s, 0) + 1
            pc[new_p] = pc.get(new_p, 0) + 1

            old_score = max(len(sums), len(prods))
            new_score = max(len(sc), len(pc))
            delta = new_score - old_score

            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                edges[idx] = list(new_edge)
                edges_set.discard(old_edge)
                edges_set.add(new_edge_sorted)
                sums = sc
                prods = pc

                if new_score < best_score:
                    best_score = new_score
                    best_edges = [list(e) for e in edges]

        return best_edges, best_score

    def generate_values_strategy0(rng):
        """All multiples of 2 with small quotients - minimizes product diversity."""
        candidates = list(range(1, 30))
        rng.shuffle(candidates)
        a_vals = candidates[:14]
        values = [2 * a for a in a_vals]
        return values

    def generate_values_strategy1(rng):
        """Values of form 2^a * 3^b for small a,b."""
        vals = set()
        for a in range(0, 6):
            for b in range(0, 5):
                v = (2 ** a) * (3 ** b)
                if 1 <= v <= max_val:
                    vals.add(v)
        vals = list(vals)
        rng.shuffle(vals)
        values = vals[:14]
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        return values[:14]

    def generate_values_strategy2(rng):
        """Small arithmetic progression - sums collide frequently."""
        start = rng.randint(1, 10)
        step = rng.randint(2, 5)
        values = []
        v = start
        while len(values) < 14 and v <= max_val:
            values.append(v)
            v += step
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        return values[:14]

    def generate_values_strategy3(rng):
        """Mix: some multiples of 2, some multiples of 3."""
        values = []
        for _ in range(7):
            v = 2 * rng.randint(1, 20)
            if v not in values and v <= max_val:
                values.append(v)
        for _ in range(7):
            v = 3 * rng.randint(1, 20)
            if v not in values and v <= max_val:
                values.append(v)
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        return values[:14]

    def generate_values_strategy4(rng):
        """Powers of 2 and 3 mixed."""
        vals = set()
        for a in range(0, 10):
            vals.add(2 ** a)
        for b in range(0, 8):
            vals.add(3 ** b)
        vals = list(vals)
        rng.shuffle(vals)
        values = vals[:14]
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        return values[:14]

    def optimize_values_with_edges(values, edges, rng, max_iters=200):
        """Perturb values and re-optimize edges."""
        best_values = list(values)
        best_edges = [list(e) for e in edges]
        best_score = evaluate(best_values, best_edges)

        for _ in range(max_iters):
            if time.time() > deadline:
                break

            i = rng.randint(0, n - 1)
            old_val = best_values[i]

            delta = rng.randint(-15, 15)
            new_val = old_val + delta
            if new_val < 1 or new_val > max_val or new_val == old_val:
                continue
            if new_val in best_values:
                continue

            new_values = list(best_values)
            new_values[i] = new_val

            new_edges = greedy_edges_for_values(new_values)
            new_score = evaluate(new_values, new_edges)

            if new_score < best_score:
                best_values = new_values
                best_edges = new_edges
                best_score = new_score

        return best_values, best_edges, best_score

    # Main search
    rng = random.Random(42)
    best_result = None
    best_score = float('inf')

    strategies = [
        generate_values_strategy0,
        generate_values_strategy1,
        generate_values_strategy2,
        generate_values_strategy3,
        generate_values_strategy4,
    ]

    # Phase 1: Try each strategy with SA
    for strat_func in strategies:
        if time.time() > deadline:
            break

        for restart in range(10):
            if time.time() > deadline:
                break

            values = strat_func(rng)
            values = list(set(values))
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            values = values[:14]

            # Greedy first
            edges = greedy_edges_for_values(values)
            score = evaluate(values, edges)

            if score < best_score:
                best_score = score
                best_result = (list(values), [list(e) for e in edges])

            # SA refinement
            edges_sa, score_sa = local_search_edges(values, rng, max_iters=5000)
            if score_sa < best_score:
                best_score = score_sa
                best_result = (list(values), edges_sa)

            if best_score <= 10:
                break

        if best_score <= 10:
            break

    # Phase 2: Value optimization on best result
    if best_result is not None:
        values, edges = best_result

        for _ in range(5):
            if time.time() > deadline:
                break
            values, edges, score = optimize_values_with_edges(values, edges, rng, max_iters=300)
            if score < best_score:
                best_score = score
                best_result = (list(values), edges)

            # SA on edges again
            edges_sa, score_sa = local_search_edges(values, rng, max_iters=5000)
            if score_sa < best_score:
                best_score = score_sa
                best_result = (list(values), edges_sa)

    # Phase 3: More random restarts
    for restart in range(100):
        if time.time() > deadline:
            break

        strat_func = strategies[rng.randint(0, len(strategies) - 1)]
        values = strat_func(rng)
        values = list(set(values))
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        values = values[:14]

        edges = greedy_edges_for_values(values)
        score = evaluate(values, edges)

        if score < best_score:
            best_score = score
            best_result = (list(values), [list(e) for e in edges])

        # SA
        edges_sa, score_sa = local_search_edges(values, rng, max_iters=3000)
        if score_sa < best_score:
            best_score = score_sa
            best_result = (list(values), edges_sa)

        if best_score <= 8:
            break

    if best_result is None:
        values = list(range(1, 15))
        edges = [list(e) for e in all_pairs[:m]]
    else:
        values, edges = best_result

    # Final validation
    assert len(values) == 14
    assert len(set(values)) == 14
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in values)
    assert len(edges) == m
    edge_set = set()
    for e in edges:
        assert len(e) == 2
        assert e[0] != e[1]
        assert 0 <= e[0] < n and 0 <= e[1] < n
        key = tuple(sorted(e))
        assert key not in edge_set
        edge_set.add(key)

    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
