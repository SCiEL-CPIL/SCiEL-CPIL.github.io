import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 86.0

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def score_of(values, edge_list):
        sums = set()
        prods = set()
        for u, v in edge_list:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def fast_score_of(values, edge_list):
        """Incremental-friendly score: returns (sums_set, prods_set, score)."""
        sums = set()
        prods = set()
        for u, v in edge_list:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def score_delta(values, edge_list, e_out, e_in):
        """Compute new score after removing e_out and adding e_in.
        Returns (new_score, new_sums_count, new_prods_count) or None if invalid."""
        u_out, v_out = e_out
        u_in, v_in = e_in
        s_out = values[u_out] + values[v_out]
        p_out = values[u_out] * values[v_out]
        s_in = values[u_in] + values[v_in]
        p_in = values[u_in] * values[v_in]
        
        # Count how many times each appears in current edge_list
        # This is O(m) but m=35 is small
        sum_counts = {}
        prod_counts = {}
        for u, v in edge_list:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        # Remove e_out contribution
        sum_counts[s_out] -= 1
        if sum_counts[s_out] <= 0:
            del sum_counts[s_out]
        prod_counts[p_out] -= 1
        if prod_counts[p_out] <= 0:
            del prod_counts[p_out]
        
        # Add e_in contribution
        sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
        prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
        
        return max(len(sum_counts), len(prod_counts))

    def best_edges_for_values(values, budget):
        """Greedy + local search: pick 35 edges to minimize score, for fixed values."""
        # Greedy init: pick edges that reuse existing sum/product
        chosen = []
        chosen_set = set()
        sums = set()
        prods = set()
        # Sort all edges by a heuristic: prefer pairs with small values
        edge_order = sorted(all_edges, key=lambda e: values[e[0]] + values[e[1]])
        for e in edge_order:
            if len(chosen) == m:
                break
            chosen.append(e)
            chosen_set.add(e)
            sums.add(values[e[0]] + values[e[1]])
            prods.add(values[e[0]] * values[e[1]])
        # Local search to improve
        cur = list(chosen)
        cur_score = score_of(values, cur)
        no_improve = 0
        it = 0
        while it < budget and time.time() < deadline - 3.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = cur[i]
            e_in = random.choice(all_edges)
            if e_in in chosen_set:
                continue
            new_score = score_delta(values, cur, e_out, e_in)
            if new_score <= cur_score:
                cur[i] = e_in
                chosen_set.discard(e_out)
                chosen_set.add(e_in)
                cur_score = new_score
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 2000 and random.random() < 0.01:
                    no_improve = 0
        return cur, cur_score

    def best_values_for_edges(edges, budget):
        """Local search on values for fixed edges."""
        # Start with small distinct values
        vals = random.sample(range(1, 41), n)
        elist = list(edges)
        cur_score = score_of(vals, elist)
        best = list(vals)
        best_score = cur_score
        no_improve = 0
        it = 0
        while it < budget and time.time() < deadline - 3.0:
            it += 1
            idx = random.randint(0, n - 1)
            if random.random() < 0.5:
                # replace
                new_val = random.randint(1, 100)
            else:
                new_val = vals[idx] + random.randint(-15, 15)
            if not (1 <= new_val <= 10000) or new_val in vals:
                continue
            new_vals = list(vals)
            new_vals[idx] = new_val
            s = score_of(new_vals, elist)
            if s < cur_score:
                vals = new_vals
                cur_score = s
                no_improve = 0
                if s < best_score:
                    best = list(vals)
                    best_score = s
            elif s == cur_score and random.random() < 0.1:
                vals = new_vals
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 3000:
                    no_improve = 0
        return best, best_score

    def joint_search(vals0, edges0, budget):
        """Alternate value and edge moves with fast scoring."""
        vals = list(vals0)
        edges = set(edges0)
        elist = list(edges)
        cur_score = score_of(vals, elist)
        best_score = cur_score
        best_vals = list(vals)
        best_edges = set(edges)
        it = 0
        while it < budget and time.time() < deadline - 2.0:
            it += 1
            if random.random() < 0.5:
                # edge move - use fast delta scoring
                i = random.randint(0, m - 1)
                e_out = elist[i]
                e_in = random.choice(all_edges)
                if e_in in edges:
                    continue
                s = score_delta(vals, elist, e_out, e_in)
                if s <= cur_score:
                    elist[i] = e_in
                    edges.discard(e_out)
                    edges.add(e_in)
                    cur_score = s
            else:
                # value move
                idx = random.randint(0, n - 1)
                new_val = random.randint(1, 80)
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s = score_of(new_vals, elist)
                if s <= cur_score:
                    vals = new_vals
                    cur_score = s
            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
                best_edges = set(edges)
        return best_score, best_vals, best_edges

    def try_direct_construction():
        """Direct construction: find values + edges with high collision."""
        best = None
        # Strategy: pick 14 values from a small pool, then find 35 edges.
        # Small pool -> small product range -> collisions.
        pools = [
            list(range(1, 25)),
            list(range(2, 30)),
            list(range(1, 20)),
            [1,2,3,4,5,6,7,8,9,10,11,12,13,15],
            [1,2,3,4,5,6,7,8,10,12,15,16,18,20],
            [1,2,3,4,5,6,7,9,10,12,14,15,18,21],
        ]
        for pool in pools:
            for _ in range(3):
                vals = random.sample(pool, n)
                edges, sc = best_edges_for_values(vals, budget=3000)
                if best is None or sc < best[0]:
                    best = (sc, list(vals), edges)
                if time.time() > deadline - 30:
                    return best
        return best

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Phase 1: direct construction
    d = try_direct_construction()
    if d is not None:
        best_score, best_vals, best_edges = d

    # Phase 2: joint search from multiple starts
    random.seed(1234)
    starts = []
    if best_vals is not None:
        starts.append((list(best_vals), set(best_edges)))
    # more random starts with small values
    for _ in range(8):
        vals = random.sample(range(1, 60), n)
        edges = set(random.sample(all_edges, m))
        starts.append((vals, edges))

    per = max(2000, int((deadline - time.time() - 8.0) / len(starts) / 0.00005))
    for vals0, edges0 in starts:
        if time.time() > deadline - 8.0:
            break
        sc, vals, edges = joint_search(vals0, edges0, per)
        if sc < best_score:
            best_score = sc
            best_vals = vals
            best_edges = edges

    # Phase 3: polish edges for best values
    if time.time() < deadline - 4.0:
        elist = list(best_edges)
        it = 0
        while it < 20000 and time.time() < deadline - 2.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = elist[i]
            e_in = random.choice(all_edges)
            if e_in in best_edges:
                continue
            s = score_delta(best_vals, elist, e_out, e_in)
            if s < best_score:
                elist[i] = e_in
                best_edges.discard(e_out)
                best_edges.add(e_in)
                best_score = s

    # Phase 4: polish values for best edges
    if time.time() < deadline - 1.0:
        elist = list(best_edges)
        vals = list(best_vals)
        it = 0
        while it < 15000 and time.time() < deadline - 0.5:
            it += 1
            idx = random.randint(0, n - 1)
            new_val = random.randint(1, 100)
            if new_val in vals:
                continue
            new_vals = list(vals)
            new_vals[idx] = new_val
            s = score_of(new_vals, elist)
            if s < best_score:
                vals = new_vals
                best_score = s
                best_vals = new_vals

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = set(all_edges[:m])
        best_score = score_of(best_vals, list(best_edges))

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    return {'values': final_values, 'edges': final_edges}
