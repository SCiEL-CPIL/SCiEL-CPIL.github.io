import time
import random
import math
from itertools import combinations


def construct():
    start = time.time()
    deadline = start + 85.0          # safety margin under the 90s kill

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    m_total = len(all_edges)         # 91
    all_edges_set = set(all_edges)

    # ------------------------------------------------------------------
    # Fast incremental scorer: keeps the multiset (counts) of edge sums
    # and edge products so a single edge swap updates in O(1).
    # ------------------------------------------------------------------
    def make_scoring(values):
        sv = values[:]
        cnt_sum = {}
        cnt_prod = {}
        distinct_sum = 0
        distinct_prod = 0

        def add(x, cnt, which):
            nonlocal distinct_sum, distinct_prod
            c = cnt.get(x, 0)
            if c == 0:
                if which == 's':
                    distinct_sum += 1
                else:
                    distinct_prod += 1
            cnt[x] = c + 1

        def rem(x, cnt, which):
            nonlocal distinct_sum, distinct_prod
            c = cnt.get(x, 0)
            if c == 1:
                del cnt[x]
                if which == 's':
                    distinct_sum -= 1
                else:
                    distinct_prod -= 1
            else:
                cnt[x] = c - 1

        def swap(out, inn):
            u, v = out
            s0 = sv[u] + sv[v]
            p0 = sv[u] * sv[v]
            rem(s0, cnt_sum, 's')
            rem(p0, cnt_prod, 'p')
            u, v = inn
            s1 = sv[u] + sv[v]
            p1 = sv[u] * sv[v]
            add(s1, cnt_sum, 's')
            add(p1, cnt_prod, 'p')

        def add_edge(e):
            u, v = e
            add(sv[u] + sv[v], cnt_sum, 's')
            add(sv[u] * sv[v], cnt_prod, 'p')

        def raw():
            return max(distinct_sum, distinct_prod)

        def init_from(edges):
            for e in edges:
                add_edge(e)

        return init_from, swap, add_edge, raw, (cnt_sum, cnt_prod)

    def full_score(values, edge_list):
        s = set()
        p = set()
        for u, v in edge_list:
            s.add(values[u] + values[v])
            p.add(values[u] * values[v])
        return max(len(s), len(p))

    # ------------------------------------------------------------------
    # Candidate value pools: small / dense / composite-rich so that both
    # sums and products collide heavily among the 35 chosen edges.
    # ------------------------------------------------------------------
    def value_pools():
        pools = []
        # dense small
        for hi in (18, 22, 28, 36, 48, 64, 90, 140, 220):
            pools.append(list(range(1, hi + 1)))
        
        # Structured sets with many additive relations (arithmetic progressions)
        for start in (1, 2, 3, 4, 5):
            for step in (1, 2, 3, 4):
                for length in (14, 16, 18, 20):
                    pool = [start + i * step for i in range(length)]
                    if all(1 <= v <= 10000 for v in pool):
                        pools.append(pool)
        
        # Multiplicative structure: geometric progressions
        for base in (2, 3, 4, 5):
            for start in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
                pool = [start * (base ** i) for i in range(14)]
                if all(1 <= v <= 10000 for v in pool):
                    pools.append(pool)
        
        # Mixed: small numbers with many factorizations
        base_sets = [
            # Rich in small factors
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 27, 30,
             32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 60, 63, 64, 70, 72, 80, 81,
             84, 90, 96, 98, 100, 105, 108, 112, 120, 125, 126, 128, 135, 140, 144,
             147, 150, 154, 160, 162, 168, 175, 180, 189, 196, 200],
            # Powers and multiples
            [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 20, 24, 25, 27, 32, 36, 40, 45,
             49, 54, 64, 72, 80, 81, 90, 98, 100, 108, 121, 128, 144, 147, 162, 169,
             196, 200, 225, 242, 243, 256, 288, 289, 324, 338, 343, 361, 392, 400,
             441, 484, 512, 576, 625, 676],
            # powers-of-2 rich
            [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72,
             81, 96, 108, 128, 144, 162, 192, 216, 256, 288, 324, 384, 432, 486,
             512, 576, 648, 768, 864, 972, 1024, 1152, 1296, 1536, 1728, 2048,
             2304, 2592, 3072, 4096],
            # Very small: 1-14 with duplicates removed
            list(range(1, 15)),
            # 1-20
            list(range(1, 21)),
            # 2-15 (no 1)
            list(range(2, 16)),
            # Even numbers 2-28
            list(range(2, 29, 2)),
            # Odd numbers 1-27
            list(range(1, 28, 2)),
            # Multiples of 2: 2,4,6,...,28
            list(range(2, 29, 2)),
            # Multiples of 3: 3,6,9,...,42
            list(range(3, 43, 3)),
            # Multiples of 4: 4,8,12,...,56
            list(range(4, 57, 4)),
        ]
        pools.extend(base_sets)
        return pools

    def random_values(pool):
        if len(pool) >= n:
            return random.sample(pool, n)
        vals = []
        seen = set()
        while len(vals) < n:
            v = random.randint(1, 10000)
            if v not in seen:
                seen.add(v)
                vals.append(v)
        return vals

    # ------------------------------------------------------------------
    # Greedy edge selection for fixed values (build high-collision set).
    # ------------------------------------------------------------------
    def greedy_edges(values):
        init_from, swap, add_edge, raw, _ = make_scoring(values)
        chosen = []
        chosen_set = set()
        # order edges by small sum (tends to reuse small sums/products)
        order = sorted(all_edges, key=lambda e: (values[e[0]] + values[e[1]],
                                                 values[e[0]] * values[e[1]]))
        for e in order:
            if len(chosen) == m:
                break
            add_edge(e)
            chosen.append(e)
            chosen_set.add(e)
        return chosen, chosen_set

    # ------------------------------------------------------------------
    # Edge-only local search (incremental).
    # ------------------------------------------------------------------
    def polish_edges(values, chosen, chosen_set, steps, deadline):
        init_from, swap, add_edge, raw, _ = make_scoring(values)
        init_from(chosen)
        cur = raw()
        best = cur
        for _ in range(steps):
            if time.time() > deadline:
                break
            i = random.randrange(m)
            e_out = chosen[i]
            e_in = random.choice(all_edges)
            if e_in in chosen_set:
                continue
            swap(e_out, e_in)
            r = raw()
            if r <= cur:
                cur = r
                chosen[i] = e_in
                chosen_set.discard(e_out)
                chosen_set.add(e_in)
                if r < best:
                    best = r
            else:
                # revert
                swap(e_in, e_out)
        return chosen, chosen_set, best

    # ------------------------------------------------------------------
    # Joint value+edge local search (recompute score on value moves,
    # incremental on edge moves).
    # ------------------------------------------------------------------
    def joint_search(vals0, edges0, steps, deadline):
        vals = list(vals0)
        edges = list(edges0)
        edges_set = set(edges0)
        cur = full_score(vals, edges)
        best = cur
        best_vals = list(vals)
        best_edges = list(edges)
        init_from, swap, add_edge, raw, _ = make_scoring(vals)
        init_from(edges)
        
        # Precompute candidate values for smart proposals
        small_vals = list(range(1, 31))
        medium_vals = list(range(1, 61))
        
        for _ in range(steps):
            if time.time() > deadline:
                break
            if random.random() < 0.45:
                # edge move (incremental)
                i = random.randrange(m)
                e_out = edges[i]
                e_in = random.choice(all_edges)
                if e_in in edges_set:
                    continue
                swap(e_out, e_in)
                r = raw()
                if r <= cur:
                    cur = r
                    edges[i] = e_in
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                else:
                    swap(e_in, e_out)
            else:
                # value move (recompute)
                idx = random.randrange(n)
                old = vals[idx]
                
                # Smart value proposal: try values that create collisions
                # with existing values
                r = random.random()
                if r < 0.3:
                    # Very small values
                    nv = random.choice(small_vals)
                elif r < 0.5:
                    # Small values
                    nv = random.choice(small_vals)
                elif r < 0.7:
                    # Medium values
                    nv = random.choice(medium_vals)
                elif r < 0.85:
                    # Try to match sum/product with an existing edge
                    # Pick a random other vertex and try values that create same sum
                    other_idx = random.randrange(n)
                    if other_idx != idx:
                        target_sum = vals[other_idx] + random.randint(1, 40)
                        nv = target_sum - vals[other_idx]
                        if nv < 1:
                            nv = random.randint(1, 40)
                    else:
                        nv = random.randint(1, 40)
                else:
                    # Random
                    nv = random.randint(1, 80)
                
                if nv < 1 or nv > 10000:
                    continue
                if nv == old or (nv in vals):
                    continue
                vals[idx] = nv
                r = full_score(vals, edges)
                if r <= cur:
                    cur = r
                    # rebuild incremental state (only on accepted value move)
                    init_from, swap, add_edge, raw, _ = make_scoring(vals)
                    init_from(edges)
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                else:
                    vals[idx] = old
        return best, best_vals, best_edges

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    best_score = float('inf')
    best_vals = None
    best_edges = None

    def consider(sc, vals, edges):
        nonlocal best_score, best_vals, best_edges
        if sc < best_score:
            best_score = sc
            best_vals = list(vals)
            best_edges = list(edges)

    # Phase 1: structured value pools -> greedy edges -> polish edges.
    pools = value_pools()
    random.shuffle(pools)
    
    # Sort pools by potential: prefer pools with more internal structure
    # (many pairs with same sum/product)
    def pool_score(pool):
        if len(pool) < n:
            return 0
        # Count collisions in sums and products for random samples
        samples = random.sample(pool, min(len(pool), 30))
        sum_counts = {}
        prod_counts = {}
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                s = samples[i] + samples[j]
                p = samples[i] * samples[j]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1
        sum_collisions = sum(c for c in sum_counts.values() if c > 1)
        prod_collisions = sum(c for c in prod_counts.values() if c > 1)
        return sum_collisions + prod_collisions
    
    # Try best pools first, but also keep some random exploration
    scored_pools = [(pool_score(p), p) for p in pools]
    scored_pools.sort(key=lambda x: -x[0])
    
    # Take top pools + some random
    top_pools = [p for _, p in scored_pools[:20]]
    remaining_pools = [p for _, p in scored_pools[20:]]
    random.shuffle(remaining_pools)
    ordered_pools = top_pools + remaining_pools[:10]
    random.shuffle(ordered_pools)
    
    for pool in ordered_pools:
        if time.time() > deadline - 6.0:
            break
        for _ in range(3):  # More restarts per pool
            if time.time() > deadline - 5.0:
                break
            vals = random_values(pool)
            chosen, chosen_set = greedy_edges(vals)
            sc = full_score(vals, chosen)
            consider(sc, vals, chosen)
            # polish edges for this fixed value set
            steps = 5000
            polished, pset, psc = polish_edges(vals, chosen[:], chosen_set.copy(),
                                               steps, deadline - 4.0)
            consider(psc, vals, polished)

    # Phase 2: joint search from several starts.
    starts = []
    if best_vals is not None:
        starts.append((list(best_vals), list(best_edges)))
    
    # Add diverse starts from top pools
    for pool in top_pools[:8]:
        vals = random_values(pool)
        edges = random.sample(all_edges, m)
        starts.append((vals, edges))
    
    # Add a few random starts
    for _ in range(4):
        vals = random_values(pools[random.randrange(len(pools))])
        edges = random.sample(all_edges, m)
        starts.append((vals, edges))

    remaining = deadline - 5.0 - time.time()
    per = max(3000, int(remaining / max(1, len(starts)) / 0.00025))
    for vals0, edges0 in starts:
        if time.time() > deadline - 5.0:
            break
        sc, vals, edges = joint_search(vals0, edges0, per, deadline - 4.0)
        consider(sc, vals, edges)

    # Phase 3: final edge polish for best values.
    if time.time() < deadline - 3.0 and best_vals is not None:
        chosen, chosen_set = list(best_edges), set(best_edges)
        polished, pset, psc = polish_edges(best_vals, chosen, chosen_set,
                                           30000, deadline - 2.0)
        consider(psc, best_vals, polished)

    # Phase 4: final value polish for best edges with smarter proposals.
    if time.time() < deadline - 1.0 and best_edges is not None:
        vals = list(best_vals)
        edges = list(best_edges)
        cur = full_score(vals, edges)
        best = cur
        it = 0
        small_vals = list(range(1, 31))
        while it < 15000 and time.time() < deadline - 1.0:
            it += 1
            idx = random.randrange(n)
            old = vals[idx]
            
            # Smarter value proposal
            r = random.random()
            if r < 0.4:
                nv = random.choice(small_vals)
            elif r < 0.6:
                # Try to create sum collision with an existing edge
                other_idx = random.randrange(n)
                if other_idx != idx:
                    target = random.randint(2, 60)
                    nv = target - vals[other_idx]
                    if nv < 1:
                        nv = random.randint(1, 30)
                else:
                    nv = random.randint(1, 30)
            elif r < 0.8:
                nv = random.randint(1, 50)
            else:
                nv = random.randint(1, 100)
            
            if nv < 1 or nv > 10000:
                continue
            if nv == old or nv in vals:
                continue
            vals[idx] = nv
            r = full_score(vals, edges)
            if r < cur:
                cur = r
                if r < best:
                    best = r
                    best_vals = list(vals)
                    best_score = best
            else:
                vals[idx] = old

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])
        best_score = full_score(best_vals, best_edges)
    
    # Final refinement: try small perturbations of the best solution
    if time.time() < deadline - 0.5:
        vals = list(best_vals)
        edges = list(best_edges)
        cur = full_score(vals, edges)
        small_vals = list(range(1, 25))
        for _ in range(5000):
            if time.time() > deadline - 0.5:
                break
            idx = random.randrange(n)
            old = vals[idx]
            nv = random.choice(small_vals)
            if nv == old or nv in vals:
                continue
            vals[idx] = nv
            r = full_score(vals, edges)
            if r < cur:
                cur = r
                best_vals = list(vals)
                best_score = cur
            else:
                vals[idx] = old

    # ------------------------------------------------------------------
    # Strict type conversion (Python int / list only).
    # ------------------------------------------------------------------
    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    # defensive: ensure exactly 35 unique simple edges, 14 distinct ints
    seen = set()
    clean = []
    for e in final_edges:
        a, b = e
        if a == b or a < 0 or b < 0 or a >= n or b >= n:
            continue
        t = tuple(sorted((a, b)))
        if t in seen:
            continue
        seen.add(t)
        clean.append([a, b])
    # if somehow short, pad with missing edges (should not happen)
    if len(clean) < m:
        for e in all_edges:
            t = tuple(sorted(e))
            if t not in seen:
                seen.add(t)
                clean.append([e[0], e[1]])
                if len(clean) == m:
                    break
    final_edges = clean[:m]

    return {'values': final_values, 'edges': final_edges}
