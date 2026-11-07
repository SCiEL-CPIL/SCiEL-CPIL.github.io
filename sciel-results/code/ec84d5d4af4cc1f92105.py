import time
import random
import math
from itertools import combinations


def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under the 90s kill

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    m_total = len(all_edges)  # 91
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
        # dense small - these create many sum collisions
        for hi in (18, 22, 28, 36, 48, 64, 90, 140, 220):
            pools.append(list(range(1, hi + 1)))

        # Arithmetic progressions (great for sum collisions)
        for start_v in range(1, 20):
            for step in [2, 3, 4, 5, 6]:
                pool = [start_v + i * step for i in range(30) if start_v + i * step <= 200]
                if len(pool) >= n:
                    pools.append(pool)

        # Composites / multiples (product collisions)
        base_sets = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 27, 30,
             32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 60, 63, 64, 70, 72, 80, 81,
             84, 90, 96, 98, 100, 105, 108, 112, 120, 125, 126, 128, 135, 140, 144,
             147, 150, 154, 160, 162, 168, 175, 180, 189, 196, 200],
            [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 20, 24, 25, 27, 32, 36, 40, 45,
             49, 54, 64, 72, 80, 81, 90, 98, 100, 108, 121, 128, 144, 147, 162, 169,
             196, 200, 225, 242, 243, 256, 288, 289, 324, 338, 343, 361, 392, 400,
             441, 484, 512, 576, 625, 676],
            # powers-of-2 rich (multiplicative structure)
            [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72,
             81, 96, 108, 128, 144, 162, 192, 216, 256, 288, 324, 384, 432, 486,
             512, 576, 648, 768, 864, 972, 1024, 1152, 1296, 1536, 1728, 2048,
             2304, 2592, 3072, 4096],
            # Multiples of small numbers
            [i * 2 for i in range(1, 50)],
            [i * 3 for i in range(1, 40)],
            [i * 4 for i in range(1, 30)],
            [i * 6 for i in range(1, 25)],
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

    def generate_structured_values():
        """Generate values with strong collision properties."""
        strategies = []

        # Strategy 1: Small consecutive values
        for start_v in range(1, 10):
            for span in range(14, 25):
                pool = list(range(start_v, start_v + span))
                if len(pool) >= n:
                    strategies.append(random.sample(pool, n))

        # Strategy 2: Values clustered around a center
        for center in range(5, 50):
            for spread in [3, 5, 8, 12]:
                pool = []
                for _ in range(50):
                    v = center + random.randint(-spread, spread)
                    if 1 <= v <= 10000:
                        pool.append(v)
                # Ensure distinct
                seen = set()
                unique = []
                for v in pool:
                    if v not in seen:
                        seen.add(v)
                        unique.append(v)
                    if len(unique) == n:
                        break
                if len(unique) == n:
                    strategies.append(unique)

        # Strategy 3: Mix of small values with some structure
        for _ in range(20):
            base = random.randint(1, 5)
            vals = []
            seen = set()
            for i in range(n):
                if random.random() < 0.7:
                    v = base + random.randint(0, 20)
                else:
                    v = random.randint(1, 40)
                while v in seen or v < 1:
                    v = random.randint(1, 50)
                seen.add(v)
                vals.append(v)
            strategies.append(vals)

        return strategies

    # ------------------------------------------------------------------
    # Greedy edge selection for fixed values (build high-collision set).
    # ------------------------------------------------------------------
    def greedy_edges(values):
        chosen = []
        chosen_set = set()

        # Try multiple ordering strategies and keep the best
        strategies = []

        # Strategy 1: Small sums
        order1 = sorted(all_edges, key=lambda e: values[e[0]] + values[e[1]])
        strategies.append(order1)

        # Strategy 2: Small products
        order2 = sorted(all_edges, key=lambda e: values[e[0]] * values[e[1]])
        strategies.append(order2)

        # Strategy 3: Balanced (sum + normalized product)
        max_val = max(values)
        order3 = sorted(all_edges, key=lambda e: values[e[0]] + values[e[1]] +
                        (values[e[0]] * values[e[1]]) / (max_val * max_val + 1))
        strategies.append(order3)

        best_chosen = None
        best_score = float('inf')

        for order in strategies:
            chosen_temp = []
            chosen_set_temp = set()
            # Reset scoring state
            cnt_sum = {}
            cnt_prod = {}
            distinct_sum = 0
            distinct_prod = 0

            for e in order:
                if len(chosen_temp) == m:
                    break
                u, v = e
                s = values[u] + values[v]
                p = values[u] * values[v]

                if s not in cnt_sum:
                    distinct_sum += 1
                cnt_sum[s] = cnt_sum.get(s, 0) + 1

                if p not in cnt_prod:
                    distinct_prod += 1
                cnt_prod[p] = cnt_prod.get(p, 0) + 1

                chosen_temp.append(e)
                chosen_set_temp.add(e)

            score = max(distinct_sum, distinct_prod)
            if score < best_score:
                best_score = score
                best_chosen = chosen_temp
                best_chosen_set = chosen_set_temp

        return best_chosen, best_chosen_set

    # ------------------------------------------------------------------
    # Simulated Annealing for joint value and edge optimization.
    # Supports 1-edge swaps, 2-edge swaps, and value perturbations.
    # ------------------------------------------------------------------
    def sa_search(vals0, edges0, deadline):
        vals = list(vals0)
        edges = list(edges0)
        edges_set = set(edges0)
        
        # Initialize scoring
        init_from, swap, add_edge, raw, _ = make_scoring(vals)
        init_from(edges)
        
        cur_score = raw()
        best_score = cur_score
        best_vals = list(vals)
        best_edges = list(edges)
        
        # SA Parameters
        T = 10.0
        T_min = 0.1
        cooling = 0.99995
        
        # Precompute all possible 2-edge swaps? No, too many (91*90/2 * 35*34/2).
        # We will sample moves.
        
        while time.time() < deadline and T > T_min:
            T *= cooling
            
            # Choose move type: 50% value, 30% 1-edge swap, 20% 2-edge swap
            r = random.random()
            
            if r < 0.5:
                # Value move
                idx = random.randrange(n)
                old_val = vals[idx]
                # Perturb value
                # Try small changes first, then larger
                if random.random() < 0.7:
                    new_val = old_val + random.randint(-5, 5)
                else:
                    new_val = random.randint(1, 100)
                
                if new_val < 1 or new_val > 10000 or new_val == old_val or new_val in vals:
                    continue
                
                vals[idx] = new_val
                # Recompute score (expensive, but necessary for value change)
                new_score = full_score(vals, edges)
                delta = new_score - cur_score
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_vals = list(vals)
                        best_edges = list(edges)
                        # Rebuild incremental state for edge moves
                        init_from, swap, add_edge, raw, _ = make_scoring(vals)
                        init_from(edges)
                else:
                    vals[idx] = old_val
                    # No need to rebuild if rejected, state is consistent
            
            elif r < 0.8:
                # 1-Edge Swap
                i = random.randrange(m)
                e_out = edges[i]
                # Find a random e_in not in edges
                while True:
                    e_in = random.choice(all_edges)
                    if e_in not in edges_set:
                        break
                
                swap(e_out, e_in)
                new_score = raw()
                delta = new_score - cur_score
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    cur_score = new_score
                    edges[i] = e_in
                    edges_set.discard(e_out)
                    edges_set.add(e_in)
                    if cur_score < best_score:
                        best_score = cur_score
                        best_vals = list(vals)
                        best_edges = list(edges)
                else:
                    swap(e_in, e_out)
            
            else:
                # 2-Edge Swap
                # Pick 2 distinct indices
                i1, i2 = random.sample(range(m), 2)
                e_out1 = edges[i1]
                e_out2 = edges[i2]
                
                # Find 2 distinct e_in not in edges and not equal to each other
                while True:
                    e_in1 = random.choice(all_edges)
                    e_in2 = random.choice(all_edges)
                    if e_in1 not in edges_set and e_in2 not in edges_set and e_in1 != e_in2:
                        break
                
                # Apply swaps
                swap(e_out1, e_in1)
                swap(e_out2, e_in2)
                
                new_score = raw()
                delta = new_score - cur_score
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    cur_score = new_score
                    edges[i1] = e_in1
                    edges[i2] = e_in2
                    edges_set.discard(e_out1)
                    edges_set.discard(e_out2)
                    edges_set.add(e_in1)
                    edges_set.add(e_in2)
                    if cur_score < best_score:
                        best_score = cur_score
                        best_vals = list(vals)
                        best_edges = list(edges)
                else:
                    # Revert
                    swap(e_in1, e_out1)
                    swap(e_in2, e_out2)
        
        return best_score, best_vals, best_edges

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

    # Phase 1: Generate good initial solutions using greedy + light polish
    pools = value_pools()
    random.shuffle(pools)
    structured_starts = generate_structured_values()
    
    starts = []
    
    # From pools
    for pool in pools[:10]: # Limit to keep time low
        if time.time() > deadline - 20.0:
            break
        for _ in range(2):
            vals = random_values(pool)
            chosen, chosen_set = greedy_edges(vals)
            sc = full_score(vals, chosen)
            consider(sc, vals, chosen)
            starts.append((list(vals), list(chosen)))
            
    # From structured
    for vals in structured_starts[:10]:
        if time.time() > deadline - 20.0:
            break
        chosen, chosen_set = greedy_edges(vals)
        sc = full_score(vals, chosen)
        consider(sc, vals, chosen)
        starts.append((list(vals), list(chosen)))
        
    # Add some random starts
    for _ in range(5):
        vals = random_values(pools[random.randrange(len(pools))])
        edges = random.sample(all_edges, m)
        starts.append((list(vals), list(edges)))

    # Phase 2: Simulated Annealing on the best starts
    # We will run SA for a fixed time budget per start, or split the remaining time.
    remaining = deadline - time.time()
    if remaining < 10.0:
        return {'values': [int(v) for v in best_vals] if best_vals else list(range(1, 15)), 
                'edges': [[int(u), int(v)] for u, v in best_edges] if best_edges else []}
    
    # Sort starts by initial score
    starts.sort(key=lambda x: full_score(x[0], x[1]))
    
    # Run SA on top 3 starts, split time
    time_per_start = remaining / 3.0
    sa_deadline = time.time() + time_per_start
    
    for vals0, edges0 in starts[:3]:
        if time.time() > deadline - 5.0:
            break
        sc, vals, edges = sa_search(vals0, edges0, sa_deadline)
        consider(sc, vals, edges)
        sa_deadline = time.time() + time_per_start # Next start gets next chunk
        
    # Phase 3: Final Polish on the best result found
    if best_vals is not None and time.time() < deadline - 2.0:
        # Run a very short SA with low T to polish
        polish_deadline = deadline - 1.0
        sc, vals, edges = sa_search(best_vals, best_edges, polish_deadline)
        consider(sc, vals, edges)

    # Final cleanup and type conversion
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])
        
    # Ensure distinct values
    seen_vals = set()
    clean_vals = []
    for v in best_vals:
        if v not in seen_vals and 1 <= v <= 10000:
            seen_vals.add(v)
            clean_vals.append(v)
    while len(clean_vals) < n:
        v = random.randint(1, 10000)
        if v not in seen_vals:
            seen_vals.add(v)
            clean_vals.append(v)
    best_vals = clean_vals[:n]
    
    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    # Defensive edge cleaning
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
