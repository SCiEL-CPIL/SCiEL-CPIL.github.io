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

    # Precompute for faster full_score
    edge_sums_cache = {}
    edge_prods_cache = {}

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

    def full_score_fast(values, edge_list):
        """Optimized version using local variables for speed."""
        s = set()
        p = set()
        vals = values
        for e in edge_list:
            u, v = e
            s.add(vals[u] + vals[v])
            p.add(vals[u] * vals[v])
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
        init_from, swap, add_edge, raw, _ = make_scoring(values)
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
    # Edge-only local search (incremental).
    # ------------------------------------------------------------------
    def polish_edges(values, chosen, chosen_set, steps, deadline):
        init_from, swap, add_edge, raw, _ = make_scoring(values)
        init_from(chosen)
        cur = raw()
        best = cur
        no_improve = 0
        consecutive_no_improve = 0
        
        # Precompute edge sums and products for faster evaluation
        val = values
        
        for _ in range(steps):
            if time.time() > deadline:
                break
            
            i = random.randrange(m)
            e_out = chosen[i]
            
            # Try a few random candidates and pick the best
            best_e_in = None
            best_r = float('inf')
            best_swap_data = None
            
            for _ in range(5):
                e_in = random.choice(all_edges)
                if e_in in chosen_set:
                    continue
                
                swap(e_out, e_in)
                r = raw()
                
                if r < best_r:
                    best_r = r
                    best_e_in = e_in
                    best_swap_data = (e_out, e_in)
                
                # Revert for next trial
                swap(e_in, e_out)
            
            if best_e_in is None:
                no_improve += 1
                consecutive_no_improve += 1
                if consecutive_no_improve > 5000 and random.random() < 0.05:
                    consecutive_no_improve = 0
                continue
            
            # Apply the best swap
            e_out, e_in = best_swap_data
            swap(e_out, e_in)
            r = raw()
            
            if r <= cur:
                cur = r
                chosen[i] = e_in
                chosen_set.discard(e_out)
                chosen_set.add(e_in)
                if r < best:
                    best = r
                no_improve = 0
                consecutive_no_improve = 0
            else:
                swap(e_in, e_out)
                no_improve += 1
                consecutive_no_improve += 1
        
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
        no_improve = 0
        consecutive_no_improve = 0
        
        for _ in range(steps):
            if time.time() > deadline:
                break
            
            if random.random() < 0.5:
                # edge move (incremental) - try multiple candidates
                i = random.randrange(m)
                e_out = edges[i]
                
                best_e_in = None
                best_r = float('inf')
                best_swap_data = None
                
                for _ in range(3):
                    e_in = random.choice(all_edges)
                    if e_in in edges_set:
                        continue
                    
                    swap(e_out, e_in)
                    r = raw()
                    
                    if r < best_r:
                        best_r = r
                        best_e_in = e_in
                        best_swap_data = (e_out, e_in)
                    
                    swap(e_in, e_out)
                
                if best_e_in is None:
                    no_improve += 1
                    consecutive_no_improve += 1
                    continue
                
                e_out, e_in = best_swap_data
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
                    no_improve = 0
                    consecutive_no_improve = 0
                else:
                    swap(e_in, e_out)
                    no_improve += 1
                    consecutive_no_improve += 1
            else:
                # value move (recompute) - smarter proposals
                idx = random.randrange(n)
                old = vals[idx]
                
                # Generate multiple candidate values and pick the best
                candidates = []
                
                # Small values
                for _ in range(3):
                    candidates.append(random.randint(1, 30))
                # Medium values
                for _ in range(2):
                    candidates.append(random.randint(10, 60))
                # Related to existing values
                for _ in range(2):
                    other_idx = random.randrange(n)
                    if other_idx != idx:
                        candidates.append(vals[other_idx] + random.randint(-5, 5))
                        candidates.append(vals[other_idx] * random.choice([2, 3]))
                
                best_nv = None
                best_r = float('inf')
                
                for nv in candidates:
                    if nv < 1 or nv > 10000 or nv == old or nv in vals:
                        continue
                    
                    vals[idx] = nv
                    r = full_score(vals, edges)
                    
                    if r < best_r:
                        best_r = r
                        best_nv = nv
                    
                    vals[idx] = old
                
                if best_nv is None:
                    no_improve += 1
                    consecutive_no_improve += 1
                    continue
                
                vals[idx] = best_nv
                r = best_r
                
                if r <= cur:
                    cur = r
                    # rebuild incremental state (only on accepted value move)
                    init_from, swap, add_edge, raw, _ = make_scoring(vals)
                    init_from(edges)
                    if r < best:
                        best = r
                        best_vals = list(vals)
                        best_edges = list(edges)
                    no_improve = 0
                    consecutive_no_improve = 0
                else:
                    vals[idx] = old
                    no_improve += 1
                    consecutive_no_improve += 1
        
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
    
    # Also add structured value generators
    structured_starts = generate_structured_values()
    
    # Process pools
    for pool in pools:
        if time.time() > deadline - 8.0:
            break
        for _ in range(3):  # More attempts per pool
            if time.time() > deadline - 7.0:
                break
            vals = random_values(pool)
            chosen, chosen_set = greedy_edges(vals)
            sc = full_score(vals, chosen)
            consider(sc, vals, chosen)
            # polish edges for this fixed value set
            steps = 8000
            polished, pset, psc = polish_edges(vals, chosen[:], chosen_set.copy(),
                                               steps, deadline - 5.0)
            consider(psc, vals, polished)
    
    # Process structured starts
    for vals in structured_starts:
        if time.time() > deadline - 5.0:
            break
        chosen, chosen_set = greedy_edges(vals)
        sc = full_score(vals, chosen)
        consider(sc, vals, chosen)
        # polish edges
        steps = 6000
        polished, pset, psc = polish_edges(vals, chosen[:], chosen_set.copy(),
                                           steps, deadline - 4.0)
        consider(psc, vals, polished)

    # Phase 2: joint search from several starts.
    starts = []
    if best_vals is not None:
        starts.append((list(best_vals), list(best_edges)))
    
    # Add diverse starting points
    for _ in range(8):
        if random.random() < 0.5:
            vals = random_values(pools[random.randrange(len(pools))])
        else:
            vals = random.choice(structured_starts)
        edges = random.sample(all_edges, m)
        starts.append((vals, edges))
    
    # Also try from structured values with greedy edges
    for vals in structured_starts[:5]:
        chosen, chosen_set = greedy_edges(vals)
        starts.append((list(vals), list(chosen)))
    
    remaining = deadline - 6.0 - time.time()
    if remaining < 2.0:
        remaining = 2.0
    per = max(5000, int(remaining / max(1, len(starts)) / 0.00025))
    
    for vals0, edges0 in starts:
        if time.time() > deadline - 6.0:
            break
        sc, vals, edges = joint_search(vals0, edges0, per, deadline - 5.0)
        consider(sc, vals, edges)

    # Phase 3: final edge polish for best values.
    if time.time() < deadline - 4.0 and best_vals is not None:
        chosen, chosen_set = list(best_edges), set(best_edges)
        polished, pset, psc = polish_edges(best_vals, chosen, chosen_set,
                                           60000, deadline - 3.0)
        consider(psc, best_vals, polished)

    # Phase 4: final value polish for best edges.
    if time.time() < deadline - 2.0 and best_edges is not None:
        vals = list(best_vals)
        edges = list(best_edges)
        cur = full_score(vals, edges)
        best = cur
        it = 0
        no_improve = 0
        consecutive_no_improve = 0
        
        while it < 30000 and time.time() < deadline - 1.5:
            it += 1
            idx = random.randrange(n)
            old = vals[idx]
            
            # Generate multiple candidates
            candidates = []
            for _ in range(4):
                candidates.append(random.randint(1, 50))
            for _ in range(2):
                candidates.append(random.randint(10, 100))
            # Related values
            for _ in range(2):
                other_idx = random.randrange(n)
                if other_idx != idx:
                    candidates.append(vals[other_idx] + random.randint(-5, 5))
                    candidates.append(vals[other_idx] * random.choice([2, 3]))
            
            best_nv = None
            best_r = float('inf')
            
            for nv in candidates:
                if nv < 1 or nv > 10000 or nv == old or nv in vals:
                    continue
                
                vals[idx] = nv
                r = full_score(vals, edges)
                
                if r < best_r:
                    best_r = r
                    best_nv = nv
                
                vals[idx] = old
            
            if best_nv is None:
                no_improve += 1
                consecutive_no_improve += 1
                continue
            
            vals[idx] = best_nv
            r = best_r
            
            if r < cur:
                cur = r
                if r < best:
                    best = r
                    best_vals = list(vals)
                    best_score = best
                no_improve = 0
                consecutive_no_improve = 0
            else:
                vals[idx] = old
                no_improve += 1
                consecutive_no_improve += 1

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])
        best_score = full_score(best_vals, best_edges)
    
    # Final verification and cleanup
    # Ensure all values are distinct positive integers
    seen_vals = set()
    clean_vals = []
    for v in best_vals:
        if v not in seen_vals and 1 <= v <= 10000:
            seen_vals.add(v)
            clean_vals.append(v)
    
    # If we lost some values, add new distinct ones
    while len(clean_vals) < n:
        v = random.randint(1, 10000)
        if v not in seen_vals:
            seen_vals.add(v)
            clean_vals.append(v)
    
    best_vals = clean_vals[:n]

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
