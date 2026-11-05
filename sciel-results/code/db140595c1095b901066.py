import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 148  # safety margin before 150s kill

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V = list(elements)
        V_set = set(V)
        n = len(V)

        # Build hyperedges: (a, b, c) such that a + b = c
        hyperedges = []
        V_list = list(V_set)
        for i, a in enumerate(V_list):
            if 2 * a in V_set:
                hyperedges.append((a, a, 2 * a))
            for j in range(i + 1, len(V_list)):
                b = V_list[j]
                c = a + b
                if c in V_set:
                    hyperedges.append((a, b, c))

        # Build adjacency: adj[v] = list of (u, w) such that {v, u, w} is a hyperedge.
        adj = {v: [] for v in V_set}
        for (a, b, c) in hyperedges:
            if a == b:
                adj[a].append((a, 2 * a))
                adj[2 * a].append((a, a))
            else:
                adj[a].append((b, c))
                adj[b].append((a, c))
                adj[c].append((a, b))

        # Precompute degree
        deg = {v: 0 for v in V_set}
        for (a, b, c) in hyperedges:
            deg[a] += 1
            deg[b] += 1
            deg[c] += 1

        def get_conflicts(v, S):
            """
            Return list of conflicts for adding v to S.
            Each conflict is a pair (u, w) such that {v, u, w} is a hyperedge and u, w in S.
            """
            conflicts = []
            for (u, w) in adj[v]:
                if u == v:
                    # v + v = w. Need w in S.
                    if w in S:
                        conflicts.append((w, w))
                else:
                    # v + u = w. Need u in S and w in S.
                    if u in S and w in S:
                        conflicts.append((u, w))
            return conflicts

        def hypergraph_greedy(V_set, deg, adj, rng, time_budget):
            """
            Hypergraph-aware greedy: iteratively add vertices that create
            the fewest new conflicts, removing hyperedges with minimum impact.
            """
            t_hg = time.time()
            S = set()
            
            # Sort candidates by degree (fewest conflicts first)
            candidates = sorted(V_set, key=lambda x: (deg[x], abs(x)))
            
            for v in candidates:
                if time.time() - t_hg > time_budget:
                    break
                if v in S:
                    continue
                
                # Check if adding v creates any active hyperedges
                new_active = []
                for (u, w) in adj[v]:
                    if u == v:
                        # v + v = w
                        if w in S:
                            new_active.append((v, v, w))
                    else:
                        # v + u = w
                        if u in S and w in S:
                            new_active.append((v, u, w))
                
                if not new_active:
                    S.add(v)
                else:
                    # We need to decide: add v and remove some vertices to break conflicts,
                    # or skip v.
                    conflict_verts = set()
                    for (a, b, c) in new_active:
                        conflict_verts.add(a)
                        conflict_verts.add(b)
                        conflict_verts.add(c)
                    conflict_verts.discard(v)  # We're keeping v
                    
                    if len(conflict_verts) == 1:
                        rem = list(conflict_verts)[0]
                        # Check if removing rem breaks all new_active
                        all_broken = True
                        for (a, b, c) in new_active:
                            if rem not in (a, b, c):
                                all_broken = False
                                break
                        if all_broken:
                            S.discard(rem)
                            S.add(v)
                    # Otherwise, skip v for now
            
            return S

        def local_search_1for2(S, V_set, adj, budget, rng):
            """
            Local search with 1-for-2 swaps: try to add v and remove up to 2 vertices.
            """
            t_ls = time.time()
            S = set(S)
            best_S = set(S)
            best_size = len(S)
            no_improve_count = 0
            
            while time.time() - t_ls < budget:
                if no_improve_count > 50:
                    break
                
                remaining = [x for x in V_set if x not in S]
                if not remaining:
                    break
                
                # Try a few random candidates
                tried = set()
                for _ in range(min(10, len(remaining))):
                    v = rng.choice(remaining)
                    if v in tried:
                        continue
                    tried.add(v)
                    
                    conflicts = get_conflicts(v, S)
                    if not conflicts:
                        S.add(v)
                        no_improve_count = 0
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                        break
                    
                    # Find minimal hitting set of conflicts (size <= 2)
                    freq = {}
                    for (u, w) in conflicts:
                        if u == w:
                            freq[u] = freq.get(u, 0) + 1
                        else:
                            freq[u] = freq.get(u, 0) + 1
                            freq[w] = freq.get(w, 0) + 1
                    
                    if not freq:
                        continue
                    
                    # Try removing 1 vertex
                    max_f1 = max(freq.values())
                    candidates1 = [k for k, c in freq.items() if c == max_f1]
                    
                    found_swap = False
                    for rem in candidates1:
                        trial_S = set(S)
                        trial_S.discard(rem)
                        trial_S.add(v)
                        
                        valid = True
                        for (u, w) in conflicts:
                            if u == w:
                                if u in trial_S:
                                    valid = False
                                    break
                            else:
                                if u in trial_S and w in trial_S:
                                    valid = False
                                    break
                        
                        if valid:
                            # 1-for-1 swap: accept with some probability to explore
                            if rng.random() < 0.3:
                                S = trial_S
                            found_swap = True
                            break
                    
                    if found_swap:
                        continue
                    
                    # Try removing 2 vertices (1-for-2)
                    top_verts = sorted(freq.keys(), key=lambda x: -freq[x])[:4]
                    for i in range(len(top_verts)):
                        for j in range(i + 1, len(top_verts)):
                            rem1, rem2 = top_verts[i], top_verts[j]
                            trial_S = set(S)
                            trial_S.discard(rem1)
                            trial_S.discard(rem2)
                            trial_S.add(v)
                            
                            valid = True
                            for (u, w) in conflicts:
                                if u == w:
                                    if u in trial_S:
                                        valid = False
                                        break
                                else:
                                    if u in trial_S and w in trial_S:
                                        valid = False
                                        break
                            
                            if valid:
                                # 1-for-2: net -1, but v might enable future additions
                                if rng.random() < 0.05:
                                    S = trial_S
                                found_swap = True
                                break
                        if found_swap:
                            break
                    
                    if not found_swap:
                        no_improve_count += 1
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                    no_improve_count = 0
            
            return best_S

        # Initial Solutions
        best_S = set()
        best_size = 0
        rng = random.Random(seed_off)
        
        # 1. Odds
        S_odds = set(x for x in V_set if x % 2 != 0)
        if len(S_odds) > best_size:
            best_size = len(S_odds)
            best_S = S_odds
            
        # 2. Hypergraph-aware greedy
        time_hg = budget * 0.3
        S_hg = hypergraph_greedy(V_set, deg, adj, rng, time_hg)
        if len(S_hg) > best_size:
            best_size = len(S_hg)
            best_S = S_hg
            
        # 3. Local search with 1-for-2 swaps
        time_ls = budget * 0.6
        best_S = local_search_1for2(best_S, V_set, adj, time_ls, rng)
        
        # 4. Final polish: try adding any remaining vertices that fit
        for v in V_set:
            if v not in best_S:
                if not get_conflicts(v, best_S):
                    best_S.add(v)
        
        return list(best_S)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    budgets = [12, 30, 60]
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 5
        if remaining < 4:
            res = [x for x in elems if x % 2 != 0]
            results.append(res)
            continue
        b = min(budgets[i], max(4, remaining))
        res = solve_instance(elems, seed_off=801 + i * 7, budget=b)
        results.append(res)

    return results
