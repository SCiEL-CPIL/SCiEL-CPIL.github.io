import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s kill

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

    def score_delta(values, edge_list, e_out, e_in):
        """Compute new score after removing e_out and adding e_in. O(m)."""
        u_out, v_out = e_out
        u_in, v_in = e_in
        s_out = values[u_out] + values[v_out]
        p_out = values[u_out] * values[v_out]
        s_in = values[u_in] + values[v_in]
        p_in = values[u_in] * values[v_in]

        sum_counts = {}
        prod_counts = {}
        for u, v in edge_list:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1

        sum_counts[s_out] -= 1
        if sum_counts[s_out] <= 0:
            del sum_counts[s_out]
        prod_counts[p_out] -= 1
        if prod_counts[p_out] <= 0:
            del prod_counts[p_out]

        sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
        prod_counts[p_in] = prod_counts.get(p_in, 0) + 1

        return max(len(sum_counts), len(prod_counts))

    def best_edges_for_values(values, budget):
        """Greedy init (sum-sorted) + fast local search on edges."""
        chosen = []
        chosen_set = set()
        # Sort by sum to maximize sum collisions
        edge_order = sorted(all_edges, key=lambda e: values[e[0]] + values[e[1]])
        for e in edge_order:
            if len(chosen) == m:
                break
            chosen.append(e)
            chosen_set.add(e)

        cur = list(chosen)
        cur_score = score_of(values, cur)
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
        return cur, cur_score

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
            r = random.random()
            if r < 0.45:
                # Edge move (fast delta)
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
            elif r < 0.85:
                # Single value move
                idx = random.randint(0, n - 1)
                # Try values near current and in small range
                if random.random() < 0.6:
                    new_val = random.randint(1, 50)
                else:
                    new_val = vals[idx] + random.randint(-10, 10)
                    new_val = max(1, min(10000, new_val))
                if new_val in vals:
                    continue
                new_vals = list(vals)
                new_vals[idx] = new_val
                s = score_of(new_vals, elist)
                if s <= cur_score:
                    vals = new_vals
                    cur_score = s
            else:
                # Pair value move: swap two values to create product collisions
                i1, i2 = random.sample(range(n), 2)
                # Try making vals[i1]*vals[j] collide with vals[i2]*vals[k] for some edges
                # Simple: try small systematic values
                candidates = list(range(1, 40))
                c1 = random.choice(candidates)
                c2 = random.choice(candidates)
                if c1 in vals or c2 in vals or c1 == c2:
                    continue
                new_vals = list(vals)
                new_vals[i1] = c1
                new_vals[i2] = c2
                s = score_of(new_vals, elist)
                if s <= cur_score:
                    vals = new_vals
                    cur_score = s
            if cur_score < best_score:
                best_score = cur_score
                best_vals = list(vals)
                best_edges = set(edges)
        return best_score, best_vals, best_edges

    def polish_edges(vals, edges, budget):
        """Intensive edge polish with fast delta."""
        elist = list(edges)
        edges_set = set(edges)
        cur_score = score_of(vals, elist)
        it = 0
        while it < budget and time.time() < deadline - 1.0:
            it += 1
            i = random.randint(0, m - 1)
            e_out = elist[i]
            e_in = random.choice(all_edges)
            if e_in in edges_set:
                continue
            s = score_delta(vals, elist, e_out, e_in)
            if s < cur_score:
                elist[i] = e_in
                edges_set.discard(e_out)
                edges_set.add(e_in)
                cur_score = s
        return cur_score, elist, edges_set

    def polish_values(vals, edges, budget):
        """Intensive value polish."""
        elist = list(edges)
        cur_score = score_of(vals, elist)
        it = 0
        while it < budget and time.time() < deadline - 0.5:
            it += 1
            idx = random.randint(0, n - 1)
            if random.random() < 0.5:
                new_val = random.randint(1, 60)
            else:
                new_val = vals[idx] + random.randint(-5, 5)
                new_val = max(1, min(10000, new_val))
            if new_val in vals:
                continue
            new_vals = list(vals)
            new_vals[idx] = new_val
            s = score_of(new_vals, elist)
            if s < cur_score:
                vals = new_vals
                cur_score = s
        return cur_score, vals

    def collision_seeking_move(vals, edges, elist):
        """Targeted move: try to find a value assignment that creates more collisions.
        For each vertex, try values that make its edge sums/products collide with existing ones."""
        cur_score = score_of(vals, elist)
        # Compute current sums and products
        sum_set = set()
        prod_set = set()
        for u, v in elist:
            sum_set.add(vals[u] + vals[v])
            prod_set.add(vals[u] * vals[v])
        
        # For each vertex, find candidate values that create collisions
        # A value x at vertex i creates collisions if for neighbors j:
        #   x + vals[j] in sum_set OR x * vals[j] in prod_set
        improvements = 0
        for idx in range(n):
            # Find neighbors of idx
            neighbors = []
            for u, v in elist:
                if u == idx:
                    neighbors.append(v)
                elif v == idx:
                    neighbors.append(u)
            
            # Try a range of candidate values
            base = vals[idx]
            candidates = set()
            # Near current value
            for d in range(-15, 16):
                cv = base + d
                if 1 <= cv <= 10000 and cv != base:
                    candidates.add(cv)
            # Small values
            for cv in range(1, 30):
                if cv != base:
                    candidates.add(cv)
            
            # Score each candidate by how many collisions it creates
            best_candidate = None
            best_gain = 0
            for cv in candidates:
                if cv in vals:
                    continue
                # Count how many edges from idx would have sum in sum_set
                sum_hits = 0
                prod_hits = 0
                for nb in neighbors:
                    s = cv + vals[nb]
                    p = cv * vals[nb]
                    if s in sum_set:
                        sum_hits += 1
                    if p in prod_set:
                        prod_hits += 1
                gain = sum_hits + prod_hits
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = cv
            
            if best_candidate is not None and best_gain > 0:
                new_vals = list(vals)
                new_vals[idx] = best_candidate
                new_score = score_of(new_vals, elist)
                if new_score < cur_score:
                    vals = new_vals
                    cur_score = new_score
                    improvements += 1
        return cur_score, vals

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Phase 1: Structured initialization with degree-aware value assignment
    # Try multiple initial edge structures
    init_structures = []
    
    # Structure 1: Sum-sorted edges on AP values
    ap_vals = list(range(1, n + 1))
    edge_order = sorted(all_edges, key=lambda e: ap_vals[e[0]] + ap_vals[e[1]])
    init_edges = edge_order[:m]
    init_score = score_of(ap_vals, init_edges)
    best_score = init_score
    best_vals = list(ap_vals)
    best_edges = set(init_edges)
    init_structures.append((list(ap_vals), set(init_edges)))
    
    # Structure 2: Random 3-regular-ish graph
    random.seed(42)
    for trial in range(5):
        # Generate a random sparse graph
        deg = [0] * n
        cand_edges = []
        for _ in range(m):
            while True:
                u, v = random.sample(range(n), 2)
                e = (min(u, v), max(u, v))
                if e not in all_edges_set and deg[u] < 5 and deg[v] < 5:
                    cand_edges.append(e)
                    deg[u] += 1
                    deg[v] += 1
                    break
        if len(cand_edges) == m:
            # Assign values based on degree: higher degree gets smaller values
            # to create more sum collisions
            deg_sorted = sorted(range(n), key=lambda i: deg[i], reverse=True)
            vals = [0] * n
            # Use small values for high-degree vertices
            for rank, vidx in enumerate(deg_sorted):
                vals[vidx] = rank + 1
            # Ensure distinct
            if len(set(vals)) != n:
                # Fix duplicates
                used = set()
                for i in range(n):
                    while vals[i] in used or vals[i] < 1 or vals[i] > 10000:
                        vals[i] = random.randint(1, 50)
                    used.add(vals[i])
            sc = score_of(vals, cand_edges)
            if sc < best_score:
                best_score = sc
                best_vals = list(vals)
                best_edges = set(cand_edges)
            init_structures.append((list(vals), set(cand_edges)))
    
    # Structure 3: AP variants
    for start in [1, 2, 3, 5]:
        for step in [1, 2, 3]:
            ap = [start + i * step for i in range(n)]
            if len(set(ap)) != n or max(ap) > 10000 or min(ap) < 1:
                continue
            eo = sorted(all_edges, key=lambda e: ap[e[0]] + ap[e[1]])
            ie = eo[:m]
            sc = score_of(ap, ie)
            if sc < best_score:
                best_score = sc
                best_vals = list(ap)
                best_edges = set(ie)
            init_structures.append((list(ap), set(ie)))
    
    # Phase 2: Joint search from multiple starts
    random.seed(42)
    starts = init_structures[:4]  # Use up to 4 best starts
    
    remaining = deadline - time.time() - 6.0
    if remaining > 0:
        per = max(3000, int(remaining / len(starts) / 0.00008))
        for vals0, edges0 in starts:
            if time.time() > deadline - 6.0:
                break
            sc, vals, edges = joint_search(vals0, edges0, per)
            if sc < best_score:
                best_score = sc
                best_vals = vals
                best_edges = edges
    
    # Phase 3: Collision-seeking polish
    if time.time() < deadline - 4.0:
        elist = list(best_edges)
        sc, vals = collision_seeking_move(best_vals, best_edges, elist)
        if sc < best_score:
            best_score = sc
            best_vals = vals
    
    # Phase 4: Polish edges
    if time.time() < deadline - 3.0:
        sc, elist, eset = polish_edges(best_vals, best_edges, budget=25000)
        if sc < best_score:
            best_score = sc
            best_edges = eset
    
    # Phase 5: Polish values
    if time.time() < deadline - 2.0:
        sc, vals = polish_values(best_vals, best_edges, budget=15000)
        if sc < best_score:
            best_score = sc
            best_vals = vals
    
    # Phase 6: Another collision-seeking pass
    if time.time() < deadline - 1.5:
        elist = list(best_edges)
        sc, vals = collision_seeking_move(best_vals, best_edges, elist)
        if sc < best_score:
            best_score = sc
            best_vals = vals
    
    # Phase 7: Final edge polish
    if time.time() < deadline - 1.0:
        sc, elist, eset = polish_edges(best_vals, best_edges, budget=10000)
        if sc < best_score:
            best_score = sc
            best_edges = eset
    
    # Fallback
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = set(all_edges[:m])
        best_score = score_of(best_vals, list(best_edges))

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    return {'values': final_values, 'edges': final_edges}
