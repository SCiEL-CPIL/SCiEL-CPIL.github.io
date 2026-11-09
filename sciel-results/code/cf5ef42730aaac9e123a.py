import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Clique part
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    results = []
    
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        # Build adjacency structure for cross edges
        # C is 0..k-1, I is k..n-1
        # adj[i] is a bitmask or set of neighbors in C for i in I
        adj_in_C = [set() for _ in range(n)]
        for u, v in edges:
            if u < k and v >= k:
                adj_in_C[v].add(u)
            elif v < k and u >= k:
                adj_in_C[u].add(v)
        
        # Candidate cliques M_v for v in I
        # M_v = {v} U N(v)
        # Only consider if |M_v| >= 2
        candidates = []
        for v in range(k, n):
            if len(adj_in_C[v]) > 0:
                clique = [v] + list(adj_in_C[v])
                candidates.append((clique, frozenset(clique)))
        
        # Sort candidates by size descending for greedy
        candidates.sort(key=lambda x: len(x[0]), reverse=True)
        
        best_cliques = []
        best_gain = -1
        best_set_mask = 0 # bitmask of vertices in C used by selected M_v
        
        # Try multiple random shuffles to find a better packing
        rng_local = random.Random(seed + 1000)
        
        # Fixed greedy first
        selected = []
        used_C = set()
        for clique, cset in candidates:
            # Check if disjoint from used_C
            # The intersection with C is exactly adj_in_C[v]
            v = clique[0]
            if not (adj_in_C[v] & used_C):
                selected.append(clique)
                used_C.update(adj_in_C[v])
        
        # Calculate gain for this selection
        def calc_gain(selected_cliques, used_C_set):
            gain = 0
            for cl in selected_cliques:
                gain += (len(cl) * (len(cl) - 1) // 2) - 1
            # C_rem
            C_rem = set(range(k)) - used_C_set
            if len(C_rem) >= 2:
                gain += (len(C_rem) * (len(C_rem) - 1) // 2) - 1
            return gain

        current_gain = calc_gain(selected, used_C)
        
        # Try random shuffles
        for _ in range(20):
            if time.time() > deadline:
                break
            shuffled = candidates[:]
            rng_local.shuffle(shuffled)
            sel = []
            used = set()
            for clique, cset in shuffled:
                v = clique[0]
                if not (adj_in_C[v] & used):
                    sel.append(clique)
                    used.update(adj_in_C[v])
            g = calc_gain(sel, used)
            if g > current_gain:
                current_gain = g
                selected = sel
                used_C = used
        
        # Finalize selected cliques
        final_cliques = list(selected)
        
        # Add C_rem if possible
        C_rem = set(range(k)) - used_C
        if len(C_rem) >= 2:
            final_cliques.append(list(C_rem))
        
        # Determine covered edges
        # An edge is covered if both endpoints are in the same clique in final_cliques
        # We need to list all edges NOT covered, and add them as individual cliques.
        
        # Efficiently find uncovered edges:
        # 1. Edges within C:
        #    Covered if both endpoints in C_rem (and C_rem is in final_cliques) OR
        #    both endpoints in N(v) for some selected v?
        #    Actually, if u, v in C, they are covered if there exists a selected M_w such that u, v in N(w).
        #    This is equivalent to: u, v in used_C AND there exists w in selected such that u, v in N(w).
        #    Wait, if u, v in used_C, they are in the union of N(w)'s.
        #    Are they in the SAME N(w)?
        #    If u in N(w1) and v in N(w2) with w1 != w2, and w1, w2 are disjoint in C, then u != v and u not in N(w2), v not in N(w1).
        #    So u, v cannot be in the same M_w if they come from different selected M's.
        #    So edges within C are covered ONLY if both endpoints are in N(w) for the SAME selected w.
        #    Or if both are in C_rem (and C_rem is added).
        
        # 2. Edges between I and C:
        #    Covered if the I-vertex is in a selected M_w and the C-vertex is in N(w).
        #    This is exactly the edges in the selected M_w's.
        
        # 3. Edges within I:
        #    None exist (I is independent).
        
        # So, uncovered edges are:
        # - Cross edges where I-vertex is NOT in any selected M_w.
        # - Cross edges where I-vertex IS in selected M_w but C-vertex is NOT in N(w) (impossible by definition).
        # - Edges within C where both endpoints are in used_C but NOT in the same N(w).
        # - Edges within C where one or both endpoints are in C_rem (if C_rem not added? No, we add it if size>=2).
        #   If C_rem size < 2, edges within C_rem (if any) are uncovered?
        #   If C_rem size 1, no edges within C_rem.
        #   If C_rem size 0, no edges.
        
        # Let's just iterate through all edges and check if they are covered.
        # n is small (<=90), E is small.
        
        covered = set()
        for cl in final_cliques:
            for u, v in combinations(cl, 2):
                covered.add((min(u, v), max(u, v)))
        
        uncovered_edges = []
        for e in edges:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key not in covered:
                uncovered_edges.append(list(e))
        
        final_cliques.extend(uncovered_edges)
        
        # Validate: all cliques size >= 2
        # final_cliques from selected have size >= 2 (checked in candidates)
        # C_rem added only if size >= 2
        # uncovered_edges are size 2
        
        results.append(final_cliques)
        
    return results
