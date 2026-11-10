import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch(adj, P, X, C, maximal_cliques):
        if not P and not X:
            if len(C) >= 2:
                maximal_cliques.append(frozenset(C))
            return
        
        P_union_X = P | X
        if P_union_X:
            # Pivot selection: choose vertex u in P union X with max degree in P
            pivot = None
            max_inter = -1
            for v in P_union_X:
                inter = len(P & adj[v])
                if inter > max_inter:
                    max_inter = inter
                    pivot = v
            
            # Branch on P \ N(pivot)
            candidates = P - adj[pivot] if pivot is not None else P
            
            for v in list(candidates):
                new_P = P & adj[v]
                new_X = X & adj[v]
                new_C = C | {v}
                bron_kerbosch(adj, new_P, new_X, new_C, maximal_cliques)
                P.remove(v)
                X.add(v)

    def find_min_transversal(n, edges):
        if not edges:
            return []
        
        adj = build_adj(n, edges)
        maximal_cliques = []
        # Bron-Kerbosch
        bron_kerbosch(adj, set(range(n)), set(), set(), maximal_cliques)
        
        if not maximal_cliques:
            return []

        n_cliques = len(maximal_cliques)
        
        # Map each clique to an index
        # Precompute: for each vertex, which cliques does it hit?
        vertex_cliques = [[] for _ in range(n)]
        for i, c in enumerate(maximal_cliques):
            for v in c:
                vertex_cliques[v].append(i)
        
        # Precompute bitmask for each vertex
        vertex_clique_mask = [0] * n
        for i in range(n_cliques):
            bit = 1 << i
            for v in maximal_cliques[i]:
                vertex_clique_mask[v] |= bit
        
        # Greedy initialization with better heuristic
        unhit_mask = (1 << n_cliques) - 1  # All cliques unhit
        transversal = set()
        
        while unhit_mask:
            if time.time() > deadline:
                break
            # Count frequency of each vertex in unhit cliques
            freq = [0] * n
            # Iterate through unhit cliques using bitmask
            m = unhit_mask
            while m:
                # Get lowest set bit
                lowbit = m & (-m)
                idx = lowbit.bit_length() - 1
                m ^= lowbit
                for v in maximal_cliques[idx]:
                    freq[v] += 1
            
            if not any(freq):
                break
                
            best_v = None
            best_score = None
            for v in range(n):
                f = freq[v]
                if f == 0:
                    continue
                # Score: frequency is primary. Degree is secondary.
                score = (f, len(adj[v]), -v)
                if best_score is None or score > best_score:
                    best_score = score
                    best_v = v
            
            transversal.add(best_v)
            
            # Remove all cliques containing best_v from unhit_mask
            unhit_mask &= ~vertex_clique_mask[best_v]

        # Ensure validity if broken by time
        if unhit_mask:
            m = unhit_mask
            while m:
                lowbit = m & (-m)
                idx = lowbit.bit_length() - 1
                m ^= lowbit
                transversal.add(next(iter(maximal_cliques[idx])))
        
        # Greedy removal
        changed = True
        while changed:
            changed = False
            for v in list(transversal):
                if time.time() > deadline:
                    return sorted(transversal)
                redundant = True
                for ci in vertex_cliques[v]:
                    c = maximal_cliques[ci]
                    others = c - {v}
                    if not (others & transversal):
                        redundant = False
                        break
                if redundant:
                    transversal.remove(v)
                    changed = True

        # Branch and Bound for minimum hitting set
        # Use recursive search with pruning
        best_size = len(transversal)
        best_solution = set(transversal)
        
        # Precompute for each vertex: bitmask of cliques it hits
        vertex_clique_mask = [0] * n
        for i in range(n_cliques):
            bit = 1 << i
            for v in maximal_cliques[i]:
                vertex_clique_mask[v] |= bit
        
        # Precompute clique membership as frozensets for fast intersection
        clique_frozensets = [frozenset(c) for c in maximal_cliques]
        
        def search(unhit_mask, current_set, current_size, remaining_budget):
            nonlocal best_size, best_solution
            
            if current_size >= best_size:
                return
            
            if not unhit_mask:
                best_size = current_size
                best_solution = set(current_set)
                return
            
            if time.time() > remaining_budget:
                return
            
            # Lower bound: at least 1 if unhit_mask exists
            if current_size + 1 >= best_size:
                return
            
            # Find a clique to branch on (choose one with fewest candidate vertices)
            min_cand_count = float('inf')
            best_clique_idx = -1
            best_candidates = []
            
            m = unhit_mask
            while m:
                lowbit = m & (-m)
                idx = lowbit.bit_length() - 1
                m ^= lowbit
                
                c = maximal_cliques[idx]
                cand_count = 0
                for v in c:
                    if v not in current_set:
                        cand_count += 1
                
                if cand_count < min_cand_count:
                    min_cand_count = cand_count
                    best_clique_idx = idx
                    best_candidates = [v for v in c if v not in current_set]
                    if min_cand_count <= 1:
                        break
            
            if not best_candidates:
                return
            
            # Sort candidates by frequency in unhit cliques (descending)
            # Use bitmask to count quickly
            def candidate_score(v):
                mask = vertex_clique_mask[v]
                # Count bits in (mask & unhit_mask)
                return -(mask & unhit_mask).bit_count()
            
            best_candidates.sort(key=candidate_score)
            
            for v in best_candidates:
                if time.time() > remaining_budget:
                    return
                
                # Add v
                new_set = current_set | {v}
                
                # Update unhit_mask
                new_unhit = unhit_mask & ~vertex_clique_mask[v]
                
                search(new_unhit, new_set, current_size + 1, remaining_budget)

        # Run B&B with time limit - allocate remaining time
        bb_start = time.time()
        bb_budget = deadline - bb_start - 2  # 2s safety margin
        if bb_budget > 0:
            search((1 << n_cliques) - 1, set(), 0, bb_start + bb_budget)
        
        if best_size < len(transversal):
            transversal = best_solution
        
        # Local search for further reduction
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            trans_list = list(transversal)
            n_trans = len(trans_list)
            
            # 1-for-0: try removing single vertices
            for i in range(n_trans):
                v1 = trans_list[i]
                candidate = transversal - {v1}
                # Check if still valid
                valid = True
                for c in maximal_cliques:
                    if not (c & candidate):
                        valid = False
                        break
                if valid:
                    transversal = candidate
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2-for-1: try replacing two vertices with one
            for i in range(n_trans):
                for j in range(i + 1, n_trans):
                    v1 = trans_list[i]
                    v2 = trans_list[j]
                    
                    candidate = transversal - {v1, v2}
                    
                    unhit_after_removal = []
                    for c in maximal_cliques:
                        if not (c & candidate):
                            unhit_after_removal.append(c)
                    
                    if not unhit_after_removal:
                        transversal = candidate
                        improved = True
                        break
                    
                    potential_w = set()
                    for c in unhit_after_removal:
                        potential_w.update(c)
                    
                    found_w = None
                    for w in potential_w:
                        if w == v1 or w == v2:
                            continue
                        ok = True
                        for c in unhit_after_removal:
                            if w not in c:
                                ok = False
                                break
                        if ok:
                            found_w = w
                            break
                    
                    if found_w is not None:
                        transversal = candidate | {found_w}
                        improved = True
                        break
                
                if improved:
                    break
        
        # Final greedy removal
        changed = True
        while changed:
            changed = False
            for v in list(transversal):
                if time.time() > deadline:
                    break
                redundant = True
                for c in maximal_cliques:
                    if v in c:
                        others = c - {v}
                        if not (others & transversal):
                            redundant = False
                            break
                if redundant:
                    transversal.remove(v)
                    changed = True

        return sorted(transversal)

    results = []
    # Allocate time: give more time to harder instances
    # Instance 1: n=30, Instance 2: n=35, Instance 3: n=40
    time_per_instance = [30, 40, 50]  # Total 120s, with safety margin
    
    for idx, (n, p, seed) in enumerate(((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))):
        if time.time() > deadline - 2:
            # Not enough time left, return empty for remaining
            results.append([])
            continue
        edges = instance(n, p, seed)
        sol = find_min_transversal(n, edges)
        results.append(sol)
        if time.time() > deadline - 2:
             results.append([])
             if len(results) < 3:
                 results.append([])
             break

    while len(results) < 3:
        results.append([])
    
    return results
