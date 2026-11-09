import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        
        # Internal C edges
        internal_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                if v in adj[u]:
                    internal_edges.add((u, v))
        
        # Edges from I to C
        i_c_edges = {}
        for v in I:
            nbrs = adj[v] & C_set
            i_c_edges[v] = nbrs
        
        # Group I vertices by their exact neighborhood in C
        groups = defaultdict(list)
        for v in I:
            if i_c_edges[v]:
                groups[frozenset(i_c_edges[v])].append(v)
        
        # Sort groups by shadow size descending
        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[0]))
        
        best_partition = None
        best_count = float('inf')
        
        # Run multiple strategies to find the best partition
        num_strategies = 5
        
        for strategy_idx in range(num_strategies):
            if time.time() > deadline - 8.0:
                break
                
            partition = []
            used_c_edges = set()
            
            # Different group orderings for different strategies
            if strategy_idx == 0:
                group_order = sorted_groups  # Descending shadow size
            elif strategy_idx == 1:
                group_order = sorted(sorted_groups, key=lambda x: len(x[0]))  # Ascending
            elif strategy_idx == 2:
                # Sort by number of I vertices descending, then shadow size
                group_order = sorted(sorted_groups, key=lambda x: (-len(x[1]), -len(x[0])))
            elif strategy_idx == 3:
                # Sort by ratio of I vertices to shadow size
                group_order = sorted(sorted_groups, key=lambda x: -(len(x[1]) / max(1, len(x[0]))))
            else:
                # Descending shadow size again (default)
                group_order = sorted_groups
            
            for shadow, i_vertices in group_order:
                if not i_vertices:
                    continue
                
                available = list(shadow)
                
                if not available:
                    continue
                
                def get_unused_degree(u, avail_set):
                    cnt = 0
                    for w in avail_set:
                        if u != w:
                            e = (min(u, w), max(u, w))
                            if e in internal_edges and e not in used_c_edges:
                                cnt += 1
                    return cnt
                
                # Sort available by unused degree descending
                available.sort(key=lambda u: -get_unused_degree(u, set(available)))
                
                clique = []
                for u in available:
                    ok = True
                    for w in clique:
                        e = (min(u, w), max(u, w))
                        if e not in internal_edges or e in used_c_edges:
                            ok = False
                            break
                    if ok:
                        clique.append(u)
                
                if len(clique) >= 1:
                    new_clique = sorted(i_vertices + clique)
                    partition.append(new_clique)
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            used_c_edges.add((min(clique[i], clique[j]), max(clique[i], clique[j])))
                
                rem_i = list(i_vertices)
                rem_c = list(shadow - set(clique))
                
                while rem_i and rem_c:
                    if time.time() > deadline - 4.0:
                        break
                    
                    avail_c = rem_c
                    avail_c.sort(key=lambda u: -get_unused_degree(u, set(avail_c)))
                    
                    c_clique = []
                    for u in avail_c:
                        ok = True
                        for w in c_clique:
                            e = (min(u, w), max(u, w))
                            if e not in internal_edges or e in used_c_edges:
                                ok = False
                                break
                        if ok:
                            c_clique.append(u)
                    
                    if not c_clique:
                        c_clique = [rem_c[0]]
                        
                    new_clique = sorted(rem_i + c_clique)
                    partition.append(new_clique)
                    
                    for i_idx in range(len(c_clique)):
                        for j_idx in range(i_idx + 1, len(c_clique)):
                            used_c_edges.add((min(c_clique[i_idx], c_clique[j_idx]), max(c_clique[i_idx], c_clique[j_idx])))
                    
                    rem_c = [c for c in rem_c if c not in c_clique]
                
            # Cover remaining C-C edges using improved greedy clique packing
            rem_set = internal_edges - used_c_edges
            
            while rem_set:
                if time.time() > deadline - 2.0:
                    break
                
                # Build a degree map for the remaining graph
                rem_deg = {}
                for (a, b) in rem_set:
                    rem_deg[a] = rem_deg.get(a, 0) + 1
                    rem_deg[b] = rem_deg.get(b, 0) + 1
                
                # Find the best edge to start with: maximize (deg[a] + deg[b] + common_neighbors_in_rem)
                best_edge = None
                best_score = -1
                for (a, b) in rem_set:
                    # Count common neighbors in the remaining graph
                    common = 0
                    for c in range(k):
                        if c == a or c == b:
                            continue
                        if (min(a,c), max(a,c)) in rem_set and (min(b,c), max(b,c)) in rem_set:
                            common += 1
                    score = rem_deg.get(a, 0) + rem_deg.get(b, 0) + common
                    if score > best_score:
                        best_score = score
                        best_edge = (a, b)
                
                if best_edge is None:
                    break
                
                a, b = best_edge
                clique = [a, b]
                
                # Greedily expand the clique by adding vertices with highest remaining degree
                # that are connected to all current clique members
                while True:
                    candidates = []
                    for c in range(k):
                        if c in clique:
                            continue
                        ok = True
                        for u in clique:
                            if (min(u,c), max(u,c)) not in rem_set:
                                ok = False
                                break
                        if ok:
                            candidates.append(c)
                    
                    if not candidates:
                        break
                    
                    # Pick the candidate with highest remaining degree
                    best_cand = max(candidates, key=lambda c: rem_deg.get(c, 0))
                    clique.append(best_cand)
                
                if len(clique) >= 2:
                    partition.append(sorted(clique))
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            rem_set.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))
                    
                    # Update remaining degrees
                    for u in clique:
                        rem_deg.pop(u, None)
                    for (a, b) in list(rem_set):
                        rem_deg[a] = rem_deg.get(a, 0) - 1
                        rem_deg[b] = rem_deg.get(b, 0) - 1
                    # Remove zero/negative degree entries
                    rem_deg = {u: d for u, d in rem_deg.items() if d > 0}
            
            cnt = len(partition)
            if cnt < best_count:
                best_count = cnt
                best_partition = partition
        
        if best_partition is None:
            return []
        
        cleaned = []
        for clique in best_partition:
            cleaned.append([int(x) for x in clique])
        return cleaned

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        results.append(res)

    return results
