"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build cross adjacency
        cross_adj = defaultdict(set)  # v in I -> set of u in K
        for u, v in edges:
            if u < k and v >= k:
                cross_adj[v].add(u)
        
        # All K-edges
        all_k_edges = set((u, w) for u in range(k) for w in range(u + 1, k))
        
        i_vertices = [v for v in range(k, n) if cross_adj[v]]
        
        best_cliques = None
        best_count = float('inf')
        
        # Try multiple strategies
        for trial in range(300):
            if time.time() > deadline - 10:
                break
            
            current_k_edges = set(all_k_edges)
            cliques = []
            used_i = set()
            
            # Strategy: sort I vertices by some criterion, then greedily assign
            # Different trials use different sorting criteria
            
            def get_order():
                """Return I vertices in a particular order based on trial strategy."""
                if trial % 5 == 0:
                    # Sort by |N_K(v)| descending
                    return sorted(i_vertices, key=lambda v: len(cross_adj[v]), reverse=True)
                elif trial % 5 == 1:
                    # Sort by number of K-edges in N_K(v) descending
                    def k_edge_count(v):
                        nv = cross_adj[v]
                        cnt = 0
                        for u in nv:
                            for w in nv:
                                if u < w and (u, w) in current_k_edges:
                                    cnt += 1
                        return cnt
                    return sorted(i_vertices, key=k_edge_count, reverse=True)
                elif trial % 5 == 2:
                    # Random order
                    order = list(i_vertices)
                    random.shuffle(order)
                    return order
                elif trial % 5 == 3:
                    # Sort by |N_K(v)| ascending (try to save large neighborhoods for later)
                    return sorted(i_vertices, key=lambda v: len(cross_adj[v]))
                else:
                    # Random with bias toward high degree
                    weights = [len(cross_adj[v]) ** 2 for v in i_vertices]
                    order = []
                    remaining = list(i_vertices)
                    while remaining:
                        total_w = sum(weights[i] for i in range(len(remaining)))
                        if total_w == 0:
                            idx = random.randint(0, len(remaining) - 1)
                        else:
                            r = random.random() * total_w
                            cum = 0
                            idx = 0
                            for i in range(len(remaining)):
                                cum += weights[i]
                                if cum >= r:
                                    idx = i
                                    break
                        order.append(remaining.pop(idx))
                    return order
            
            order = get_order()
            
            for v in order:
                if v in used_i:
                    continue
                if time.time() > deadline - 5:
                    break
                
                nv = cross_adj[v]
                if not nv:
                    continue
                
                # Find largest subset S of nv that forms a clique in current_k_edges
                # Greedy: iteratively remove vertices with missing edges
                S = set(nv)
                changed = True
                while changed and len(S) >= 2:
                    changed = False
                    for u in list(S):
                        missing = []
                        for w in S:
                            if u != w and (min(u, w), max(u, w)) not in current_k_edges:
                                missing.append(w)
                        if missing:
                            # Remove u if it has more missing edges than average
                            if len(missing) >= 1:
                                S.discard(u)
                                changed = True
                                break
                
                if len(S) >= 2:
                    clique = sorted(list(S) + [v])
                    cliques.append(clique)
                    used_i.add(v)
                    # Remove K-edges within S
                    S_list = list(S)
                    for i in range(len(S_list)):
                        for j in range(i + 1, len(S_list)):
                            u, w = S_list[i], S_list[j]
                            e = (min(u, w), max(u, w))
                            if e in current_k_edges:
                                current_k_edges.remove(e)
                elif len(S) == 1:
                    u = list(S)[0]
                    cliques.append(sorted([u, v]))
                    used_i.add(v)
                else:
                    for u in nv:
                        cliques.append(sorted([u, v]))
                    used_i.add(v)
            
            # Cover remaining K-edges
            rem_k_edges = current_k_edges
            while rem_k_edges:
                if time.time() > deadline - 3:
                    for e in rem_k_edges:
                        cliques.append(sorted([e[0], e[1]]))
                    rem_k_edges.clear()
                    break
                
                degrees = defaultdict(int)
                adj = defaultdict(set)
                for u, w in rem_k_edges:
                    degrees[u] += 1
                    degrees[w] += 1
                    adj[u].add(w)
                    adj[w].add(u)
                
                if not degrees:
                    break
                
                max_v = max(degrees, key=degrees.get)
                clique = [max_v]
                candidates = list(adj[max_v])
                candidates.sort(key=lambda x: degrees.get(x, 0), reverse=True)
                
                for cand in candidates:
                    ok = True
                    for cv in clique:
                        if (min(cv, cand), max(cv, cand)) not in rem_k_edges:
                            ok = False
                            break
                    if ok:
                        clique.append(cand)
                
                if len(clique) >= 2:
                    cliques.append(sorted(clique))
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            u, w = clique[i], clique[j]
                            e = (min(u, w), max(u, w))
                            if e in rem_k_edges:
                                rem_k_edges.remove(e)
                else:
                    break
            
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = [list(c) for c in cliques]
        
        clean_cliques = []
        for c in best_cliques:
            clean_cliques.append([int(x) for x in c])
        
        return clean_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
