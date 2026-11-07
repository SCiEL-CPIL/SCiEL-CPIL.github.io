import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 150.0
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency for K and cross edges
        # K is 0..k-1, I is k..n-1
        # N[v] for v in I is set of neighbors in K
        N = [set() for _ in range(n)]
        for u, v in edges:
            if u < k and v < k:
                # internal edge, ignore for N
                pass
            elif u < k and v >= k:
                N[v].add(u)
            elif u >= k and v < k:
                N[u].add(v)
            else:
                # both in I, should not happen
                pass
        
        cliques = []
        
        # Identify V_good and V_bad
        V_good = []
        V_bad = []
        for v in range(k, n):
            if len(N[v]) >= 2:
                V_good.append(v)
            elif len(N[v]) == 1:
                V_bad.append(v)
            
        # Handle V_bad: add {u, v}
        for v in V_bad:
            u = next(iter(N[v]))
            cliques.append([u, v])
        
        # Handle V_good: Select subset S with edge-disjoint N(v) in K
        # Greedy: sort by size of N(v) descending
        V_good.sort(key=lambda v: len(N[v]), reverse=True)
        
        selected = []
        covered_edges_in_K = set()
        
        for v in V_good:
            # Check if N[v] edges are disjoint from covered_edges_in_K
            n_v = N[v]
            conflict = False
            for u1, u2 in combinations(sorted(n_v), 2):
                edge = (min(u1, u2), max(u1, u2))
                if edge in covered_edges_in_K:
                    conflict = True
                    break
            if not conflict:
                selected.append(v)
                for u1, u2 in combinations(sorted(n_v), 2):
                    edge = (min(u1, u2), max(u1, u2))
                    covered_edges_in_K.add(edge)
        
        # Add cliques for selected V_good
        for v in selected:
            clique = [v] + list(N[v])
            cliques.append(clique)
        
        # Handle remaining edges in K
        # All edges in K
        all_K_edges = set()
        for u1 in range(k):
            for u2 in range(u1 + 1, k):
                all_K_edges.add((u1, u2))
        
        rem_edges = all_K_edges - covered_edges_in_K
        
        # Cover rem_edges with maximal cliques greedily
        # Build graph of rem_edges
        # Adjacency list for rem_edges
        adj_rem = {u: set() for u in range(k)}
        for u1, u2 in rem_edges:
            adj_rem[u1].add(u2)
            adj_rem[u2].add(u1)
        
        current_rem_edges = set(rem_edges)
        
        while current_rem_edges:
            # Find a maximal clique in current_rem_edges
            # Start with an arbitrary edge
            u1, u2 = next(iter(current_rem_edges))
            # Expand to maximal clique
            # Greedy expansion: add vertices connected to all in clique
            clique = {u1, u2}
            changed = True
            while changed:
                changed = False
                for u in range(k):
                    if u not in clique:
                        # Check if u is connected to all in clique
                        is_connected = all(w in adj_rem[u] for w in clique)
                        if is_connected:
                            clique.add(u)
                            changed = True
            
            # Add this clique
            cliques.append(list(clique))
            
            # Remove edges within this clique from current_rem_edges
            for u1, u2 in combinations(sorted(clique), 2):
                edge = (min(u1, u2), max(u1, u2))
                if edge in current_rem_edges:
                    current_rem_edges.remove(edge)
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline - 10:
            # Safety break, though it should be fast
            break
        res = solve_instance(*spec)
        # Ensure types are Python int and list
        clean_res = []
        for c in res:
            clean_c = [int(x) for x in c]
            clean_res.append(clean_c)
        results.append(clean_res)
        
    return results
