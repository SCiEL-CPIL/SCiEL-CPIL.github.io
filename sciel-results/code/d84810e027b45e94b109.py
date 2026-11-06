"""Rewritten program: clique partition via max-clique extraction (B&B on complement) with greedy fallback."""
import time
from random import Random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def get_graph(n, p, seed):
        rng = Random(seed)
        # Build adjacency as a set of frozensets for edge lookup, and a list of edges
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def greedy_clique(adj, candidates):
        """Greedy maximal clique: repeatedly pick vertex with max degree within candidates."""
        if not candidates:
            return []
        candidates = set(candidates)
        clique = []
        while candidates:
            # Pick vertex with max degree within current candidates
            best_v = max(candidates, key=lambda v: len(candidates & adj[v]))
            clique.append(best_v)
            candidates = candidates & adj[best_v]
        return clique

    def max_clique_branch_and_bound(adj, candidates, time_limit=5.0):
        """Find a maximum clique in the subgraph induced by candidates using B&B.
        If time exceeds time_limit, return the best found so far."""
        t0 = time.time()
        candidates = set(candidates)
        if not candidates:
            return []
        
        # Greedy initial solution
        best_clique = greedy_clique(adj, candidates)
        best_size = len(best_clique)
        
        # Upper bound: for a set of vertices, the max clique size <= |candidates|
        # Better bound: greedy coloring bound
        def greedy_color_bound(cand_set):
            """Return an upper bound on clique size using greedy coloring of the subgraph."""
            if not cand_set:
                return 0
            cand_list = list(cand_set)
            # Order by degree descending
            cand_list.sort(key=lambda v: -len(cand_set & adj[v]))
            colors = {}
            color_count = 0
            for v in cand_list:
                # Find smallest color not used by neighbors in cand_set
                used_colors = set()
                for nb in cand_set & adj[v]:
                    if nb in colors:
                        used_colors.add(colors[nb])
                c = 0
                while c in used_colors:
                    c += 1
                colors[v] = c
                color_count = max(color_count, c + 1)
            return color_count
        
        # B&B
        import sys
        sys.setrecursionlimit(10000)
        
        def bb(cand_set, current_clique):
            nonlocal best_clique, best_size
            if time.time() - t0 > time_limit:
                return
            if not cand_set:
                if len(current_clique) > best_size:
                    best_clique = list(current_clique)
                    best_size = len(best_clique)
                return
            # Upper bound
            ub = greedy_color_bound(cand_set)
            if len(current_clique) + ub <= best_size:
                return
            # Pick vertex with max degree in cand_set
            v = max(cand_set, key=lambda x: len(cand_set & adj[x]))
            # Include v
            bb(cand_set & adj[v], current_clique | {v})
            # Exclude v
            cand_set2 = cand_set - {v}
            bb(cand_set2, current_clique)
        
        bb(candidates, frozenset())
        return best_clique

    def solve_instance(n, p, seed, time_budget):
        edges, adj = get_graph(n, p, seed)
        if not edges:
            return []
        
        remaining_edges = set(edges)
        cliques = []
        
        while remaining_edges:
            if time.time() - start_time > DEADLINE - 5.0:
                break
            
            # Find a vertex that is in some remaining edge
            # Pick the vertex with highest degree in remaining edges
            deg_in_remaining = {v: 0 for v in range(n)}
            for u, v in remaining_edges:
                deg_in_remaining[u] += 1
                deg_in_remaining[v] += 1
            
            # Pick vertex with max degree
            if not any(deg_in_remaining.values()):
                break
            pivot = max(range(n), key=lambda v: deg_in_remaining[v])
            
            # Candidates: pivot + its neighbors that share an edge with pivot in remaining_edges
            candidates = {pivot}
            for u, v in remaining_edges:
                if u == pivot:
                    candidates.add(v)
                elif v == pivot:
                    candidates.add(u)
            
            # Find max clique among candidates (all must be pairwise adjacent in original graph)
            # Use B&B with time limit
            time_limit_for_bb = min(3.0, max(0.1, DEADLINE - 10.0 - (time.time() - start_time)))
            if time_limit_for_bb > 0.05:
                clique = max_clique_branch_and_bound(adj, candidates, time_limit=time_limit_for_bb)
            else:
                clique = greedy_clique(adj, candidates)
            
            if not clique:
                # Fallback: just take pivot and one neighbor
                for u, v in remaining_edges:
                    if u == pivot:
                        clique = [pivot, v]
                        break
                    elif v == pivot:
                        clique = [pivot, u]
                        break
                if not clique:
                    break
            
            # Remove all edges within this clique
            clique_set = set(clique)
            new_remaining = set()
            for e in remaining_edges:
                if e[0] in clique_set and e[1] in clique_set:
                    continue
                new_remaining.add(e)
            remaining_edges = new_remaining
            cliques.append(clique)
        
        # If we ran out of time, handle remaining edges with greedy pairs
        if remaining_edges and time.time() - start_time > DEADLINE - 2.0:
            # Just pair up edges greedily
            for u, v in remaining_edges:
                cliques.append([u, v])
        
        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        time_budget = DEADLINE - (time.time() - start_time) - 5.0
        if time_budget <= 0:
            # Emergency: return trivial edge partition
            edges, _ = get_graph(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = solve_instance(n, p, seed, time_budget)
        results.append(cliques)
    
    return results
