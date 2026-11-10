import time
import random
from itertools import combinations

def construct():
    """
    Finds minimum vertex transversals for maximal cliques (size >= 2) 
    in three specific random graphs.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Graph Generation
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    # 2. Bron-Kerbosch for Maximal Cliques
    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P union X to minimize recursion
        # Standard heuristic: pick u that maximizes |P \cap N(u)|
        U = P.union(X)
        if not U:
            u = None
        else:
            # Pick arbitrary u from U, or the one with max neighbors in P
            # For small graphs, arbitrary is often fine, but let's try max degree in P
            u = max(U, key=lambda x: len(adj[x].intersection(P)))
        
        # Recurse on P \ N(u)
        for v in list(P.difference(adj[u])):
            bron_kerbosch(adj, R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]), cliques)
            P.remove(v)
            X.add(v)

    def get_maximal_cliques(adj):
        n = len(adj)
        cliques = []
        P = set(range(n))
        X = set()
        bron_kerbosch(adj, set(), P, X, cliques)
        return cliques

    # 3. Set Cover Solver (Minimum Vertex Transversal)
    def solve_transversal(cliques, n):
        if not cliques:
            return []
        
        # Preprocess: Convert cliques to list of sets for easy manipulation
        # We want to cover all cliques.
        
        # Greedy Initialization
        def greedy_cover(cliques):
            uncovered = [set(c) for c in cliques]
            solution = []
            # Count frequency of each vertex in uncovered cliques
            # Actually, just pick vertex covering most uncovered cliques
            while uncovered:
                # Find vertex covering most uncovered cliques
                best_v = -1
                best_count = -1
                for v in range(n):
                    count = 0
                    for c in uncovered:
                        if v in c:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_v = v
                
                if best_count == 0:
                    # Should not happen if cliques are non-empty and vertices exist
                    break
                
                solution.append(best_v)
                
                # Remove all cliques containing best_v
                new_uncovered = []
                for c in uncovered:
                    if best_v not in c:
                        new_uncovered.append(c)
                uncovered = new_uncovered
                
                # Early exit if time is tight
                if time.time() > deadline:
                    return solution
            
            return solution

        initial_solution = greedy_cover(cliques)
        best_size = len(initial_solution)
        best_solution = initial_solution
        
        # Branch and Bound
        # State: uncovered cliques (list of sets), current path
        # We use a list of sets for uncovered cliques.
        
        # Precompute for each vertex, which cliques it covers?
        # It's dynamic, so we just filter.
        
        def lower_bound(uncovered_cliques):
            # Simple lower bound: number of disjoint cliques
            # Greedy set packing:
            remaining = [set(c) for c in uncovered_cliques]
            lb = 0
            while remaining:
                # Pick smallest clique
                remaining.sort(key=len)
                c = remaining.pop(0)
                lb += 1
                # Remove all cliques that intersect c
                new_remaining = []
                for rc in remaining:
                    if not rc.intersection(c):
                        new_remaining.append(rc)
                remaining = new_remaining
            return lb

        def branch_and_bound(uncovered_cliques, current_size, current_solution):
            nonlocal best_size, best_solution
            
            if time.time() > deadline:
                return
                
            if not uncovered_cliques:
                if current_size < best_size:
                    best_size = current_size
                    best_solution = list(current_solution)
                return
                
            # Pruning
            if current_size + lower_bound(uncovered_cliques) >= best_size:
                return
                
            # Heuristic: Pick the uncovered clique with the fewest vertices
            # This is the "most constrained" clique
            # If multiple, pick one with vertices that cover many other cliques?
            # Let's just pick the smallest clique.
            
            # Sort cliques by size to find smallest quickly
            # Since we modify the list, we might want to find min index
            min_idx = 0
            min_len = len(uncovered_cliques[0])
            for i in range(1, len(uncovered_cliques)):
                l = len(uncovered_cliques[i])
                if l < min_len:
                    min_len = l
                    min_idx = i
                    if l == 1: break # Can't get smaller than 1 for size>=2 cliques? 
                                     # Wait, cliques are size >= 2. But uncovered set might be smaller?
                                     # No, the clique set itself doesn't change size, only which ones are uncovered.
            
            target_clique = uncovered_cliques[min_idx]
            
            # Order vertices in target_clique by how many other uncovered cliques they cover
            # Higher coverage first (likely to lead to better solution sooner)
            def coverage_count(v):
                count = 0
                for c in uncovered_cliques:
                    if v in c:
                        count += 1
                return count
                
            vertices_to_try = list(target_clique)
            vertices_to_try.sort(key=coverage_count, reverse=True)
            
            for v in vertices_to_try:
                # Branch: select v
                new_uncovered = []
                for c in uncovered_cliques:
                    if v not in c:
                        new_uncovered.append(c)
                
                current_solution.append(v)
                branch_and_bound(new_uncovered, current_size + 1, current_solution)
                current_solution.pop()
                
                if time.time() > deadline:
                    return

        # Start B&B
        # Initial uncovered is all cliques
        initial_uncovered = [set(c) for c in cliques]
        branch_and_bound(initial_uncovered, 0, [])
        
        return best_solution

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback if we run out of time
            # Return greedy solution or all vertices
            adj = random_graph(n, p, seed)
            touched = {x for u in range(n) for v in adj[u] for x in (u, v)}
            results.append(sorted(touched))
            continue
            
        adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        # If no cliques of size >= 2, empty set is valid
        if not cliques:
            results.append([])
            continue
            
        solution = solve_transversal(cliques, n)
        results.append(sorted(solution))

    return results
