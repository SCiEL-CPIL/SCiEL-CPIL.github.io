import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch(R, P, X, graph, n):
        """
        Enumerates maximal cliques.
        R: current clique
        P: candidates that can extend R
        X: candidates that were in P but are now excluded (to avoid duplicates)
        graph: adjacency list or set of neighbors
        """
        if not P and not X:
            if len(R) >= 2:
                yield frozenset(R)
            return
        
        # Pivot selection: choose vertex in P union X with max degree in P
        # This reduces the number of recursive calls
        pivot_candidates = list(P) + list(X)
        if not pivot_candidates:
            return
            
        # Find pivot with max |P \cap N(p)|
        max_inter = -1
        pivot = pivot_candidates[0]
        for p in pivot_candidates:
            inter = len(P & graph[p])
            if inter > max_inter:
                max_inter = inter
                pivot = p
        
        # Iterate over P \ N(pivot)
        for v in list(P - graph[pivot]):
            new_R = R | {v}
            new_P = P & graph[v]
            new_X = X & graph[v]
            yield from bron_kerbosch(new_R, new_P, new_X, graph, n)
            P.remove(v)
            X.add(v)

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find maximal cliques of size >= 2
        cliques = set()
        # Start with empty R, P = all vertices, X = empty
        # To be efficient, we can start with each vertex? 
        # Standard BK: R={}, P=all, X={}
        # But we want cliques of size >= 2.
        # We can just filter later.
        
        # Optimization: Only start with vertices that have degree >= 1
        # Actually, standard BK handles it.
        
        all_cliques = list(bron_kerbosch(frozenset(), set(range(n)), set(), adj, n))
        # Filter
        valid_cliques = [c for c in all_cliques if len(c) >= 2]
        
        if not valid_cliques:
            return []
            
        # Greedy Hitting Set
        # Map vertex to set of clique indices it hits
        # Map clique index to set of vertices
        # For speed, use sets of vertex IDs directly if number of cliques is small
        # Number of cliques is likely < 10000 for n=40, p=0.2
        
        # Let's use a list of cliques (frozen sets)
        # And for each vertex, the list of cliques it is in
        vertex_cliques = defaultdict(set)
        for i, c in enumerate(valid_cliques):
            for v in c:
                vertex_cliques[v].add(i)
                
        best_set = None
        best_size = float('inf')
        
        # Number of restarts depends on time
        # Estimate cost of one run
        # Greedy is O(V * C) roughly
        # Local search is O(V * C)
        
        # Let's do a few restarts with random tie-breaking
        num_restarts = 100
        rng_solver = random.Random(seed + 42)
        
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Greedy Phase
            hit_cliques = set()
            current_set = set()
            
            # Pre-compute for greedy: for each vertex, the count of unhitted cliques
            # This is dynamic.
            
            while len(hit_cliques) < len(valid_cliques):
                best_v = -1
                max_hits = -1
                candidates = []
                
                # To speed up, we can maintain a count array
                # But V is small (40), C is moderate.
                # Iterating all V is fine.
                
                for v in range(n):
                    if v in current_set:
                        continue
                    # Count how many cliques containing v are not yet hit
                    hits = 0
                    for ci in vertex_cliques[v]:
                        if ci not in hit_cliques:
                            hits += 1
                    if hits > max_hits:
                        max_hits = hits
                        best_v = v
                        candidates = [v]
                    elif hits == max_hits and hits > 0:
                        candidates.append(v)
                        
                if max_hits == 0:
                    # Should not happen if there are unhitted cliques
                    # Unless a clique is empty? No, size >= 2.
                    break
                    
                # Random tie-breaking
                v = rng_solver.choice(candidates)
                current_set.add(v)
                for ci in vertex_cliques[v]:
                    hit_cliques.add(ci)
                    
            # Local Search: 1-removal
            # Try to remove each vertex
            improved = True
            while improved:
                improved = False
                # Shuffle order for randomness?
                vs = list(current_set)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        break
                    # Check if v is essential
                    # v is essential if there is a clique hit ONLY by v
                    # i.e., a clique C such that C \ current_set == {v} ?
                    # No, C \ (current_set - {v}) must be empty.
                    # So, is there a clique C in hit_cliques such that C is subset of {v} U (current_set - {v})?
                    # Simpler: For each clique C containing v, check if C has any other vertex in current_set.
                    # If for ALL cliques C containing v, there is another vertex in current_set, then v can be removed.
                    
                    can_remove = True
                    for ci in vertex_cliques[v]:
                        # If this clique is hit
                        # Check if any other vertex in this clique is in current_set
                        c = valid_cliques[ci]
                        others = [u for u in c if u != v]
                        # If any of 'others' is in current_set, then C is still hit
                        # If none, then removing v unhits C.
                        if not any(u in current_set for u in others):
                            can_remove = False
                            break
                            
                    if can_remove:
                        current_set.remove(v)
                        improved = True
                        
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
        return sorted(best_set) if best_set else []

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return empty or minimal valid?
            # Must be valid.
            # If we run out of time, we should return the best found so far or a safe default.
            # The safe default is all vertices with degree > 0.
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
            
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results
