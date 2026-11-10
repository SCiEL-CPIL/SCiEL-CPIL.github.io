import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose u in P union X with max degree in P
            # If P is empty, pick from X? Standard BK picks from P union X.
            # If P is empty, no cliques can be formed from P, but we check X.
            # Actually, if P is empty, we return after checking R.
            # So pivot should be chosen from P if P is not empty, else from X?
            # Standard: u = argmax_{u in P union X} |P intersect N(u)|
            
            P_X = P.union(X)
            if not P_X:
                return
            
            # Choose pivot
            # For small n, we can just iterate or pick first. 
            # To be efficient, pick one with max neighbors in P.
            max_deg = -1
            pivot = None
            for u in P_X:
                deg = len(P.intersection(adj[u]))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            # Iterate over P \ N(pivot)
            for v in list(P - adj[pivot]):
                bk(R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
                P.remove(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj)
        
        # Filter cliques of size >= 2 (already done in bk, but double check)
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []
            
        # Precompute vertex to cliques mapping
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
                
        best_solution = None
        best_size = float('inf')
        
        # Randomized Greedy
        # Number of restarts
        restarts = 20
        
        for r in range(restarts):
            if time.time() > deadline - 5.0:
                break
                
            uncovered = set(range(len(cliques)))
            solution = set()
            
            # To randomize tie-breaking, we can shuffle vertices or add noise to counts
            # We will shuffle the order of vertices considered for tie-breaking
            
            while uncovered:
                best_v = -1
                best_count = -1
                # Iterate vertices
                # We can optimize by only considering vertices that cover at least one uncovered clique
                candidates = []
                for v in range(n):
                    if v in solution:
                        continue
                    count = 0
                    for ci in vertex_to_cliques[v]:
                        if ci in uncovered:
                            count += 1
                    if count > 0:
                        candidates.append((count, v))
                        
                if not candidates:
                    break # Should not happen if problem is valid
                    
                # Pick best
                # If ties, pick random
                max_count = max(c[0] for c in candidates)
                tied = [c[1] for c in candidates if c[0] == max_count]
                v = random.choice(tied)
                
                solution.add(v)
                
                # Update uncovered
                for ci in vertex_to_cliques[v]:
                    uncovered.discard(ci)
                    
            # Local Search / Redundancy Removal
            improved = True
            while improved:
                improved = False
                # Try to remove vertices
                # Order: random or by degree
                verts_to_check = list(solution)
                random.shuffle(verts_to_check)
                for v in verts_to_check:
                    if v not in solution:
                        continue
                    # Check if v is redundant
                    redundant = True
                    for ci in vertex_to_cliques[v]:
                        # Check if clique ci is covered by other vertices in solution
                        covered = False
                        for u in cliques[ci]:
                            if u != v and u in solution:
                                covered = True
                                break
                        if not covered:
                            redundant = False
                            break
                    if redundant:
                        solution.remove(v)
                        improved = True
                        
            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = set(solution)
                
        return sorted(best_solution) if best_solution else []

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10.0:
            # Fallback: return all touched vertices
            adj = random_graph(n, p, seed)
            touched = {x for u in range(n) for v in adj[u] for x in (u,v)}
            out.append(sorted(touched))
        else:
            out.append(solve_instance(n, p, seed))
            
    return out
