import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, n):
        """
        Returns a list of maximal cliques (as frozensets of vertices) with size >= 2.
        Uses pivoting to speed up enumeration.
        """
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose vertex u in P U X with max |P \cap N(u)|
            best_pivot = None
            max_count = -1
            for v in P:
                count = len(P.intersection(adj[v]))
                if count > max_count:
                    max_count = count
                    best_pivot = v
            for v in X:
                count = len(P.intersection(adj[v]))
                if count > max_count:
                    max_count = count
                    best_pivot = v
            
            if best_pivot is None:
                # No pivot, iterate over all P
                candidates = list(P)
            else:
                # Iterate over P \ N(best_pivot)
                candidates = [v for v in P if v not in adj[best_pivot]]
            
            for v in candidates:
                new_R = R | {v}
                new_P = P.intersection(adj[v])
                new_X = X.intersection(adj[v])
                bk(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return cliques

    def greedy_cover(cliques, n, rng):
        """
        Greedy set cover: pick vertex in most uncovered cliques.
        """
        # Map vertex to list of indices of cliques it belongs to
        # Or simpler: maintain list of uncovered cliques
        uncovered = list(cliques)
        if not uncovered:
            return set()
            
        # Precompute for each vertex, which cliques it is in?
        # Since cliques are frozensets, we can just iterate.
        # For efficiency with small N, let's just count on the fly or precompute.
        
        # Precompute incidence: vertex -> set of clique indices
        # This might be memory heavy if many cliques, but N=40 is small.
        # Let's just do simple greedy:
        
        selected = set()
        # To speed up, we can track which cliques are still uncovered
        # and for each vertex, how many uncovered cliques it covers.
        
        # Initial counts
        cover_count = {v: 0 for v in range(n)}
        for c in cliques:
            for v in c:
                cover_count[v] += 1
        
        # We need to update counts as we remove cliques.
        # Simpler approach:
        # Loop until all covered.
        remaining_cliques = [c for c in cliques]
        
        while remaining_cliques:
            # Find vertex with max coverage of remaining_cliques
            best_v = -1
            max_cov = -1
            for v in range(n):
                if v in selected:
                    continue
                cov = sum(1 for c in remaining_cliques if v in c)
                if cov > max_cov:
                    max_cov = cov
                    best_v = v
                    # Tie breaking: random if needed?
            
            if best_v == -1:
                break # Should not happen if cliques exist and graph is connected enough, but safe
            
            selected.add(best_v)
            # Remove covered cliques
            remaining_cliques = [c for c in remaining_cliques if best_v not in c]
            
        return selected

    def local_search_improve(selected, cliques, n, rng, iterations=100):
        """
        Try to remove vertices (deletion) and swap vertices.
        """
        selected = set(selected)
        
        def is_valid(sel):
            for c in cliques:
                if not c.intersection(sel):
                    return False
            return True
        
        # 1. Deletion Pass
        improved = True
        while improved:
            improved = False
            verts = list(selected)
            rng.shuffle(verts)
            for v in verts:
                if v not in selected:
                    continue
                new_sel = selected - {v}
                safe = True
                for c in cliques:
                    if v in c:
                        if not c.intersection(new_sel):
                            safe = False
                            break
                if safe:
                    selected = new_sel
                    improved = True
        
        # 2. Swap Pass with critical-clique targeting
        for _ in range(iterations):
            if time.time() > deadline:
                break
            verts_in = list(selected)
            verts_out = [v for v in range(n) if v not in selected]
            if not verts_in or not verts_out:
                break
            
            # Find cliques that are uniquely covered by a single selected vertex
            unique_cover = {v: set() for v in verts_in}
            for c in cliques:
                covering = [v for v in c if v in selected]
                if len(covering) == 1:
                    unique_cover[covering[0]].add(c)
            
            # Find best swap: replace v with u where u covers all of v's critical cliques
            best_swap = None
            best_score = 0
            for v in verts_in:
                critical = unique_cover[v]
                if not critical:
                    continue
                for u in verts_out:
                    if all(u in c for c in critical):
                        # Score: number of critical cliques covered + bonus for extra coverage
                        extra = sum(1 for c in cliques if u in c and v not in c)
                        score = len(critical) + extra * 0.1
                        if score > best_score:
                            best_score = score
                            best_swap = (v, u)
            
            if best_swap:
                v, u = best_swap
                selected = (selected - {v}) | {u}
                # Run deletion pass again after swap
                improved = True
                while improved:
                    improved = False
                    verts = list(selected)
                    rng.shuffle(verts)
                    for v2 in verts:
                        if v2 not in selected:
                            continue
                        new_sel = selected - {v2}
                        safe = True
                        for c in cliques:
                            if v2 in c:
                                if not c.intersection(new_sel):
                                    safe = False
                                    break
                        if safe:
                            selected = new_sel
                            improved = True
            else:
                break
        
        return selected

    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return empty or best known? 
            # Better to return a valid solution if possible, or just stop.
            # If we haven't started this instance, we might return a large valid set (all vertices)
            # to ensure validity, though score will be bad.
            results.append(list(range(n)))
            continue
            
        adj = generate_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        
        # If no cliques of size >= 2, empty set is valid
        if not cliques:
            results.append([])
            continue
            
        best_solution = None
        best_size = float('inf')
        
        # Multiple random restarts for greedy + local search
        # Limit restarts based on time
        max_restarts = 20
        for i in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            rng = random.Random(seed + i)
            # Greedy
            sol = greedy_cover(cliques, n, rng)
            # Local Search
            sol = local_search_improve(sol, cliques, n, rng, iterations=50)
            
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
                
        if best_solution is None:
            # Fallback: all vertices
            best_solution = set(range(n))
            
        results.append(sorted(best_solution))

    return results
# EVOLVE-BLOCK-END
