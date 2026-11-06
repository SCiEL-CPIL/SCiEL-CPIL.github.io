import time
import math
import random as _random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = _random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n):
        cliques = []
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Pivot selection for efficiency
            if P:
                best_u = max(P, key=lambda u: len(P & adj[u]))
                candidates = list(P - adj[best_u])
            else:
                candidates = list(P)
            for v in candidates:
                if time.time() - start_time > DEADLINE:
                    return
                R.add(v)
                bk(R, P & adj[v], X & adj[v])
                R.discard(v)
                P.discard(v)
                X.add(v)
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = bron_kerbosch(adj, n)
        if not cliques:
            return []
        
        m = len(cliques)
        # Precompute for each vertex, which cliques it belongs to
        vert_cliques = [set() for _ in range(n)]
        for ci, c in enumerate(cliques):
            for v in c:
                vert_cliques[v].add(ci)
        
        # Identify dominated vertices
        # A vertex u is dominated by v if N(u) subseteq N(v) in the context of the hypergraph
        # i.e., every clique containing u also contains v.
        dominated_by = {}
        for u in range(n):
            if not vert_cliques[u]:
                continue
            for v in range(n):
                if u == v:
                    continue
                if vert_cliques[u] <= vert_cliques[v]:
                    dominated_by[u] = v
                    break
        
        # Heuristic: Greedy Set Cover
        # We want to hit all cliques.
        uncovered = set(range(m))
        result = set()
        
        # Order vertices by frequency (descending) to speed up greedy
        # But we should avoid dominated vertices if possible?
        # Actually, if u is dominated by v, picking v is always at least as good as picking u.
        # So we can restrict candidate set to non-dominated vertices.
        candidates = [v for v in range(n) if v not in dominated_by]
        candidates.sort(key=lambda v: -len(vert_cliques[v]))
        
        while uncovered:
            if time.time() - start_time > DEADLINE:
                break
            best_v = -1
            best_gain = 0
            for v in candidates:
                if v in result:
                    continue
                gain = len(uncovered & vert_cliques[v])
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_v == -1 or best_gain == 0:
                break
            result.add(best_v)
            uncovered -= vert_cliques[best_v]
        
        # Local Search: 1-exchange (remove one if possible, or swap one for one to reduce size? No, size is fixed in swap. We want to reduce size.)
        # Actually, standard local search for minimum set cover often involves:
        # 1. Remove a vertex if it doesn't uncover any new "essential" cliques (i.e., all cliques it covers are covered by others).
        # 2. Swap a vertex v with a set of vertices? Or just try to remove.
        
        # Let's implement a robust local search.
        def local_search(sol):
            sol = set(sol)
            improved = True
            while improved:
                improved = False
                # 1. Try to remove a vertex
                for v in list(sol):
                    # Check if v is essential
                    # v is essential if there is a clique covered by v that is not covered by any other vertex in sol
                    # i.e., for all c in vert_cliques[v], there exists u in sol \ {v} such that c in vert_cliques[u]
                    is_essential = False
                    for c in vert_cliques[v]:
                        covered_by_other = False
                        for u in sol:
                            if u != v and c in vert_cliques[u]:
                                covered_by_other = True
                                break
                        if not covered_by_other:
                            is_essential = True
                            break
                    if not is_essential:
                        sol.discard(v)
                        improved = True
                        break # restart loop as state changed
                # 2. Try to swap a vertex v with a vertex w to reduce size? 
                # Swapping keeps size same. We need size reduction.
                # However, a swap might enable a subsequent removal.
                # Let's try 1-1 swaps to improve "quality" (e.g., cover more cliques with remaining vertices) 
                # but since we just need to hit all, and we already hit all, 
                # a swap that allows a removal is the key.
                # Simpler: just try removing. If stuck, try swapping v with w where w covers all cliques v uniquely covers, 
                # AND w allows another vertex to be removed? That's complex.
                # Let's stick to removal first. It often finds a minimal cover.
                # To get smaller, we might need to swap v for w where w is "better" positioned to allow removal of another vertex.
                # Let's try: for each v in sol, for each w not in sol:
                #   if we replace v with w, can we remove some other vertex?
                # This is expensive. Let's try a simpler 2-exchange:
                # Remove v1, v2 and add w1, w2 such that size decreases? No, 2 for 1 is size decrease.
                # Let's try "Remove 2, Add 1" move.
                if len(sol) > 1:
                    found_move = False
                    # To speed up, we can limit this or do it less frequently
                    for v1 in list(sol):
                        for v2 in list(sol):
                            if v1 == v2:
                                continue
                            # Remove v1, v2
                            temp_sol = sol - {v1, v2}
                            # Find uncovered cliques
                            unc = set()
                            for c in vert_cliques[v1]:
                                if c not in [c2 for u in temp_sol for c2 in vert_cliques[u]]:
                                    unc.add(c)
                            for c in vert_cliques[v2]:
                                if c not in [c2 for u in temp_sol for c2 in vert_cliques[u]]:
                                    unc.add(c)
                            
                            if not unc:
                                # We can remove both!
                                sol = temp_sol
                                improved = True
                                found_move = True
                                break
                            
                            # Can we add one vertex w to cover all unc?
                            for w in range(n):
                                if w in sol:
                                    continue
                                if unc <= vert_cliques[w]:
                                    # Add w
                                    sol = temp_sol | {w}
                                    improved = True
                                    found_move = True
                                    break
                            if found_move:
                                break
                        if found_move:
                            break
            return sol

        sol = local_search(result)
        
        # Iterated Local Search to escape local minima
        best_sol = set(sol)
        best_size = len(sol)
        
        # Time budget for ILS
        time_limit = DEADLINE - 10 # leave some time for other instances
        
        for i in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Perturb: remove a random vertex and repair
            perturbed = set(best_sol)
            if not perturbed:
                break
            
            # Remove 1 or 2 vertices
            k = 1 if len(perturbed) > 5 else 1
            to_remove = _random.sample(list(perturbed), min(k, len(perturbed)))
            for v in to_remove:
                perturbed.discard(v)
            
            # Repair: Greedily add vertices to cover uncovered cliques
            # Calculate uncovered
            covered = set()
            for v in perturbed:
                covered |= vert_cliques[v]
            uncovered = set(range(m)) - covered
            
            while uncovered:
                best_v = -1
                best_gain = 0
                for v in range(n):
                    if v in perturbed:
                        continue
                    gain = len(uncovered & vert_cliques[v])
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_v == -1 or best_gain == 0:
                    break
                perturbed.add(best_v)
                uncovered -= vert_cliques[best_v]
            
            # Local search on perturbed
            perturbed = local_search(perturbed)
            
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_sol = set(perturbed)
        
        return sorted(best_sol)

    results = []
    for idx, (n, p, seed) in enumerate(((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))):
        remaining_instances = 3 - idx
        time_per_instance = (DEADLINE - (time.time() - start_time)) / remaining_instances
        if time_per_instance < 5:
            # Not enough time, return empty or best effort?
            # We must return a valid solution.
            # Let's just run a very fast greedy.
            edges = random_graph(n, p, seed)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            cliques = bron_kerbosch(adj, n)
            if not cliques:
                results.append([])
                continue
            m = len(cliques)
            vert_cliques = [set() for _ in range(n)]
            for ci, c in enumerate(cliques):
                for v in c:
                    vert_cliques[v].add(ci)
            uncovered = set(range(m))
            res = set()
            while uncovered:
                best_v = -1
                best_gain = 0
                for v in range(n):
                    if v in res: continue
                    gain = len(uncovered & vert_cliques[v])
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_v == -1: break
                res.add(best_v)
                uncovered -= vert_cliques[best_v]
            results.append(sorted(res))
            continue
        
        sol = solve_instance(n, p, seed)
        results.append(sol)

    return results
