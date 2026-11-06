import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 135  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch_pivot(R, P, X, adj, cliques, time_limit, current_time):
        """
        Bron-Kerbosch with pivot to find all maximal cliques.
        R: current clique
        P: candidates that can be added to R
        X: candidates that were removed from P (for maximality)
        adj: adjacency list
        cliques: list to store found maximal cliques
        """
        if time.time() > time_limit:
            return False  # Timeout
        
        if not P and not X:
            if len(R) >= 2:
                cliques.append(list(R))
            return True

        # Choose pivot u from P ∪ X to minimize branching
        # Heuristic: vertex with max degree in P ∪ X
        union_px = P | X
        if not union_px:
            return True
            
        # Pick pivot
        pivot = max(union_px, key=lambda v: len(P & adj[v]) + len(X & adj[v]))
        
        # Iterate over vertices in P \ N(pivot)
        candidates = P - adj[pivot]
        
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            # Recursive call
            success = bron_kerbosch_pivot(new_R, new_P, new_X, adj, cliques, time_limit, current_time)
            if not success:
                return False
                
            P.remove(v)
            X.add(v)
            
        return True

    def find_large_cliques(adj, n, time_budget):
        """
        Find a set of large cliques using Bron-Kerbosch with a time limit.
        """
        cliques = []
        initial_P = set(range(n))
        initial_X = set()
        initial_R = set()
        
        # We might not find ALL maximal cliques if time runs out.
        # That's okay, we'll handle leftovers later.
        bron_kerbosch_pivot(initial_R, initial_P, initial_X, adj, cliques, time.time() + time_budget, time.time())
        
        return cliques

    def greedy_exact_cover(cliques, edges, n, adj, time_budget):
        """
        Select a subset of cliques that covers all edges exactly once.
        Greedy: Sort by size (number of edges) descending.
        Pick a clique if all its edges are still uncovered.
        """
        # Map edge to index? No, just use sets.
        uncovered = set(edges)
        selected_cliques = []
        
        # Sort cliques by number of edges (size*(size-1)/2) descending
        # Also, we can filter out cliques that are subsets of others? 
        # Bron-Kerbosch returns maximal cliques, so no clique is a subset of another.
        
        # Heuristic: Sort by size descending
        cliques_sorted = sorted(cliques, key=lambda c: len(c), reverse=True)
        
        for c in cliques_sorted:
            if time.time() > time.time() + time_budget:
                break
            
            # Check if all edges in c are in uncovered
            c_edges = set()
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    e = (min(c[i], c[j]), max(c[i], c[j]))
                    c_edges.add(e)
            
            # If any edge in c is already covered, skip
            if c_edges.issubset(uncovered):
                selected_cliques.append(c)
                uncovered -= c_edges
                
        return selected_cliques, uncovered

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        adj = build_adj(edges, n)
        
        if not edges:
            return []
            
        # Phase 1: Find maximal cliques
        # Allocate 70% of time to finding cliques, 30% to covering
        clique_find_time = time_budget * 0.7
        cover_time = time_budget * 0.3
        
        maximal_cliques = find_large_cliques(adj, n, clique_find_time)
        
        # Phase 2: Greedy Exact Cover
        selected, uncovered = greedy_exact_cover(maximal_cliques, edges, n, adj, cover_time)
        
        # Phase 3: Handle remaining edges
        if uncovered:
            # Run a simple greedy on the remaining edges
            # This is the fallback logic from the original, but on a smaller graph
            remaining_edges = uncovered
            remaining_adj = [set() for _ in range(n)]
            for u, v in remaining_edges:
                remaining_adj[u].add(v)
                remaining_adj[v].add(u)
                
            # Simple greedy: pick edge, grow
            while remaining_edges:
                if time.time() > time.time() + cover_time: # Reuse remaining budget
                    break
                # Pick random edge
                e = next(iter(remaining_edges))
                u, v = e
                clique = [u, v]
                # Try to grow
                candidates = remaining_adj[u] & remaining_adj[v]
                # Sort by degree
                candidates = sorted(candidates, key=lambda x: -len(remaining_adj[x]))
                for w in candidates:
                    # Check if w connects to all in clique
                    ok = True
                    for c in clique:
                        if (min(c, w), max(c, w)) not in remaining_edges:
                            ok = False
                            break
                    if ok:
                        clique.append(w)
                        # Update candidates? Simplified: just check against clique
                        # This is O(n^3) worst case but n is small in remaining graph
                        new_candidates = set()
                        for x in candidates:
                            if x == w: continue
                            # Check if x connects to all in new clique
                            ok2 = True
                            for c in clique:
                                if (min(c, x), max(c, x)) not in remaining_edges:
                                    ok2 = False
                                    break
                            if ok2:
                                new_candidates.add(x)
                        candidates = list(new_candidates)
                        candidates.sort(key=lambda x: -len(remaining_adj[x]))
                        
                selected.append(clique)
                # Remove edges in clique from remaining
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        ce = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        remaining_edges.discard(ce)
                        # Update remaining_adj
                        remaining_adj[clique[i]].discard(clique[j])
                        remaining_adj[clique[j]].discard(clique[i])

        return selected

    # Solve both instances
    t1 = time.time()
    budget1 = min(60, DEADLINE - (t1 - start_time) - 10)
    cliques1 = solve_instance(32, 0.6, 601, budget1)

    t2 = time.time()
    budget2 = min(60, DEADLINE - (t2 - start_time))
    cliques2 = solve_instance(64, 0.55, 602, budget2)

    return [cliques1, cliques2]
