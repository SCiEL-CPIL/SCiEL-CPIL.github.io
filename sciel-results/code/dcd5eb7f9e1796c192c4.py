"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def max_clique(adj, n):
        """
        Finds a maximum clique in the graph defined by adj.
        Uses Tomita's algorithm (branch and bound).
        """
        best_clique = []
        
        # Order vertices by degree (descending) to improve pruning
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        
        # We will work on a list of candidate vertices
        # The algorithm:
        # expand(R, P)
        # R: current clique
        # P: set of candidates that can be added to R (neighbors of all in R, not yet in R)
        
        def expand(R, P):
            nonlocal best_clique
            
            if not P:
                if len(R) > len(best_clique):
                    best_clique = R[:]
                return
            
            # Simple bound: if len(R) + len(P) <= len(best_clique), prune
            if len(R) + len(P) <= len(best_clique):
                return
                
            # Iterate through candidates in P
            # We process them in reverse order of the 'order' list usually, 
            # but here we just iterate through the set/list.
            # To be efficient, we should order P based on some heuristic.
            
            P_list = list(P)
            # Sort P by degree in the subgraph induced by P? 
            # Or just use the global order.
            P_list.sort(key=lambda x: order.index(x), reverse=True)
            
            for v in P_list:
                # v is a candidate
                # New candidates are neighbors of v that are also in P
                # and have not been processed yet? 
                # Actually, the standard Tomita algorithm removes v from P after processing.
                
                # Check if v is still in P (it is, we are iterating a copy)
                if v not in P:
                    continue
                    
                R.append(v)
                
                # New P is intersection of P and adj[v]
                new_P = P & adj[v]
                
                expand(R, new_P)
                
                R.pop()
                P.remove(v)
                
                # Update best if needed (handled in base case)
            
        # Initial call
        # R is empty, P is all vertices
        # But we must ensure the clique found is valid in the *remaining* graph.
        # The 'adj' passed in IS the remaining graph's adjacency.
        
        expand([], set(range(n)))
        
        return best_clique

    def clique_partition(n, p, seed):
        adj = get_graph(n, p, seed)
        cliques = []
        
        # We need to track remaining edges.
        # Since we remove cliques, we can just update the adj sets.
        # If an edge (u,v) is removed, we remove v from adj[u] and u from adj[v].
        
        while True:
            if time.time() > deadline:
                # Fallback: if we run out of time, we must ensure validity.
                # The current 'cliques' might not cover all edges if we broke early.
                # This is a risk. We should try to finish or handle remaining edges.
                # For now, let's break and handle remaining edges as singletons (size 2 cliques).
                break
            
            # Check if any edges remain
            has_edges = False
            for u in range(n):
                if adj[u]:
                    has_edges = True
                    break
            
            if not has_edges:
                break
            
            # Find max clique in the current remaining graph
            clique = max_clique(adj, n)
            
            if not clique:
                # Should not happen if has_edges is true, unless n=1 or similar edge case
                # But graph is simple, so min clique size is 2 (an edge).
                # If max_clique returns empty, it means no edges?
                # Or maybe the algorithm failed?
                # Let's pick a random edge if this happens.
                for u in range(n):
                    if adj[u]:
                        v = next(iter(adj[u]))
                        clique = [u, v]
                        break
            
            if not clique:
                 break

            # Remove all edges in this clique from adj
            for u in clique:
                for v in clique:
                    if u != v:
                        if v in adj[u]:
                            adj[u].remove(v)
            
            cliques.append(clique)
            
        # If we broke due to time, we must cover remaining edges.
        # The problem requires EVERY edge to be in exactly one clique.
        # If we stop early, we have uncovered edges.
        # We must add them as cliques of size 2 (edges).
        # This is valid because an edge is a clique of size 2.
        
        # Find all remaining edges
        remaining_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    remaining_edges.add((u, v))
        
        for u, v in remaining_edges:
            cliques.append([u, v])
            
        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = clique_partition(n, p, seed)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
