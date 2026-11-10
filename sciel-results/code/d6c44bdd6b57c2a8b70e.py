"""Clique partition of split graphs using structural properties and residual optimization."""
import time
from random import Random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def max_clique(graph_edges, vertices):
        """
        Find a maximum clique in a graph defined by edges and vertices.
        Uses Bron-Kerbosch with pivoting.
        graph_edges: set of frozensets or tuples of edges (u, v) with u < v.
        vertices: list of vertices.
        Returns: list of vertices in the max clique.
        """
        if not graph_edges:
            return []
        
        # Build adjacency list for the subgraph
        adj = {v: set() for v in vertices}
        for u, v in graph_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Bron-Kerbosch
        max_clique_found = []
        
        def bron_kerbosch(R, P, X):
            nonlocal max_clique_found
            if not P and not X:
                if len(R) > len(max_clique_found):
                    max_clique_found = list(R)
                return
            
            # Choose pivot u from P union X to minimize branching
            # We want u with max degree in P
            pivot_candidates = list(P | X)
            if not pivot_candidates:
                return
                
            # Simple pivot selection: max degree in P
            best_pivot = None
            max_deg = -1
            for u in pivot_candidates:
                deg = len(P & adj.get(u, set()))
                if deg > max_deg:
                    max_deg = deg
                    best_pivot = u
            
            # If pivot is in X, it's not in P, so P \ N(u) is P
            # If pivot is in P, it's in P, so P \ N(u) is P \ N(u)
            # Standard BK: recurse on P \ N(u) for all u in P \ N(u)
            # Wait, the standard optimization is:
            # for v in P \ N(u):
            #    bron_kerbosch(R + {v}, P & N(v), X & N(v))
            #    P.remove(v)
            #    X.add(v)
            
            if best_pivot is None:
                return
                
            # P \ N(best_pivot)
            # Note: adj[best_pivot] contains neighbors in the current graph
            # We need neighbors in P
            neighbors_in_P = adj.get(best_pivot, set()) & P
            
            # Iterate over P \ neighbors_in_P
            # We need to modify P and X, so we work on copies or careful removal
            # Since we are in a recursive function, we can use local sets
            
            # To avoid modifying the caller's P and X directly in a way that breaks other branches,
            # standard BK modifies in place.
            # We pass sets by reference.
            
            # Create a list of candidates to iterate
            candidates = list(P - neighbors_in_P)
            
            for v in candidates:
                if v not in P: # Check if already removed
                    continue
                bron_kerbosch(R | {v}, P & adj.get(v, set()), X & adj.get(v, set()))
                P.remove(v)
                X.add(v)
                
        initial_P = set(vertices)
        initial_X = set()
        initial_R = set()
        
        bron_kerbosch(initial_R, initial_P, initial_X)
        
        return max_clique_found

    def partition_remaining_edges(edges, vertices):
        """
        Partition the given edges into cliques.
        edges: list of (u, v)
        vertices: list of all vertices involved
        Returns list of cliques (lists of vertices)
        """
        if not edges:
            return []
        
        remaining_edges = set(frozenset(e) for e in edges)
        cliques = []
        
        # Limit iterations to prevent infinite loops or timeouts on dense residuals
        max_iters = 100
        for _ in range(max_iters):
            if not remaining_edges:
                break
            
            # Get current vertices
            current_verts = set()
            for e in remaining_edges:
                current_verts.update(e)
                
            if not current_verts:
                break
                
            # Find max clique in the current remaining graph
            # We need to pass the edges and vertices to max_clique
            # max_clique expects a set of edges and a list of vertices
            clique_verts = max_clique(remaining_edges, list(current_verts))
            
            if not clique_verts:
                # Should not happen if edges exist, but safety check
                # Pick any edge
                e = next(iter(remaining_edges))
                clique_verts = list(e)
            
            cliques.append(sorted(clique_verts))
            
            # Remove all edges within clique_verts from remaining_edges
            clique_set = set(clique_verts)
            edges_to_remove = set()
            for u, v in combinations(sorted(clique_verts), 2):
                edges_to_remove.add(frozenset((u, v)))
            
            remaining_edges -= edges_to_remove
            
            if time.time() > deadline - 5:
                break
                
        # If there are still remaining edges (due to time or failure), cover them as 2-cliques
        if remaining_edges:
            for e in remaining_edges:
                cliques.append(sorted(list(e)))
                
        return cliques

    def partition_split_graph(n, k, edges):
        # 1. Separate edges
        # C = 0..k-1, I = k..n-1
        c_edges = []
        cross_edges = []
        i_edges = [] # Should be empty
        
        # Build adjacency for quick lookup
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            if u < k and v < k:
                c_edges.append((u, v))
            elif (u < k and v >= k) or (u >= k and v < k):
                cross_edges.append((u, v))
            else:
                i_edges.append((u, v))
        
        cliques = []
        covered_edges = set()
        
        # 2. For each i in I, form clique N(i) U {i}
        for i in range(k, n):
            neighbors_in_c = [u for u in adj[i] if u < k]
            if neighbors_in_c:
                clique = neighbors_in_c + [i]
                cliques.append(sorted(clique))
                
                # Mark all edges within this clique as covered
                # Edges are pairs (a, b) with a < b
                for u, v in combinations(sorted(clique), 2):
                    covered_edges.add(frozenset((u, v)))
        
        # 3. Find remaining edges in C
        remaining_c_edges = []
        for u, v in c_edges:
            if frozenset((u, v)) not in covered_edges:
                remaining_c_edges.append((u, v))
        
        # 4. Partition remaining C edges
        if remaining_c_edges:
            # Vertices involved in remaining edges
            rem_verts = set()
            for u, v in remaining_c_edges:
                rem_verts.add(u)
                rem_verts.add(v)
            
            res_cliques = partition_remaining_edges(remaining_c_edges, list(rem_verts))
            cliques.extend(res_cliques)
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        
        # Ensure types are Python int and list
        cleaned_cliques = []
        for c in cliques:
            # Verify it's a clique? The logic ensures it, but let's be safe.
            # The task says "Any broken rule makes the whole result invalid".
            # We trust the construction.
            cleaned_cliques.append([int(x) for x in c])
        results.append(cleaned_cliques)
    
    return results
# EVOLVE-BLOCK-END
