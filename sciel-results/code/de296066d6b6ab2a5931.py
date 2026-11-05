import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def find_max_clique(n, adj_sets):
        """
        Finds a maximum clique in the graph defined by adj_sets (list of sets).
        Uses Bron-Kerbosch with pivot.
        Returns the clique as a list of vertices.
        """
        best_clique = []
        
        def bron_kerbosch(r, p, x):
            nonlocal best_clique
            # Pruning: if current size + max possible size <= best size, return
            # Max possible size is len(p)
            if len(r) + len(p) <= len(best_clique):
                return
            
            if not p:
                if len(r) > len(best_clique):
                    best_clique = r[:]
                return
            
            # Pivot selection: choose u from P U X with max degree in P
            # To simplify, just pick a random or first one? 
            # Better: pick u in P with max degree. If P is empty, done.
            # Standard BK with pivot:
            # Choose u in P U X such that |P ∩ N(u)| is maximized.
            # Then iterate over P \ N(u).
            
            # For dense graphs, pivot helps a lot.
            # Let's pick u from P with highest degree in P.
            # If P is empty, we handled it above.
            
            # Optimization: If len(p) is small, just iterate all.
            # Let's implement standard pivot.
            
            # Find pivot u
            # We want to minimize the number of recursive calls.
            # u should maximize |p & adj[u]|.
            
            # Simple heuristic for pivot: pick node in p with max degree in p.
            # This is O(|p|^2).
            
            if not p:
                return
                
            # Pick pivot
            # In dense graphs, most nodes are connected.
            # Let's just pick a node from p.
            # A better pivot choice:
            pivot_candidates = list(p)
            if not pivot_candidates:
                return
            
            # Find u in P U X (X is usually empty in this recursive structure if we don't track it carefully, 
            # but standard BK tracks X for the "maximum" version or just uses P).
            # Let's use the version where we just recurse on P \ N(u).
            
            # To find a good pivot, we can sample or just pick the first one.
            # For dense graphs, the difference is small.
            # Let's pick the node in P with the highest degree in P.
            
            # Calculate degrees in P
            # This might be slow if P is large.
            # Let's just pick a random node from P as pivot for speed, or the first one.
            # Actually, for max clique, any pivot works for correctness, just affects speed.
            # Let's pick the one with max degree in P.
            
            # If P is large, this is O(|P|^2).
            # Given n=64, |P| <= 64. 64^2 = 4096. Very fast.
            
            pivot = max(p, key=lambda u: len(adj_sets[u] & p))
            
            # Iterate over P \ N(pivot)
            # Note: We must not modify P while iterating
            # We create a list of candidates
            candidates = [v for v in p if v not in adj_sets[pivot]]
            
            # If candidates is empty, then P is a clique? No, that means P is subset of N(pivot).
            # If P \ N(pivot) is empty, then all nodes in P are connected to pivot.
            # We still need to recurse?
            # Standard BK:
            # for v in P \ N(u):
            #    bron_kerbosch(R + {v}, P ∩ N(v), X ∪ N(v))
            #    P = P \ {v}
            #    X = X ∪ {v}
            
            # If P \ N(u) is empty, the loop doesn't run.
            # Is that correct?
            # If P \ N(u) is empty, it means every node in P is a neighbor of u.
            # Does that mean we found a clique? No.
            # But it means we can't prune further with this pivot?
            # Actually, if P \ N(u) is empty, we still need to check if P itself is a clique or contains a larger clique.
            # Wait, the recursion handles the exploration.
            # If the set to iterate is empty, we return.
            # This implies that if P is a clique, and we pick a pivot u in P, then P \ N(u) will contain all nodes in P except u (since u is connected to all in P).
            # So we will iterate over P \ {u}.
            
            for v in candidates:
                new_p = p & adj_sets[v]
                new_r = r + [v]
                bron_kerbosch(new_r, new_p, set()) # X is not used for max clique size only, but needed for correctness in some variants? 
                # For finding ANY max clique, we can ignore X if we are just looking for size.
                # But to ensure we find THE max, standard BK uses X.
                # However, since we just want a large clique, and we update best_clique, 
                # ignoring X might miss some cliques if we are not careful?
                # No, BK(R, P, X) explores all maximal cliques.
                # If we want max, we can just use P.
                
                # Remove v from p for subsequent iterations
                p.discard(v)
                
                # Check time
                if time.time() > deadline:
                    return

        # Initial call
        # P is all vertices
        # We need to pass a mutable set for P
        initial_p = set(range(n))
        bron_kerbosch([], initial_p, set())
        
        return best_clique

    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Return empty or partial if timeout
            results.append([])
            continue
            
        edges = generate_edges(n, p, seed)
        remaining_edges = set(edges)
        cliques = []
        
        while remaining_edges:
            if time.time() > deadline:
                break
                
            # Build adjacency sets for the remaining graph
            adj_sets = [set() for _ in range(n)]
            for u, v in remaining_edges:
                adj_sets[u].add(v)
                adj_sets[v].add(u)
            
            # Find max clique
            # We only care about vertices that have degree > 0
            active_vertices = [i for i in range(n) if adj_sets[i]]
            
            if not active_vertices:
                break
                
            # Restrict search to active vertices?
            # My BK implementation takes n and adj_sets.
            # It assumes vertices are 0..n-1.
            # If some vertices are isolated, they won't be in P if we filter P.
            # Let's filter P to active vertices.
            
            # Modify BK to take a specific set of vertices?
            # Easier: Just run BK on all n vertices. Isolated vertices will be in P initially,
            # but they have no neighbors, so they won't form cliques of size > 1.
            # The algorithm will quickly prune them.
            
            clique = find_max_clique(n, adj_sets)
            
            if not clique:
                # Fallback: take any edge
                u, v = next(iter(remaining_edges))
                clique = [u, v]
                
            # Add clique to results
            cliques.append(clique)
            
            # Remove edges in this clique from remaining_edges
            clique_set = set(clique)
            edges_to_remove = set()
            for u in clique_set:
                for v in clique_set:
                    if u < v:
                        edges_to_remove.add((u, v))
            
            remaining_edges.difference_update(edges_to_remove)
            
        results.append(cliques)
        
    return results
