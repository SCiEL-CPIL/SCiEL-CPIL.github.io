import time
import sys

def construct():
    start_time = time.time()
    time_limit = 120.0
    safety_margin = 5.0
    
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def max_clique(adj, n):
        """
        Finds a maximum clique in the graph defined by adjacency lists 'adj' of size n.
        Uses a branch-and-bound approach similar to Tomita's algorithm.
        """
        best_clique = []
        
        # Pre-process: order vertices by degree (descending) to help pruning
        # But for simplicity and speed on small n, we can use the standard 
        # pivot-based Bron-Kerbosch variant for Max Clique.
        
        # We will use a recursive search with coloring bound (Tomita's improvement)
        # However, for n=40, a simpler branch and bound might suffice.
        
        # Let's implement a standard Max Clique B&B.
        # State: P (candidates), X (excluded), current_clique
        
        # To be efficient, we sort vertices initially.
        # But the graph is fixed.
        
        # We will use a list-based representation for speed.
        # adj[i] is a set of neighbors.
        
        # Tomita's algorithm uses greedy coloring of P to provide an upper bound.
        # For n=40, we can implement this.
        
        def color_sort(P):
            """
            Greedily color the vertices in P.
            Returns a list of vertices sorted by color, and the max color.
            This provides an upper bound on the clique size extendable from P.
            """
            if not P:
                return [], 0
            
            # P is a list of vertices
            # We perform greedy coloring
            colors = [0] * n
            # We want to assign colors such that adjacent vertices have different colors
            # Actually, for Max Clique, we color the *complement* of the subgraph induced by P?
            # No, Tomita's algorithm colors the graph induced by P such that 
            # vertices with the same color form an independent set in P.
            # Thus, a clique can have at most one vertex of each color.
            
            # Let's implement the standard greedy coloring for the upper bound.
            # We iterate through P and assign the smallest available color.
            
            uncolored = list(P)
            color_class = []
            current_color = 0
            # To make it fast, we just assign colors arbitrarily to independent sets.
            # A simple way:
            # 1. Pick a vertex v from uncolored.
            # 2. Assign it current_color.
            # 3. Remove v and its neighbors from uncolored.
            # 4. Increment current_color.
            # This is O(n^2) which is fine for n=40.
            
            ordered_vertices = []
            while uncolored:
                # Pick the first vertex (or one with highest degree in remaining for better bounds)
                # For simplicity, pick the first one.
                v = uncolored.pop(0)
                ordered_vertices.append((v, current_color))
                # Remove neighbors of v from uncolored
                neighbors = adj[v]
                new_uncolored = []
                for u in uncolored:
                    if u not in neighbors:
                        new_uncolored.append(u)
                uncolored = new_uncolored
                current_color += 1
                
            # The max color is current_color
            # The vertices are ordered by color.
            return ordered_vertices, current_color

        def expand(P, X, current_clique):
            nonlocal best_clique
            
            if not P:
                if len(current_clique) > len(best_clique):
                    best_clique = current_clique[:]
                return

            # Coloring bound
            colored_P, max_color = color_sort(P)
            
            # If current_clique + max_color <= len(best_clique), prune
            if len(current_clique) + max_color <= len(best_clique):
                return

            # Iterate through colored vertices in reverse order (highest color first)
            # Because higher colors imply larger potential? 
            # Actually, in Tomita's algorithm, we process vertices in the order they were colored,
            # but the bound is used to prune.
            
            # We process vertices from the last added color to the first.
            # This is because if we can't form a clique with the last color's vertices, 
            # we might still need to check earlier ones? 
            # Standard Tomita:
            # while P is not empty:
            #   v = last vertex in P
            #   if len(current) + color_bound(v) <= best: break
            #   ...
            
            # Let's stick to a simpler B&B without complex coloring if n is small, 
            # or implement the standard Tomita loop.
            
            # For n=40, let's use a simpler recursive B&B with a good heuristic.
            # We'll sort P by degree in P to pick good candidates.
            
            # To keep it robust and fast for n=40:
            # We will use the standard Bron-Kerbosch with pivot for Max Clique?
            # No, BK enumerates ALL maximal cliques. We want the MAXIMAL one.
            # B&B is better.
            
            # Let's implement a simple B&B with degree sorting.
            # Sort P by degree in the subgraph induced by P.
            # This helps find large cliques early.
            
            # Calculate degrees in P
            degs = {}
            for v in P:
                degs[v] = 0
            for v in P:
                for u in P:
                    if v != u and u in adj[v]:
                        degs[v] += 1
            
            # Sort P descending by degree
            sorted_P = sorted(P, key=lambda x: degs[x], reverse=True)
            
            for v in sorted_P:
                if len(current_clique) + 1 <= len(best_clique):
                    # Even if we add v, we can't beat best if we don't add others?
                    # No, we need a tighter bound. 
                    # Simple bound: len(current) + len(P_remaining)
                    # If len(current) + len(P_remaining) <= len(best), prune.
                    # But we are iterating.
                    pass
                
                # Pruning: if current_clique + v + (P \ {v} \ N(v)) cannot beat best
                # The max additional vertices is len(P) - 1 - degs_in_P(v) ? No.
                # The vertices that can be added with v are N(v) ∩ P.
                # So max size is len(current) + 1 + len(N(v) ∩ P).
                # Let's use this bound.
                
                # Calculate N(v) ∩ P
                # We have adj[v].
                # P is a set or list.
                
                # For efficiency, let's just do the recursion.
                # n=40 is small.
                
                new_P = [u for u in P if u != v and u in adj[v]]
                new_X = [u for u in X if u in adj[v]] # X is not strictly needed for Max Clique if we just want max size, 
                # but for correctness of B&B, we usually don't need X if we just explore.
                # Actually, for Max Clique, we don't need X to avoid duplicates if we are just looking for max size?
                # Yes, we can just explore all subsets. 
                # But to avoid re-exploring, we usually order vertices and only add vertices with index > v.
                
                # Let's use the index-based approach.
                # This is safer.
                
                # Rewrite expand to use indices.
                pass
            
            # The above loop is getting messy. Let's implement a clean Tomita-style Max Clique.
            
            # Clean Tomita Max Clique Implementation for small n
            
            # We'll use a global best_size and best_set.
            pass

        # Let's use a known efficient library or implement a robust one.
        # Given the constraints, a simple recursive B&B with degree sorting is sufficient for n=40.
        
        best_size = 0
        best_set = []
        
        def bbs(P, current):
            nonlocal best_size, best_set
            
            if not P:
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = current[:]
                return
            
            # Bound: current + len(P)
            if len(current) + len(P) <= best_size:
                return
            
            # Choose a pivot vertex v with max degree in P to reduce branching?
            # Or just sort by degree.
            
            # Sort P by degree in P
            degs = {}
            for v in P:
                degs[v] = sum(1 for u in P if u != v and u in adj[v])
            
            # Pick vertex with max degree? Or iterate?
            # Iterating all is O(n).
            # We can pick the best candidate to explore first.
            
            # To speed up, we can pick a vertex v from P, and branch on "include v" and "exclude v".
            # Include v: new_P = P ∩ N(v)
            # Exclude v: new_P = P \ {v}
            
            # This is 2^n worst case.
            # With bounds, it should be fast for n=40 random graphs.
            
            # Let's pick the vertex with the highest degree in P to include first?
            # Or the one with lowest?
            # Usually, including high-degree vertices leads to smaller P, which is good.
            
            # Let's sort P by degree descending.
            sorted_P = sorted(P, key=lambda x: degs[x], reverse=True)
            
            for v in sorted_P:
                # Branch 1: Include v
                new_P = [u for u in P if u != v and u in adj[v]]
                # Update bound for new_P
                if len(current) + 1 + len(new_P) > best_size:
                    bbs(new_P, current + [v])
                
                # Branch 2: Exclude v
                # We remove v from P
                P = [u for u in P if u != v]
                
                # Update bound for remaining P
                if len(current) + len(P) <= best_size:
                    break # Prune rest of P for this level
                
                # Note: We modify P in place.
                # This is a standard B&B.
            
        # Initial P is all vertices
        P = list(range(n))
        bbs(P, [])
        
        return best_set

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > time_limit - safety_margin:
            # Fallback: return all vertices (valid but not optimal)
            # Or return empty if we assume we found enough? 
            # Better to return a valid solution.
            # The seed solution is valid.
            # Let's just return the seed solution for remaining instances if timeout.
            # But we should try to solve as many as possible.
            pass
            
        edges = instance(n, p, seed)
        
        # Build adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Build complement graph adjacency
        # G_c has edge (u,v) if NOT in G
        adj_c = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    adj_c[u].add(v)
                    adj_c[v].add(u)
                    
        # Find Max Clique in G_c
        # This is the Max Independent Set in G
        max_clique_set = max_clique(adj_c, n)
        
        # The transversal is V \ MIS
        # MIS is max_clique_set
        transversal = [v for v in range(n) if v not in max_clique_set]
        
        out.append(sorted(transversal))
        
    return out

# We need to ensure the max_clique function is robust and fast.
# The bbs implementation above modifies P in place, which is tricky.
# Let's refine the max_clique function to be a standard, correct B&B.

def construct():
    start_time = time.time()
    time_limit = 120.0
    safety_margin = 5.0

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def max_clique(adj, n):
        """
        Finds a maximum clique in the graph defined by adjacency lists 'adj' of size n.
        """
        best_clique = []
        best_size = 0

        # Tomita's algorithm with coloring bound is more efficient.
        # Let's implement a simpler version first and see if it's fast enough.
        # For n=40, simple B&B might work.
        
        # We use a recursive function.
        # P: list of candidate vertices
        # X: list of excluded vertices (not strictly needed for max size if we order correctly)
        # current: list of vertices in current clique
        
        # To avoid exponential blowup, we use the bound: len(current) + len(P) <= best_size -> prune.
        # We also sort P by degree in the subgraph induced by P to find good solutions early.
        
        def expand(P, current):
            nonlocal best_clique, best_size
            
            if not P:
                if len(current) > best_size:
                    best_size = len(current)
                    best_clique = current[:]
                return

            # Bound
            if len(current) + len(P) <= best_size:
                return

            # Calculate degrees in P
            degs = {}
            for v in P:
                degs[v] = 0
            for v in P:
                for u in P:
                    if v != u and u in adj[v]:
                        degs[v] += 1

            # Sort P by degree descending
            sorted_P = sorted(P, key=lambda x: degs[x], reverse=True)

            for v in sorted_P:
                # Include v
                new_P = [u for u in P if u != v and u in adj[v]]
                
                # Bound check for new branch
                if len(current) + 1 + len(new_P) > best_size:
                    expand(new_P, current + [v])
                    
                # Exclude v
                # We remove v from P for subsequent iterations in this loop
                # Since we are iterating over a copy (sorted_P), we need to manage P carefully.
                # Actually, the standard B&B for Max Clique often uses the "include/exclude" branching.
                # If we just iterate and recurse on "include v", we are effectively exploring all subsets?
                # No. If we only recurse on "include v", we miss cliques that don't contain v.
                # We need to also explore "exclude v".
                
                # A better way: 
                # 1. Pick a vertex v from P.
                # 2. Branch 1: Include v. New P = P ∩ N(v).
                # 3. Branch 2: Exclude v. New P = P \ {v}.
                
                # This doubles the tree depth but prunes more effectively with good ordering.
                # However, the simple "iterate and include" approach combined with 
                # "remove from P" is equivalent to exploring all subsets if done correctly?
                # No, it's equivalent to finding a maximal clique?
                
                # Let's use the standard Bron-Kerbosch with pivot? 
                # BK enumerates all maximal cliques. We can track the max.
                # For n=40, the number of maximal cliques might be large but manageable?
                # In random graphs, the number of maximal cliques is often polynomial or small exponential.
                # Let's try Bron-Kerbosch. It's simpler to implement correctly.

                pass

        # Let's implement Bron-Kerbosch with Pivot (Tomita) for Max Clique.
        # It enumerates all maximal cliques. We keep the largest one.
        
        def bron_kerbosch(P, X, R):
            nonlocal best_clique, best_size
            
            if not P and not X:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = R[:]
                return

            # Choose pivot u from P ∪ X with max degree in P
            # If P is empty, no pivot.
            if P:
                # Find u in P ∪ X with max number of neighbors in P
                # For simplicity, pick u from P ∪ X
                candidates = P + X
                # If X is empty, just P
                # We want to maximize |N(u) ∩ P|
                max_deg = -1
                pivot = None
                for u in candidates:
                    deg = 0
                    for v in P:
                        if v in adj[u]:
                            deg += 1
                    if deg > max_deg:
                        max_deg = deg
                        pivot = u
            else:
                pivot = None

            # If pivot is None, iterate over all P
            # Else, iterate over P \ N(pivot)
            # Why? Because any maximal clique must contain a vertex from P \ N(pivot) 
            # if it doesn't contain pivot? 
            # Actually, the theorem is: If P \ N(pivot) is empty, then R is maximal.
            # Otherwise, we branch on vertices in P \ N(pivot).
            
            if pivot is not None:
                # Get neighbors of pivot in P
                pivot_neighbors_in_P = [v for v in P if v in adj[pivot]]
                # Vertices to branch on are P \ pivot_neighbors_in_P
                # Wait, the standard algorithm branches on P \ N(pivot).
                # Let's verify.
                # Bron-Kerbosch:
                # if P and X are both empty, output R
                # choose a pivot u from P ∪ X
                # for each v in P \ N(u):
                #     bron_kerbosch(P ∩ N(v), X ∩ N(v), R ∪ {v})
                #     P = P \ {v}
                #     X = X ∪ {v}
                
                # So we iterate over P \ N(pivot).
                to_branch = [v for v in P if v not in adj[pivot]]
            else:
                to_branch = P[:]

            for v in to_branch:
                # Recurse
                new_P = [u for u in P if u in adj[v]]
                new_X = [u for u in X if u in adj[v]]
                new_R = R + [v]
                
                # Pruning: if len(new_R) + len(new_P) <= best_size, skip?
                # This is a safe bound.
                if len(new_R) + len(new_P) > best_size:
                    bron_kerbosch(new_P, new_X, new_R)
                
                # Update P and X
                P = [u for u in P if u != v]
                X = X + [v]

        # Start with P = all vertices, X = empty, R = empty
        P = list(range(n))
        X = []
        R = []
        bron_kerbosch(P, X, R)
        
        return best_clique

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > time_limit - safety_margin:
            # Fallback to seed solution for remaining
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            out.append(sorted(touched))
            continue

        edges = instance(n, p, seed)
        
        # Build adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Build complement graph adjacency
        adj_c = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    adj_c[u].add(v)
                    adj_c[v].add(u)
                    
        # Find Max Clique in G_c (which is Max Independent Set in G)
        max_clique_set = max_clique(adj_c, n)
        
        # The transversal is V \ MIS
        transversal = [v for v in range(n) if v not in max_clique_set]
        
        out.append(sorted(transversal))
        
    return out
