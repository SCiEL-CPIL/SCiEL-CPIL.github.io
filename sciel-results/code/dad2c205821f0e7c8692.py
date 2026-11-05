import time
import sys
import random

def construct():
    start_time = time.time()
    DEADLINE = 110.0 # Safety margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(edges_set, n):
        uncovered = set(edges_set)
        cliques = []
        
        # Precompute adjacency list for the original graph
        # We will maintain a "remaining" adjacency structure implicitly via the uncovered set
        # But for fast lookup of neighbors in the uncovered subgraph, we can just check the uncovered set.
        
        # Helper to check if edge (u,v) is uncovered
        def is_uncovered(u, v):
            if u > v: u, v = v, u
            return (u, v) in uncovered

        # Helper to get neighbors of u in the uncovered subgraph
        def get_uncovered_neighbors(u):
            neighbors = []
            for v in range(n):
                if v != u and is_uncovered(u, v):
                    neighbors.append(v)
            return neighbors

        # Bron-Kerbosch with pivot to find a maximal clique
        # We want a large clique. We'll use a simple heuristic: 
        # Pick a vertex, find max clique in its neighborhood.
        
        def bron_kerbosch(R, P, X):
            # R: current clique
            # P: candidates that can extend R
            # X: candidates that were already processed (to ensure maximality)
            # This is a recursive implementation. For n=64, depth is small.
            if not P and not X:
                return list(R)
            
            # Pivot selection: choose u in P U X that maximizes |P \ N(u)|
            # Here N(u) is neighbors in P.
            # We want to minimize the branching.
            # Actually, standard pivot: u in P U X such that |P \ N(u)| is max.
            # Then we only recurse on P \ N(u).
            
            if not (P or X):
                return list(R)
            
            # Find pivot
            pivot = None
            max_uncovered = -1
            # We need to be careful with performance. n is small (32, 64).
            for u in list(P) + list(X):
                # Count neighbors of u in P
                # N_P(u) = {v in P | v adjacent to u}
                # We want to maximize |P \ N_P(u)|, which is equivalent to minimizing |N_P(u)|.
                # So we want u with fewest neighbors in P.
                count = 0
                for v in P:
                    if is_uncovered(u, v):
                        count += 1
                # We want to minimize count? 
                # Standard: pick u such that |P \ N(u)| is max.
                # |P \ N(u)| = |P| - |P \cap N(u)|.
                # So yes, minimize |P \cap N(u)|.
                if count < max_uncovered:
                    max_uncovered = count
                    pivot = u
            
            if pivot is None:
                pivot = next(iter(P)) if P else next(iter(X))
            
            # Branch on P \ N(pivot)
            # If P is empty, we are done.
            # The set of vertices to branch on is P \ N(pivot).
            # If P \ N(pivot) is empty, we have a maximal clique.
            
            # Optimization: If P is small, just iterate.
            # Get neighbors of pivot in P
            pivot_neighbors_in_P = set()
            for v in P:
                if is_uncovered(pivot, v):
                    pivot_neighbors_in_P.add(v)
            
            candidates_to_branch = [v for v in P if v not in pivot_neighbors_in_P]
            
            if not candidates_to_branch:
                return list(R)
            
            best_clique = list(R)
            
            for v in candidates_to_branch:
                # Recurse with R + {v}
                # New P = P \ N(v)
                # New X = X \ N(v)
                
                # Filter P and X
                new_P = set()
                for u in P:
                    if u != v and is_uncovered(v, u):
                        new_P.add(u)
                
                new_X = set()
                for u in X:
                    if u != v and is_uncovered(v, u):
                        new_X.add(u)
                
                # Recurse
                # To avoid deep recursion overhead, we can limit depth or use iterative?
                # n=64 is small enough for recursion if we prune well.
                # But we only need *a* large clique, not all.
                # We can do a DFS and keep the largest found.
                
                # Simple heuristic: Just pick the first branch that yields a large clique?
                # Or explore all? For n=64, the number of maximal cliques can be large.
                # Let's just do a greedy extension from R using P.
                # Actually, Bron-Kerbosch finds *all* maximal cliques. We only need one good one.
                # We can modify it to stop early if we find a clique of size > threshold?
                # Or just pick the branch with the largest potential.
                
                # Let's just do a greedy fill for simplicity and speed, 
                # using the structure of P.
                # Since we are in a recursive context, let's just return the first maximal clique found?
                # No, that might be small.
                
                # Alternative: Use a greedy heuristic on P to extend R.
                # We already have R. We want to add vertices from P to R such that they form a clique.
                # This is the Maximum Clique problem on P.
                # Given the time constraints, a greedy heuristic is safer than full BK.
                
                # Let's restart the logic:
                # We are trying to find a large clique in the remaining graph.
                # Let's just use a simple greedy clique finder on the whole remaining graph.
                
                # Break recursion and use a different approach for the main loop.
                return list(R) # Fallback, we will handle this in the main loop differently

        # Given the complexity of BK, let's use a simpler, effective heuristic:
        # 1. Find the vertex with the highest degree in the uncovered graph.
        # 2. Find the largest clique in its neighborhood using a greedy method.
        # 3. Add that clique to the partition.
        
        while uncovered:
            if time.time() > DEADLINE:
                # Fallback: add all remaining as edges
                for u, v in uncovered:
                    cliques.append([u, v])
                break

            # Find vertex with max degree in uncovered graph
            max_deg = -1
            best_vertex = -1
            for u in range(n):
                deg = 0
                for v in range(n):
                    if u != v and is_uncovered(u, v):
                        deg += 1
                if deg > max_deg:
                    max_deg = deg
                    best_vertex = u
            
            if best_vertex == -1:
                break # No edges left? Should be handled by while uncovered

            # Get neighbors of best_vertex in uncovered graph
            neighbors = get_uncovered_neighbors(best_vertex)
            
            # Find a large clique in the subgraph induced by neighbors
            # Greedy clique finder on neighbors
            # Sort neighbors by degree in the induced subgraph (among neighbors)
            # Degree in induced subgraph: count edges to other neighbors
            
            def degree_in_induced(v, node_set):
                count = 0
                for u in node_set:
                    if u != v and is_uncovered(v, u):
                        count += 1
                return count

            # Calculate degrees
            neigh_set = set(neighbors)
            degrees = {v: degree_in_induced(v, neigh_set) for v in neighbors}
            
            # Sort by degree desc
            sorted_neighs = sorted(neighbors, key=lambda x: degrees[x], reverse=True)
            
            # Greedy construction
            clique = [best_vertex]
            remaining_cands = set(sorted_neighs)
            
            while remaining_cands:
                # Pick vertex with max degree in remaining_cands
                # Or just pick the first one in sorted order that connects to all in clique?
                # Let's pick the one with highest degree in remaining_cands
                best_cand = None
                max_deg_cand = -1
                for v in remaining_cands:
                    deg = 0
                    for u in remaining_cands:
                        if u != v and is_uncovered(v, u):
                            deg += 1
                    if deg > max_deg_cand:
                        max_deg_cand = deg
                        best_cand = v
                
                if best_cand is None:
                    break
                    
                # Check if best_cand connects to all in current clique
                ok = True
                for u in clique:
                    if not is_uncovered(best_cand, u):
                        ok = False
                        break
                
                if ok:
                    clique.append(best_cand)
                    remaining_cands.remove(best_cand)
                else:
                    # If it doesn't connect, remove it from candidates?
                    # Or try next best?
                    # For simplicity, remove it and try next.
                    remaining_cands.remove(best_cand)
                    # Actually, if it doesn't connect to clique, it can't be added.
                    # We should just skip it.
                    continue
            
            # If we found a clique of size 1 (just best_vertex), we must add at least one edge
            # But best_vertex has neighbors, so clique should be at least size 2 if neighbors exist.
            # If neighbors list was empty, we wouldn't be here.
            
            # Ensure at least 2 vertices
            if len(clique) < 2:
                 # This shouldn't happen if best_vertex had degree > 0
                 # But if it did, pick any neighbor
                 if neighbors:
                     clique.append(neighbors[0])
                 else:
                     break

            cliques.append(clique)
            
            # Remove all edges in clique from uncovered
            m = len(clique)
            for i in range(m):
                for j in range(i+1, m):
                    a, b = clique[i], clique[j]
                    if a > b: a, b = b, a
                    if (a, b) in uncovered:
                        uncovered.remove((a, b))

        return cliques

    # Instance 1
    n1, p1, s1 = 32, 0.6, 601
    e1 = random_graph(n1, p1, s1)
    c1 = solve_instance(e1, n1)
    
    # Instance 2
    n2, p2, s2 = 64, 0.55, 602
    e2 = random_graph(n2, p2, s2)
    c2 = solve_instance(e2, n2)
    
    return [c1, c2]
