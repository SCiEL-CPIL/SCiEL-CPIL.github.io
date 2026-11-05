import time
import random
import math
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s
    
    N = 256
    
    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Identify valid vertices (no self-loop conflict)
    # A vertex v is valid if 2*v is NOT a perfect square.
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
    
    # Map original values to indices 0..M-1 for compact representation
    M = len(valid_vertices)
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # Build conflict graph (adjacency lists) for valid vertices
    # adj[i] contains j such that valid_vertices[i] + valid_vertices[j] is a square
    adj = [set() for _ in range(M)]
    for i in range(M):
        vi = valid_vertices[i]
        for j in range(i + 1, M):
            vj = valid_vertices[j]
            if vi + vj in squares:
                adj[i].add(j)
                adj[j].add(i)
    
    # Build complement graph for Max Clique algorithm
    # comp_adj[i] contains j such that i and j are NOT connected in adj
    # Note: i is not in comp_adj[i] (no self-loops in clique search)
    comp_adj = [set() for _ in range(M)]
    for i in range(M):
        for j in range(M):
            if i != j and j not in adj[i]:
                comp_adj[i].add(j)
                
    # Convert to lists for faster iteration if needed, or keep sets
    # For Tomita, we need to iterate neighbors. Lists are faster.
    comp_adj_list = [list(s) for s in comp_adj]
    
    # Heuristic to get a good initial lower bound (incumbent)
    # Use Greedy + Local Search on the conflict graph (MIS)
    # This is equivalent to finding a large clique in comp_adj
    
    def greedy_clique_heuristic():
        """
        Greedy construction of a clique in comp_adj (which is an independent set in adj).
        """
        # Order vertices by degree in comp_adj (descending)
        # Higher degree in comp_adj means fewer conflicts in original graph
        degrees = [len(comp_adj[i]) for i in range(M)]
        order = sorted(range(M), key=lambda x: -degrees[x])
        
        current_clique = set()
        # We need to ensure all vertices in current_clique are mutually connected in comp_adj
        # Which means they are mutually non-adjacent in adj.
        
        for v in order:
            # Check if v is connected to all in current_clique in comp_adj
            # i.e., v is NOT adjacent to any in current_clique in adj
            valid = True
            for u in current_clique:
                if u in adj[v]:
                    valid = False
                    break
            if valid:
                current_clique.add(v)
                
        return current_clique

    def local_improve_clique(clique):
        """
        Try to add vertices to the clique (independent set in adj).
        """
        clique = set(clique)
        improved = True
        while improved:
            improved = False
            for v in range(M):
                if v in clique:
                    continue
                # Check if v can be added
                valid = True
                for u in clique:
                    if u in adj[v]:
                        valid = False
                        break
                if valid:
                    clique.add(v)
                    improved = True
        return clique

    def local_search_1_2(clique):
        """
        1-2 exchange: remove one vertex, try to add two.
        """
        clique = set(clique)
        improved = True
        iter_count = 0
        while improved and iter_count < 50 and (time.time() - start_time < DEADLINE - 2):
            improved = False
            iter_count += 1
            # Try removing each vertex
            for v in list(clique):
                new_clique = clique - {v}
                # Try to add any two vertices
                added = 0
                candidates_to_add = [u for u in range(M) if u not in new_clique]
                random.shuffle(candidates_to_add)
                
                for u in candidates_to_add:
                    if added >= 2: break
                    # Check if u can be added to new_clique
                    valid = True
                    for w in new_clique:
                        if w in adj[u]:
                            valid = False
                            break
                    if valid:
                        new_clique.add(u)
                        added += 1
                
                if len(new_clique) > len(clique):
                    clique = new_clique
                    improved = True
                    break
        return clique

    # Get initial solution
    best_size = 0
    best_clique = set()
    
    # Run a few heuristic restarts to get a strong incumbent
    for _ in range(10):
        if time.time() - start_time > DEADLINE - 5:
            break
        c = greedy_clique_heuristic()
        c = local_improve_clique(c)
        c = local_search_1_2(c)
        if len(c) > best_size:
            best_size = len(c)
            best_clique = c

    # Branch and Bound (Tomita's Algorithm) for Max Clique in comp_adj
    # expand(R, P)
    # R: current clique (list of vertices)
    # P: candidate set (list of vertices)
    
    # We need a coloring routine for the subgraph induced by P in comp_adj
    # The greedy coloring of G[P] provides an upper bound on the clique size.
    
    def greedy_coloring(P):
        """
        Returns a list of colors for vertices in P.
        The number of colors used is the upper bound.
        We want to order P by color descending for the B&B loop.
        """
        if not P:
            return [], 0
        
        P_list = list(P)
        # Randomize order for diversity? Or use degree?
        # Tomita usually uses a fixed order or random. Let's use random to break ties.
        random.shuffle(P_list)
        
        colors = {}
        color_count = 0
        # Simple greedy coloring
        for v in P_list:
            # Find the smallest color available
            # Neighbors of v in P
            neighbors_in_P = comp_adj[v] & set(P_list) # This is slow. 
            # Better: precomputed adj sets.
            # Let's optimize: 
            # For each v, check colors of its neighbors in P.
            
            # To make it faster, we can just assign color 1 if no neighbor has color 1, etc.
            # But we need to know which colors are forbidden.
            
            # Efficient approach:
            # Iterate through existing colors 1..color_count
            # Check if any neighbor in P has that color.
            
            forbidden = set()
            for u in P_list:
                if u == v: continue
                if u in comp_adj[v]:
                    if u in colors:
                        forbidden.add(colors[u])
            
            c = 1
            while c in forbidden:
                c += 1
            colors[v] = c
            if c > color_count:
                color_count = c
        
        # Sort P by color descending
        # Vertices with higher color are more "promising" to extend the clique?
        # Actually, the bound is that we can add at most color(v) vertices from the set of vertices with color >= color(v).
        # Tomita's algorithm processes vertices in order of decreasing color.
        
        ordered_P = sorted(P_list, key=lambda x: -colors[x])
        return ordered_P, colors

    # To speed up coloring, we can use a bitset or just accept the O(N^2) cost per node.
    # N is at most 240. O(N^2) is ~50k ops. If we have many nodes, this might be slow.
    # However, the pruning is very effective.
    
    # Optimization: Precompute neighbor sets as bitsets? 
    # Python sets are okay for N=240.
    
    # Let's refine the coloring to be faster.
    # We can use a list of colors and update incrementally? 
    # For now, simple greedy is fine for the first pass.
    
    # Global variables for B&B
    best_clique_found = list(best_clique)
    best_size_found = len(best_clique)
    
    def expand(R, P):
        nonlocal best_size_found, best_clique_found
        
        if time.time() - start_time > DEADLINE:
            return
            
        if not P:
            if len(R) > best_size_found:
                best_size_found = len(R)
                best_clique_found = list(R)
            return
            
        # Get colored order
        ordered_P, colors = greedy_coloring(P)
        
        # Process vertices in ordered_P
        # We need to maintain the candidate set P as we iterate
        # In Tomita, we iterate through the list, and for each v, we recurse with P' = P ∩ N(v)
        # Then we remove v from P.
        
        # To avoid copying P (list) repeatedly, we can pass the list and modify it?
        # But we need the original P for the next iteration of the loop?
        # Actually, the standard implementation iterates:
        # for v in ordered_P:
        #    if len(R) + colors[v] <= best_size: break
        #    R.append(v)
        #    P_new = [u for u in P if u in comp_adj[v]]  # This is the intersection
        #    expand(R, P_new)
        #    R.pop()
        #    P.remove(v)
        
        # Since P is a list, removing is O(N). Intersection is O(N).
        
        P_set = set(P) # For fast intersection? 
        # Actually, we can just filter.
        
        for v in ordered_P:
            # Pruning bound
            if len(R) + colors[v] <= best_size_found:
                break
                
            # Include v
            R.append(v)
            
            # New candidate set: neighbors of v in P
            # Note: P contains v. We should not include v in P_new.
            # comp_adj[v] does not contain v.
            P_new = [u for u in P if u in comp_adj[v]]
            
            expand(R, P_new)
            
            R.pop()
            
            # Remove v from P for subsequent iterations
            # We need to remove v from the list P
            # Since we are iterating over ordered_P, and P is being modified,
            # we should use a while loop or careful index management.
            # Simpler: Just create a new list for P in the next iteration?
            # No, P is passed by reference.
            
            # Let's restructure: 
            # The loop should be:
            # while P:
            #   v = P.pop() # or take from ordered list
            #   ...
            
            # Let's stick to the standard pattern:
            # We iterate over a copy of the ordered list? No.
            # The algorithm assumes P is updated.
            
            # To be safe and simple, let's use a recursive step that takes a list.
            # The cost of list filtering is acceptable for N=240.
            
            # Remove v from P
            try:
                P.remove(v)
            except ValueError:
                pass

    # Initial call
    # R is empty, P is all vertices
    # We should order P initially by degree or something?
    # The greedy coloring inside expand handles the ordering.
    
    # To maximize performance, we can order the initial P by degree in comp_adj (descending)
    # This often helps the first coloring.
    initial_P = list(range(M))
    # Sort by degree descending
    initial_P.sort(key=lambda x: -len(comp_adj[x]))
    
    expand([], initial_P)
    
    # If B&B didn't find a better solution, use the heuristic one
    if len(best_clique_found) < best_size:
        best_clique_found = list(best_clique)

    # Convert indices back to values
    result = sorted([valid_vertices[i] for i in best_clique_found])
    
    return result

# EVOLVE-BLOCK-END
