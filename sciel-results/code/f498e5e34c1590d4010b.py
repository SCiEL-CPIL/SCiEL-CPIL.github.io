import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    specs = [
        (30, 10, 0.7, 301),
        (60, 20, 0.7, 302),
        (90, 30, 0.7, 303)
    ]
    
    results = []
    
    for n, k, p, seed in specs:
        if time.time() > deadline:
            break
            
        # 1. Generate edges
        # We need to be careful with randomness to match the task's implicit graph structure
        # The task says "each cross pair joined with probability p".
        # The seed program uses:
        # rng = Random(seed)
        # edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        # edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        
        # I must replicate this exactly to get the same graph.
        import random
        rng = random.Random(seed)
        
        # Edges in C
        c_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                c_edges.add((u, v))
                
        # Edges between C and I
        # Note: The seed program iterates v in I, then u in C.
        # It adds (u, v) if random < p.
        # This defines the graph uniquely.
        
        c_i_edges = set()
        neighbors_in_c = {v: set() for v in range(k, n)}
        
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    c_i_edges.add((u, v))
                    neighbors_in_c[v].add(u)
        
        # 2. Construct cliques for I vertices
        cliques = []
        covered_c_edges = set()
        
        for v in range(k, n):
            N_v = neighbors_in_c[v]
            if len(N_v) >= 1:
                # Clique is {v} U N_v
                clique = [v] + sorted(N_v)
                cliques.append(clique)
                
                # Mark all edges within N_v as covered
                N_v_list = sorted(N_v)
                for i in range(len(N_v_list)):
                    for j in range(i + 1, len(N_v_list)):
                        covered_c_edges.add((N_v_list[i], N_v_list[j]))
        
        # 3. Find remaining edges in C
        remaining_c_edges = c_edges - covered_c_edges
        
        # 4. Cover remaining C edges
        # The remaining edges form a graph on vertices in C.
        # We need to cover them with cliques.
        # Since C is a clique, any subset of vertices forms a clique.
        # So we just need to partition the edges of the graph G_rem = (C, remaining_c_edges) into cliques.
        
        if remaining_c_edges:
            # Greedy: find maximal cliques in G_rem
            # A simple approach:
            # While remaining_c_edges is not empty:
            #   Pick an edge (u, v)
            #   Find all w such that (u, w) and (v, w) are in remaining_c_edges
            #   Actually, we want a maximal clique.
            #   Let's just use a simple heuristic:
            #   Since k is small, we can try to find large cliques.
            #   Or just add edges as size-2 cliques if we don't want to overcomplicate.
            #   Given the high probability that remaining_c_edges is empty,
            #   a simple fallback is sufficient.
            
            rem_edges_list = list(remaining_c_edges)
            # Sort to be deterministic
            rem_edges_list.sort()
            
            used = set()
            for e in rem_edges_list:
                if e in used:
                    continue
                # Start a new clique with this edge
                u, v = e
                current_clique = {u, v}
                used.add(e)
                
                # Try to expand this clique
                # Check all other vertices w in C
                # If (u, w) and (v, w) are in remaining_c_edges and not yet used in this clique?
                # Actually, we just need the set of edges in the clique to be a subset of remaining_c_edges.
                # Let's find the maximal set of vertices S containing u, v such that all pairs in S are in remaining_c_edges.
                
                # Simple expansion:
                # Iterate through all w in C
                # If w not in current_clique:
                #   Check if all (x, w) for x in current_clique are in remaining_c_edges
                #   If yes, add w to current_clique and mark those edges as used?
                #   Wait, if we add w, we cover edges (x, w).
                #   We must ensure we don't double cover or miss.
                
                # Let's do a proper maximal clique search for the component containing (u,v)?
                # Or just a simple greedy:
                
                # Let's try to build a maximal clique greedily
                # Candidate vertices: all w in C such that (u, w) in remaining_c_edges and (v, w) in remaining_c_edges
                # Actually, just check all w in C.
                
                # To keep it simple and fast:
                # We'll just add the edge (u,v) as a clique of size 2 if we can't find a bigger one easily.
                # But we can do a bit better.
                
                # Let's find all w such that (u, w) and (v, w) are in remaining_c_edges.
                # Then check if they form a clique among themselves.
                
                # Given the rarity of this case, let's just use size-2 cliques for safety and speed.
                # It's valid.
                cliques.append(sorted([u, v]))
                used.add((u, v))
                # Also mark reverse if needed? Set handles it.
                # Wait, if I just add (u,v), I haven't covered other edges incident to u or v that might be in remaining_c_edges.
                # I need to mark (u,v) as covered.
                # The loop `for e in rem_edges_list` will skip used edges.
                # But what about other edges like (u, w)?
                # They are still in remaining_c_edges.
                # They will be picked up in later iterations.
                # So this works, but it might use many size-2 cliques.
                # Is that optimal?
                # If remaining_c_edges is a star, we need many.
                # If it's a triangle, we can use 1 clique of size 3.
                # Let's try to be a bit smarter.
                
                # Revised plan for remaining edges:
                # 1. Get list of remaining edges.
                # 2. While list not empty:
                #    a. Pop an edge (u, v).
                #    b. Find all other edges (u, w) and (v, w) in the list.
                #    c. Try to form a clique.
                
                # Let's just implement a simple maximal clique finder for the remaining graph.
                # Since k <= 30, we can afford a bit more work.
                
                pass # Placeholder for better logic below

        # Let's re-implement the remaining edge coverage properly.
        
        if remaining_c_edges:
            rem_set = set(remaining_c_edges)
            while rem_set:
                # Pick an arbitrary edge
                u, v = next(iter(rem_set))
                rem_set.remove((u, v))
                
                # Find maximal clique containing u and v
                # Start with {u, v}
                clique_vertices = {u, v}
                
                # Try to add other vertices
                # A vertex w can be added if (x, w) is in rem_set for all x in clique_vertices
                # Note: we must remove the edges we use from rem_set.
                
                # Let's iterate through potential vertices
                # To make it deterministic, sort vertices
                potential = sorted([w for w in range(k) if w not in clique_vertices])
                
                for w in potential:
                    if w in clique_vertices:
                        continue
                    # Check if w connects to all in clique_vertices
                    can_add = True
                    for x in clique_vertices:
                        edge = (min(x, w), max(x, w))
                        if edge not in rem_set:
                            can_add = False
                            break
                    if can_add:
                        clique_vertices.add(w)
                        # Remove edges (x, w) from rem_set
                        for x in list(clique_vertices):
                            if x == w:
                                continue
                            edge = (min(x, w), max(x, w))
                            if edge in rem_set:
                                rem_set.remove(edge)
                
                # Add this clique
                cliques.append(sorted(clique_vertices))
        
        results.append(cliques)
        
    return results

# Keep the instance function as requested?
# "The seed program contains this exact generator, under the name instance; keep it."
# The prompt says: "Keep the function names and signatures of the seed program."
# The seed program has `construct`. Inside it, it defines `instance`.
# I should probably keep `instance` inside `construct` or make it global?
# "Keep the function names and signatures of the seed program."
# The seed program only exposes `construct`. `instance` is local.
# I will define `instance` locally to be safe, or just inline it.
# The prompt says "contains this exact generator... keep it".
# I will include a helper `instance` inside `construct` to be compliant, 
# but I will use my own logic to build the graph if I need more structure (neighbors).
# Actually, the generator returns edges. I can use that.

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    import random
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
        
    specs = [
        (30, 10, 0.7, 301),
        (60, 20, 0.7, 302),
        (90, 30, 0.7, 303)
    ]
    
    results = []
    
    for n, k, p, seed in specs:
        if time.time() > deadline:
            break
            
        edges = instance(n, k, p, seed)
        
        # Reconstruct graph structure from edges
        # C = {0..k-1}, I = {k..n-1}
        
        c_edges = set()
        c_i_edges = set()
        
        for u, v in edges:
            if u < k and v < k:
                c_edges.add((u, v))
            elif u >= k and v >= k:
                # Should not happen for split graph with this generator
                pass
            else:
                # One in C, one in I
                # Ensure u is in C for consistency?
                # Generator: (u, v) where u in C, v in I.
                # So u < k, v >= k.
                c_i_edges.add((u, v))
                
        neighbors_in_c = {v: set() for v in range(k, n)}
        for u, v in c_i_edges:
            neighbors_in_c[v].add(u)
            
        cliques = []
        covered_c_edges = set()
        
        for v in range(k, n):
            N_v = neighbors_in_c[v]
            if len(N_v) >= 1:
                clique = [v] + sorted(N_v)
                cliques.append(clique)
                
                N_v_list = sorted(N_v)
                for i in range(len(N_v_list)):
                    for j in range(i + 1, len(N_v_list)):
                        covered_c_edges.add((N_v_list[i], N_v_list[j]))
                        
        remaining_c_edges = c_edges - covered_c_edges
        
        if remaining_c_edges:
            rem_set = set(remaining_c_edges)
            while rem_set:
                u, v = next(iter(rem_set))
                rem_set.remove((u, v))
                
                clique_vertices = {u, v}
                
                potential = sorted([w for w in range(k) if w not in clique_vertices])
                
                for w in potential:
                    if w in clique_vertices:
                        continue
                    can_add = True
                    for x in clique_vertices:
                        edge = (min(x, w), max(x, w))
                        if edge not in rem_set:
                            can_add = False
                            break
                    if can_add:
                        clique_vertices.add(w)
                        for x in list(clique_vertices):
                            if x == w:
                                continue
                            edge = (min(x, w), max(x, w))
                            if edge in rem_set:
                                rem_set.remove(edge)
                            
                cliques.append(sorted(clique_vertices))
                
        results.append(cliques)
        
    return results
