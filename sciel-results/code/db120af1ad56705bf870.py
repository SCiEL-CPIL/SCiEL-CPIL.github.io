import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer for safety
    
    def is_triangle_free(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True

    def max_independent_set_exact(n, adj):
        # Exact branch and bound for maximum independent set size
        # Uses greedy coloring bound (Tomita style)
        
        # Precompute neighbors as bitmasks for efficiency? 
        # n <= 41, so we can use Python integers as bitmasks.
        
        # Convert adj to list of bitmasks
        mask = [0]*n
        for i in range(n):
            m = 0
            for j in range(n):
                if adj[i][j]:
                    m |= (1 << j)
            mask[i] = m
        
        # We want to find max independent set.
        # An independent set is a set of vertices where no two are adjacent.
        # This is equivalent to finding a clique in the complement graph.
        # Or directly using the independence number algorithm.
        
        # Let's implement a recursive search with bounding.
        # State: candidate set (bitmask), current size, best found.
        
        best = [0]
        
        # Greedy coloring of the candidate set to provide an upper bound.
        # The candidates are vertices that can be added to the current independent set.
        # We color the *complement* of the induced subgraph on candidates?
        # No, standard algorithm for Max Independent Set:
        # 1. Pick a vertex v from candidates.
        # 2. Branch 1: Include v. New candidates = candidates \ (N(v) ∪ {v}).
        # 3. Branch 2: Exclude v. New candidates = candidates \ {v}.
        # Bound: If current_size + |candidates| <= best, prune.
        # Better bound: Greedy coloring of the subgraph induced by candidates.
        # The number of colors in the complement graph (which is the graph of non-edges) 
        # gives an upper bound on the size of the independent set we can still add?
        # Actually, if we color the graph induced by candidates such that adjacent vertices have different colors,
        # then an independent set can contain at most one vertex from each color class? No.
        # An independent set can contain multiple vertices from the same color class if they are not adjacent?
        # Wait. In a proper coloring, vertices of the same color are NOT adjacent.
        # So an independent set can contain ALL vertices of a color class.
        # This bound is not useful directly for independent set size.
        
        # Let's use the standard "Maximum Clique" algorithm on the complement graph.
        # Max Independent Set in G = Max Clique in G_bar.
        
        # Complement adjacency masks
        comp_mask = [0]*n
        all_mask = (1 << n) - 1
        for i in range(n):
            # Non-neighbors (excluding self)
            # i is not connected to i.
            # mask[i] has neighbors.
            # comp_mask[i] should have bits for j != i such that NOT adj[i][j].
            # So comp_mask[i] = (all_mask ^ (1 << i)) ^ mask[i]
            comp_mask[i] = (all_mask ^ (1 << i)) ^ mask[i]
        
        def expand(clique, candidates, color_class):
            # clique: current clique (list of vertices)
            # candidates: bitmask of vertices that can be added to clique
            # color_class: list of lists, color_class[k] contains vertices of color k in the greedy coloring of candidates
            # The bound is len(clique) + len(color_class)
            
            if not candidates:
                if len(clique) > best[0]:
                    best[0] = len(clique)
                return
            
            # Bound: if len(clique) + len(color_class) <= best[0], prune
            if len(clique) + len(color_class) <= best[0]:
                return
            
            # Process vertices in order of color class (reverse order usually)
            # We iterate through color classes from last to first.
            for i in range(len(color_class)-1, -1, -1):
                # Check bound
                if len(clique) + i + 1 <= best[0]:
                    break
                
                # Get vertices in this color class
                c_set = color_class[i]
                
                # Iterate through vertices in this color class
                # We must remove vertices from candidates as we go?
                # Standard Tomita algorithm:
                # for v in c_set:
                #    expand(clique + [v], candidates & comp_mask[v], recolor(candidates & comp_mask[v]))
                #    candidates = candidates & ~(1 << v)
                
                # To be efficient, we iterate through the set.
                # Note: c_set is a list of vertices.
                # We need to check if v is still in candidates.
                
                for v in c_set:
                    if not (candidates & (1 << v)):
                        continue
                    
                    # Branch: include v
                    new_clique = clique + [v]
                    new_candidates = candidates & comp_mask[v]
                    
                    # Recolor new_candidates
                    # Greedy coloring of the subgraph induced by new_candidates in the COMP graph
                    # This gives an upper bound on the clique size we can add.
                    # The number of colors is the bound.
                    
                    # Implementing a fast greedy coloring for bitmasks is complex.
                    # Given n<=41, and time limit, a simpler bound might suffice or we can use a simpler recursive structure.
                    
                    # Let's use a simpler bound: |new_candidates|
                    # This is weak but fast.
                    # If len(new_clique) + popcount(new_candidates) <= best[0], prune.
                    
                    # Let's try to implement a slightly better bound: greedy coloring.
                    # For n=41, this is doable.
                    
                    # Helper: greedy color a set of vertices (bitmask) in the comp graph
                    # Returns list of color classes
                    def greedy_color(mask):
                        classes = []
                        while mask:
                            # Pick a vertex (e.g., lowest bit)
                            # Actually, order matters. Let's just pick any.
                            # To make it deterministic and fast, pick lowest set bit.
                            v = (mask & -mask).bit_length() - 1
                            # Create a color class containing v and all non-neighbors of v in the remaining mask
                            # In the comp graph, neighbors of v are comp_mask[v].
                            # Non-neighbors in the comp graph are (mask ^ (1<<v)) ^ (comp_mask[v] & mask)
                            # Wait, we are coloring the COMP graph to find clique in COMP (which is indep in G).
                            # In Max Clique algorithm, we color the graph we are finding clique in.
                            # Here we are finding clique in COMP graph.
                            # So we color the subgraph of COMP induced by `mask`.
                            # Two vertices are adjacent in COMP if they are NOT adjacent in G.
                            # So in the coloring of COMP, adjacent vertices (in COMP) must have different colors.
                            # This means vertices that ARE adjacent in G can have the same color.
                            
                            # Let's re-read Tomita.
                            # "Coloring of the candidate set"
                            # The bound is the number of colors.
                            # This works because any clique in the graph can have at most one vertex of each color.
                            # So we color the graph in which we are finding the clique.
                            # Here, we are finding a clique in the COMP graph.
                            # So we must color the COMP graph.
                            
                            # Adjacent in COMP = Non-adjacent in G.
                            # So in a color class of COMP, no two vertices are adjacent in COMP.
                            # Which means no two vertices are Non-adjacent in G.
                            # Which means all pairs in a color class ARE adjacent in G.
                            # So a color class in the COMP graph is a CLIQUE in the G graph.
                            
                            # Wait, I want Max Independent Set in G.
                            # That is Max Clique in COMP.
                            # So I run Max Clique on COMP.
                            # The bound is the number of colors in a greedy coloring of the COMP graph induced by candidates.
                            
                            # So:
                            # 1. Start with `mask` (candidates in COMP).
                            # 2. Pick v.
                            # 3. Color class C contains v.
                            # 4. Add to C any vertex u in remaining mask that is NOT adjacent to v in COMP.
                            #    "Not adjacent in COMP" means "Adjacent in G".
                            #    So u must be a neighbor of v in G.
                            #    And u must be in the remaining mask.
                            # 5. Remove C from mask.
                            # 6. Repeat.
                            
                            # Let's implement this.
                            class_bits = 1 << v
                            # Find all u in (mask ^ class_bits) such that adj[v][u] is True.
                            # adj_mask[v] contains neighbors in G.
                            # So candidates for this class are (mask ^ class_bits) & adj_mask[v]
                            # Wait, we need to remove them from the main mask too.
                            
                            # Let's do it iteratively.
                            color_class = []
                            cur = mask
                            while cur:
                                # Pick v (lowest bit)
                                v = (cur & -cur).bit_length() - 1
                                color_class.append(v)
                                # Remove v from cur
                                cur ^= (1 << v)
                                # Find neighbors of v in G that are in cur
                                # These can share the same color in the COMP coloring
                                # because they are NOT adjacent in COMP (they ARE adjacent in G).
                                # Wait.
                                # If u is adjacent to v in G, then u is NOT adjacent to v in COMP.
                                # So u and v can have the same color in the COMP coloring.
                                # So we should add all neighbors of v in G (that are in cur) to the color class.
                                # And remove them from cur.
                                
                                # Neighbors of v in G: adj_mask[v]
                                # Intersect with cur: cur & adj_mask[v]
                                # Add these to color_class and remove from cur.
                                neigh_in_cur = cur & adj_mask[v]
                                while neigh_in_cur:
                                    u = (neigh_in_cur & -neigh_in_cur).bit_length() - 1
                                    color_class.append(u)
                                    cur ^= (1 << u)
                                    neigh_in_cur ^= (1 << u)
                                    
                                # Note: We don't need to update `mask` for the current recursion level yet?
                                # The greedy coloring function should just return the classes.
                                # The `mask` passed to greedy_color is the set of candidates.
                                # The function consumes `mask` to produce classes.
                                
                                # Wait, the logic above modifies `cur` which is a copy of `mask`.
                                # So it's fine.
                                
                            classes.append(color_class)
                        return classes

                    # Call greedy_color on new_candidates
                    # This might be slow if done at every node.
                    # Given time limit, maybe we should limit the depth or use a weaker bound initially.
                    # Let's try it.
                    
                    if len(new_clique) + new_candidates.bit_count() <= best[0]:
                        # Weak bound check before coloring
                        pass
                    else:
                        new_classes = greedy_color(new_candidates)
                        expand(new_clique, new_candidates, new_classes)
                    
                    # Branch: exclude v
                    # Remove v from candidates
                    candidates ^= (1 << v)
                    
                    # Note: The color_class list passed to expand is for the ORIGINAL candidates.
                    # When we exclude v, the candidates change, so the coloring is invalid.
                    # However, the standard algorithm iterates through the color classes of the ORIGINAL set.
                    # It relies on the fact that if v is in color class i, then any clique using v can have size at most i+1 (relative to the bound).
                    # When we exclude v, we continue with the remaining vertices in the same color class and subsequent classes.
                    # The bound `len(clique) + i + 1` is still valid for the remaining vertices in this class and earlier classes?
                    # Actually, the bound is `len(clique) + number_of_colors_remaining`.
                    # The Tomita algorithm structure is:
                    # for i in range(len(color_class)-1, -1, -1):
                    #   if len(clique) + i + 1 <= best: break
                    #   for v in color_class[i]:
                    #     if v in candidates:
                    #       expand(clique+[v], candidates & comp_mask[v], color(candidates & comp_mask[v]))
                    #       candidates ^= (1<<v)
                    
                    # This structure is correct.
                    # The `color_class` passed to expand is for the CURRENT candidates at the START of the expand call.
                    # Inside expand, we iterate through these classes.
                    
                    # So the code above is roughly correct.
                    
                    # One optimization: if best[0] is already 10, we can stop early?
                    # We want to check if alpha >= 10.
                    # If we find a clique of size 10, we can return True (alpha >= 10).
                    # The task requires alpha < 10.
                    # So if max_independent_set_exact returns >= 10, the graph is INVALID.
                    # We want to find a graph where this returns < 10.
                    
                    # We can add an early termination if best[0] >= 10.
                    # Let's modify expand to check this.
                    
                    # Actually, let's just let it run. n=41 is small enough.
                    
            # End for i in color classes

        # Initial call
        # Candidates: all vertices
        # Color all vertices
        initial_candidates = (1 << n) - 1
        initial_classes = greedy_color(initial_candidates)
        
        # We need to pass the initial candidates to expand.
        # But expand expects `candidates` and `color_class` to be consistent.
        # The `color_class` is the coloring of `candidates`.
        
        expand([], initial_candidates, initial_classes)
        
        return best[0]

    def check_graph(n, edges):
        if not is_triangle_free(n, edges):
            return False
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        alpha = max_independent_set_exact(n, adj)
        return alpha < 10

    # Strategy:
    # 1. Try to construct a known valid graph on n=39.
    #    The known graph is the Andrassai graph A_9? No, A_9 has 18 vertices.
    #    The largest known triangle-free graph with alpha < 10 is on 39 vertices.
    #    It is often constructed as a circulant or similar.
    #    Let's try to generate random graphs on n=39, 40 and check them.
    #    Random generation is unlikely to find it.
    #    We need a structured approach.
    
    # Known fact: R(3,10) > 39.
    # There is a specific graph on 39 vertices.
    # Let's try to find it by searching over "almost regular" triangle-free graphs?
    
    # Alternative: Use the seed's approach but with the exact checker.
    # The seed used a greedy heuristic which was unreliable.
    # If we use the exact checker, we can safely return a graph if it passes.
    
    # Let's try to construct the 39-vertex graph.
    # One candidate is the "Hoffman-Singleton" type? No, that's for R(3,6).
    # The graph for R(3,10) is the 39-vertex graph found by...?
    
    # Let's try a random search with the exact checker.
    # It might be too slow.
    
    # Let's try the seed's graph structure but verify it exactly.
    # Seed graph: n=26, edges if (j-i)%n % 3 == 1.
    # Let's check n=26 with exact checker.
    # If it's valid, it's a good fallback.
    
    # But we want to maximize n.
    
    # Let's try n=39 with a specific construction.
    # The 39-vertex graph is the complement of the 39-vertex graph?
    
    # Let's try to use a random restart local search to find a valid graph on n=39.
    # Start with random triangle-free graph.
    # Check alpha.
    # If alpha >= 10, try to add edges to reduce alpha while maintaining triangle-free.
    
    # Given the time limit, let's just try a few specific candidates.
    
    # Candidate 1: n=26 from seed.
    # Candidate 2: n=39 random search (limited time).
    
    # Let's implement a simple random search for n=39.
    # Time budget for search: 100 seconds.
    
    def random_search(n, time_budget):
        deadline = time.time() + time_budget
        random.seed(42) # Deterministic for reproducibility? Maybe not, but good for debugging.
        
        for _ in range(1000): # Max iterations
            if time.time() > deadline:
                break
            
            # Generate random triangle-free graph
            # Start with empty graph, add random edges if they don't form a triangle.
            adj = [[False]*n for _ in range(n)]
            edges = []
            
            # Random order of edges
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                # Check if adding (u, v) creates a triangle
                ok = True
                for k in range(n):
                    if adj[u][k] and adj[v][k]:
                        ok = False
                        break
                if ok:
                    adj[u][v] = True
                    adj[v][u] = True
                    edges.append([u, v])
            
            # Check alpha
            if check_graph(n, edges):
                return edges
        
        return None

    # Try n=40 first? R(3,10) <= 41, so n=40 is possible?
    # The survey says 39 is the largest known.
    # Let's try n=39.
    
    # To speed up, we can use a heuristic:
    # If we find a graph on n=39, return it.
    # If not, try n=38, etc.
    
    # But random search for n=39 is very unlikely to succeed quickly.
    
    # Let's try the seed's graph for n=26 first to ensure we have a valid answer.
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    if check_graph(n_seed, edges_seed):
        seed_valid = True
    else:
        seed_valid = False
    
    # Now try to find a better one.
    # Let's try n=39 with a short time budget.
    # If it fails, fall back to seed.
    
    # Actually, let's try to construct the known 39-vertex graph.
    # It is the "39-vertex Ramsey graph".
    # I recall it is a circulant graph C_39(S).
    # Let's try to guess S.
    # Degree should be around 10-12.
    
    # Let's try a few circulant graphs on n=39.
    # S must be symmetric (if s in S, 39-s in S).
    # S cannot contain 0.
    # Triangle free: if a, b, a+b in S (mod 39), then triangle.
    
    # Let's try to generate all triangle-free circulants on n=39 and check alpha.
    # There are 2^19 possible sets S (half the vertices).
    # Too many to check all.
    
    # Let's stick to the random search for a short time, then fallback.
    
    # Time budget for random search: 50 seconds.
    # If we find n=39, great.
    # If not, try n=38?
    
    # Actually, the seed score is 1.0.
    # If we return n=26, score is 26/26 = 1.0.
    # If we return n=39, score is 39/26 = 1.5.
    
    # Let's try to be more aggressive.
    
    # One known graph is the "Andrassai" graph?
    # No, let's just do the random search.
    
    # To make it faster, we can use a heuristic to add edges.
    # Start with a sparse graph, add edges to reduce alpha.
    
    # But for now, let's just run the random search for n=39 for 60 seconds.
    # If it fails, return seed.
    
    # Wait, the exact checker is slow.
    # 1000 iterations * check_graph (which is slow) might exceed 120s.
    
    # Let's limit iterations based on time.
    
    # Let's try n=39 for 30 seconds.
    # Then n=38 for 10 seconds.
    # Then fallback.
    
    # Actually, let's just try n=39 for 100 seconds.
    # If it finds one, return it.
    # Otherwise return seed.
    
    # But if the seed is invalid (unlikely), we need a fallback.
    # The seed graph is likely valid.
    
    # Let's optimize the exact checker.
    # The greedy coloring inside the recursion is the bottleneck.
    # For n=41, it should be okay.
    
    # Let's run it.
    
    # One more thing: the task says "Maximize n".
    # So we should try n=40 first?
    # R(3,10) <= 41 implies there is NO graph on 41 vertices.
    # There might be one on 40.
    # But 39 is the known one.
    # Let's try 39.
    
    # Let's try a specific known graph if I can remember it.
    # The 39-vertex graph is the complement of the 39-vertex graph?
    
    # Let's just use the random search.
    
    # To ensure we finish in time, we will set a hard limit.
    
    # Try n=39
    edges_found = None
    deadline = time.time() + 100
    while time.time() < deadline:
        edges_found = random_search(39, time.time() - start_time if time.time() > start_time else 1)
        # random_search takes a time_budget.
        # I passed a weird argument. Let's fix random_search call.
        break # Just one call with internal budget?
        
    # Let's restructure random_search to take a deadline.
    
    def random_search_deadline(n, deadline):
        random.seed(42)
        while time.time() < deadline:
            adj = [[False]*n for _ in range(n)]
            edges = []
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            for u, v in all_edges:
                ok = True
                for k in range(n):
                    if adj[u][k] and adj[v][k]:
                        ok = False
                        break
                if ok:
                    adj[u][v] = True
                    adj[v][u] = True
                    edges.append([u, v])
            
            if check_graph(n, edges):
                return edges
        return None

    # Try n=39
    deadline = time.time() + 100
    edges_39 = random_search_deadline(39, deadline)
    if edges_39:
        return {'n': 39, 'edges': edges_39}
    
    # Try n=38
    if time.time() < 110:
        edges_38 = random_search_deadline(38, 110)
        if edges_38:
            return {'n': 38, 'edges': edges_38}
            
    # Fallback to seed
    if seed_valid:
        return {'n': n_seed, 'edges': edges_seed}
    
    # If seed is invalid, return a trivial graph (e.g., n=1)
    return {'n': 1, 'edges': []}

if __name__ == '__main__':
    res = construct()
    print(res)
