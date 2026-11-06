import time
import sys
import random

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 80.0  # 80 seconds, leaving 10s buffer for the 90s limit
    
    N = 256
    # 1. Build the graph
    # Edge (a, b) exists if (a*b) % (a+b) == 0
    # We want the Maximum Independent Set, so edges represent "conflicts".
    
    # Using 0-indexed for internal logic, 1-indexed for values
    # Vertices are 0..255 representing numbers 1..256
    adj = [set() for _ in range(N)]
    
    # Precompute conflicts
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i].add(j)
                adj[j].add(i)
    
    # 2. Vertex Reordering
    # Tomita's algorithm works best if vertices are processed in a specific order.
    # One common heuristic is to order by degree (descending).
    # However, the standard Tomita implementation usually assumes a specific input order 
    # and performs a greedy coloring on the remaining vertices.
    # Let's sort vertices by degree descending to potentially get better bounds earlier.
    
    degrees = [len(adj[i]) for i in range(N)]
    order = sorted(range(N), key=lambda x: -degrees[x])
    
    # We will map the original indices to new indices based on this order?
    # Or just use the order to guide the search?
    # The standard Tomita algorithm (MaxClique/MIS) typically works on the adjacency list.
    # Let's stick to the standard recursive branch and bound on the adjacency structure.
    
    # To maximize the bound, we can reorder the vertices in the adjacency lists 
    # such that higher degree vertices come first. This helps the greedy coloring 
    # assign colors more tightly (or at least consistently).
    
    # Reorder adjacency lists by degree descending
    for i in range(N):
        # Sort neighbors by their degree descending
        adj[i] = sorted(list(adj[i]), key=lambda x: -degrees[x])
        
    # We also need to define the order in which we pick vertices for the independent set.
    # In MIS, we pick a vertex, add it to the set, and remove it and its neighbors from the candidate set.
    # The "coloring bound" is applied to the candidate set.
    
    best_size = 0
    best_set = []
    
    # We need a function to compute the greedy coloring bound for a set of vertices.
    # Greedy Coloring:
    # Assign colors to vertices in a given order such that no two adjacent vertices share a color.
    # The number of colors used is an upper bound on the size of the independent set?
    # No, the number of colors is an upper bound on the size of a *Clique* in the complement graph, 
    # which is an Independent Set in the original graph.
    # Yes. If we can color the graph with k colors, any independent set can have at most one vertex 
    # from each color class? No, that's wrong.
    # An independent set in G is a clique in G_complement.
    # The chromatic number of G_complement is an upper bound on the clique number of G_complement (which is MIS of G).
    # Greedy coloring gives an upper bound on the chromatic number.
    
    def greedy_color_bound(vertices):
        """
        Returns the number of colors used by greedy coloring on 'vertices'.
        This is an upper bound on the size of the Maximum Independent Set in the subgraph induced by 'vertices'.
        """
        if not vertices:
            return 0
        
        # We want to color the subgraph induced by 'vertices'.
        # The vertices are the candidates.
        # We process them in a specific order. 
        # A good order for coloring bound in MIS is often the reverse of the search order 
        # or simply the order they appear. 
        # In Tomita, the vertices are usually kept in a list, and we color them.
        
        # Let's use a standard greedy coloring.
        # colors[v] = color index
        # We want to minimize the number of colors to get a tight bound?
        # No, we want an *upper bound* on the MIS size.
        # The chromatic number chi(G) >= alpha(G) (independence number).
        # So any valid coloring count is an upper bound.
        # Greedy coloring produces a valid coloring.
        
        # To get a *tight* upper bound, we should try to use *few* colors.
        # So standard greedy (first fit) is good.
        
        # Order matters. Processing high degree vertices first often leads to fewer colors?
        # Or does it? 
        # Actually, for the bound to be useful, we want it to be as small as possible.
        # First-fit greedy is O(V+E).
        
        # Let's assume the input 'vertices' is a list of node indices.
        # We need to check adjacency within this set.
        
        # Optimization: 
        # Instead of building a subgraph, we just check if a neighbor is in the set and has a different color.
        # Since N=256, we can use a boolean array or set for membership.
        
        v_set = set(vertices)
        colors = {}
        max_color = 0
        
        # Sort vertices for coloring. 
        # Using the global degree order is a good heuristic.
        # Or, we can just iterate through the list.
        # Let's iterate through the list provided.
        
        for v in vertices:
            # Determine available colors
            used_colors = set()
            for u in adj[v]:
                if u in v_set and u in colors:
                    used_colors.add(colors[u])
            
            # Find the smallest color not in used_colors
            c = 0
            while c in used_colors:
                c += 1
            
            colors[v] = c
            if c > max_color:
                max_color = c
        
        return max_color + 1

    # Recursive Search
    # candidates: list of vertices available to be added to the independent set
    # current_set: list of vertices already in the independent set
    
    def search(candidates, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Upper bound
        # The size of the independent set we can form from 'candidates' is at most the coloring bound.
        ub = greedy_color_bound(candidates)
        
        if len(current_set) + ub <= best_size:
            return
            
        # Choose a vertex to branch on.
        # Heuristic: Pick the vertex with the highest degree in the candidate subgraph?
        # Or the one that appears first in the coloring order?
        # In Tomita, we usually pick the first vertex in the list, 
        # but the list is ordered by color classes from the coloring step.
        
        # Let's refine the search strategy to match Tomita more closely.
        # Tomita's MaxClique:
        # 1. Color the candidate vertices.
        # 2. Process vertices in order of color (highest color first).
        # 3. For a vertex v:
        #    - If len(current) + color(v) <= best, prune.
        #    - Add v to current.
        #    - Recurse with candidates = candidates intersect neighbors(v).
        #    - Remove v from candidates.
        
        # We need to adapt this.
        # Let's re-implement the loop inside search to use the coloring info.
        
        # We need the coloring details, not just the count.
        # Let's modify greedy_color_bound to return the order and colors.
        
        def color_and_order(vertices):
            if not vertices:
                return []
            
            v_set = set(vertices)
            colors = {}
            max_color = 0
            order = [] # List of (color, vertex)
            
            # Process vertices. 
            # To get a tight bound, we want to assign low colors to many vertices.
            # Processing high-degree vertices first might force higher colors on neighbors?
            # Let's stick to the global degree order for consistency.
            for v in sorted(vertices, key=lambda x: -degrees[x]):
                used_colors = set()
                for u in adj[v]:
                    if u in v_set and u in colors:
                        used_colors.add(colors[u])
                
                c = 0
                while c in used_colors:
                    c += 1
                
                colors[v] = c
                if c > max_color:
                    max_color = c
                order.append((c, v))
            
            # Sort by color descending, then maybe by degree?
            # Vertices with higher colors are "more constrained" or "later" in the coloring.
            # In MaxClique, we process high colors first to prune early.
            order.sort(key=lambda x: -x[0])
            
            return order, max_color + 1

        # Get coloring info
        ordered_vertices, ub = color_and_order(candidates)
        
        if len(current_set) + ub <= best_size:
            return
            
        # Iterate through vertices in the order determined by coloring
        # We need to maintain the candidate list as we go.
        # In Tomita, we maintain a list of vertices. 
        # When we pick v, the new candidate set is {u in candidates | u is neighbor of v}.
        # And we remove v from the remaining candidates for the next iteration in the loop.
        
        # To do this efficiently, we can just iterate through the ordered list.
        # But we need to ensure that when we recurse, we only consider vertices 
        # that are compatible with the current choice.
        
        # Let's use a standard approach:
        # For each vertex v in ordered_vertices (from high color to low):
        #   if len(current_set) + color(v) <= best_size: break (since colors are descending, subsequent will be lower)
        #   Actually, the bound for the *rest* of the vertices is color(v).
        #   So if len(current_set) + color(v) <= best_size, we can stop the loop entirely.
        
        # We need to track which vertices are still "available" in the current level of recursion.
        # The input 'candidates' is the set available at this level.
        # When we pick v, the next level gets candidates_next = candidates ∩ N(v).
        # The loop continues with candidates_remaining = candidates \ {v}.
        
        # This requires careful state management.
        # Let's pass a list of vertices.
        
        # Optimization: 
        # Instead of creating new sets for candidates_next, we can just filter.
        
        for c, v in ordered_vertices:
            if time.time() > deadline:
                return
                
            # Prune based on this vertex's color bound
            if len(current_set) + c <= best_size:
                # Since colors are non-increasing in this list, 
                # all subsequent vertices have color <= c.
                # So len(current_set) + color(subsequent) <= len(current_set) + c <= best_size.
                # We can break out of the loop.
                break
                
            # Branch: Include v
            current_set.append(v)
            
            # New candidates: neighbors of v that are in the current candidate set
            # AND we must ensure we don't pick vertices that were already "processed" in this loop?
            # No, the recursive call handles the subgraph.
            # The set of valid next vertices is {u in candidates | u is adjacent to v in the Conflict Graph?}
            # Wait. 
            # We are finding an Independent Set.
            # If we pick v, we CANNOT pick any neighbor of v.
            # So the next candidates are {u in candidates | u is NOT adjacent to v}.
            # Wait, my graph G has edges for CONFLICTS.
            # Independent Set in G means no two vertices are connected by an edge in G.
            # So if we pick v, we remove v and all its neighbors N_G(v) from the candidate pool.
            
            # So candidates_next = {u in candidates | u not in N_G(v) and u != v}
            
            # Let's compute this.
            # neighbors_v = adj[v]
            # candidates_next = [u for u in candidates if u not in neighbors_v and u != v]
            
            # To be efficient, we can use the fact that 'candidates' is a list.
            # And we are iterating through 'ordered_vertices' which is a permutation of 'candidates'.
            
            # Let's construct the next candidate list.
            # We only care about vertices that are still in the "remaining" pool?
            # Actually, the recursive call will re-color the new candidate set.
            
            # Construct candidates_next
            # Optimization: Pre-filter.
            # neighbors_v_set = set(adj[v])
            # candidates_next = [u for u in candidates if u not in neighbors_v_set and u != v]
            
            # This is O(N) per branch. N=256. Depth ~ 100. 25600 ops per leaf.
            # With pruning, it should be fast.
            
            # Let's do it.
            neighbors_v = adj[v]
            candidates_next = [u for u in candidates if u != v and u not in neighbors_v]
            
            search(candidates_next, current_set)
            
            current_set.pop()
            
            # Remove v from the candidate list for the next iteration of the loop
            # We are iterating over 'ordered_vertices'. 
            # The 'candidates' list for the *next* iteration in the loop should not include v.
            # But 'candidates' is immutable in this scope if we don't modify it.
            # We can just skip v in the next iterations by checking?
            # No, the loop is `for c, v in ordered_vertices`.
            # The 'candidates' passed to the recursive call was filtered.
            # But for the *next* vertex in the loop, the candidate set should be 
            # the original candidates minus the vertices already tried (including v).
            # However, the coloring bound `c` for the next vertex was calculated 
            # assuming the *original* candidate set.
            # This is the standard Tomita approach: 
            # The bound is valid for the subgraph induced by the *remaining* vertices 
            # IF the coloring was done on the full set?
            # Actually, removing vertices can only *decrease* the chromatic number (or keep it same).
            # So the bound from the full set is still a valid upper bound for the subset.
            # So we don't need to re-color in the loop, just use the pre-computed colors.
            # BUT, we must ensure that the "next" vertex is actually available.
            # Since we are iterating through the list of ALL candidates, 
            # and we just removed v from the *recursive* branch, 
            # for the *next* branch (sibling), v is still "available" in the sense that 
            # we are considering "what if we didn't pick the previous ones, but picked this one".
            # Wait.
            # Standard MIS logic:
            # We have a set C.
            # We pick v in C.
            # Branch 1: Include v. New set C' = C \ (N(v) U {v}).
            # Branch 2: Exclude v. New set C'' = C \ {v}.
            # The loop effectively does Branch 2 implicitly by moving to the next vertex 
            # while keeping the *other* candidates available?
            # No.
            # If we pick v1, we go deep.
            # When we return, we try v2.
            # The candidate set for v2 should be C \ {v1} (and potentially others removed by v1's inclusion? No, v1 is not included in this branch).
            # So for the sibling branch, the candidates are C \ {v1}.
            # Does the coloring bound for v2 (calculated on C) still hold for C \ {v1}?
            # Yes, removing vertices cannot increase the independence number.
            # So the bound is valid.
            
            # So we just need to ensure that when we recurse, we pass the correct set.
            # And in the loop, we just iterate.
            # However, we must be careful: 
            # If we picked v1, and v2 was a neighbor of v1, v2 is NOT in candidates_next for the recursive call.
            # But for the NEXT iteration of the loop (trying v2), v2 IS in the "logical" candidate set 
            # because we are in the branch where v1 was NOT picked.
            # So the logic holds.
            
    # Initial call
    # Candidates are all vertices
    initial_candidates = list(range(N))
    # Order them by degree descending for the first coloring? 
    # The search function does its own ordering/coloring.
    search(initial_candidates, [])
    
    # If no set found (should not happen), return empty or best effort
    if not best_set:
        # Fallback to simple heuristic: Greedy Independent Set
        remaining = set(range(N))
        greedy_set = []
        # Sort by degree ascending? 
        # To maximize size, pick low degree vertices first?
        # Or high degree? 
        # Usually, picking low degree vertices leaves more options for others.
        # Let's try sorting by degree ascending.
        verts = sorted(range(N), key=lambda x: degrees[x])
        for v in verts:
            if v in remaining:
                greedy_set.append(v)
                remaining.remove(v)
                for u in adj[v]:
                    if u in remaining:
                        remaining.remove(u)
        best_set = greedy_set
        best_size = len(best_set)

    # Convert back to 1-indexed values
    result = sorted([v + 1 for v in best_set])
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
