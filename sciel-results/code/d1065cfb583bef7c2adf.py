"""Strong edge colouring: DSATUR + Kempe Chain Local Search."""
import time
import math
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0  # safety margin under the 150 s kill

    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges):
        """Build adjacency list for the conflict graph of edges."""
        m = len(edges)
        v2e = {}
        for i, (u, v) in enumerate(edges):
            v2e.setdefault(u, []).append(i)
            v2e.setdefault(v, []).append(i)

        edge_idx = {}
        for i, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i

        # conflict[i] is a set of indices j such that edge i and edge j conflict
        conflict = [set() for _ in range(m)]

        # 1) Share a vertex
        for u, lst in v2e.items():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    conflict[lst[a]].add(lst[b])
                    conflict[lst[b]].add(lst[a])

        # 2) Joined by an edge
        for i, (u, v) in enumerate(edges):
            lu = v2e.get(u, [])
            lv = v2e.get(v, [])
            for a in lu:
                if a == i:
                    continue
                for b in lv:
                    if b == i or b == a:
                        continue
                    conflict[a].add(b)
                    conflict[b].add(a)
        
        # Convert to lists for faster iteration if needed, or keep as sets
        # Sets are good for "is x in y" checks.
        return conflict

    def dsatur_coloring(conflict, rng, max_colors=None):
        """
        DSATUR heuristic for graph coloring.
        Returns color array.
        """
        m = len(conflict)
        color = [-1] * m
        # sat[v] = number of distinct colors in neighborhood
        # deg[v] = degree in conflict graph
        deg = [len(c) for c in conflict]
        
        # available_colors[c] = count of available colors? No, we just track used colors.
        # We maintain a list of uncolored vertices.
        uncolored = list(range(m))
        
        # To speed up, we can use a priority queue or just sort. 
        # Given m <= 240, sorting is fine.
        
        for _ in range(m):
            if not uncolored:
                break
            
            # Select vertex with max saturation, then max degree
            # sat[u] is computed dynamically
            best_u = None
            best_sat = -1
            best_deg = -1
            
            # We need to know the colors of neighbors for each uncolored vertex
            # This is O(m * max_degree) per step, total O(m^2 * max_deg). 
            # m=240, max_deg in conflict graph can be high (up to 2*d^2 ~ 72?).
            # 240^2 * 72 ~ 4M ops. Very fast.
            
            for u in uncolored:
                # Calculate saturation
                neighbor_colors = set()
                for v in conflict[u]:
                    if color[v] != -1:
                        neighbor_colors.add(color[v])
                sat = len(neighbor_colors)
                
                if sat > best_sat or (sat == best_sat and deg[u] > best_deg):
                    best_sat = sat
                    best_deg = deg[u]
                    best_u = u
            
            if best_u is None:
                break
                
            # Assign smallest available color
            # Find colors used by neighbors
            used_colors = set()
            for v in conflict[best_u]:
                if color[v] != -1:
                    used_colors.add(color[v])
            
            # Find smallest non-negative integer not in used_colors
            c = 0
            while c in used_colors:
                c += 1
            
            color[best_u] = c
            uncolored.remove(best_u)
            
            if max_colors is not None and c >= max_colors:
                # If we exceed max_colors, we might want to fail fast or just continue
                # For now, just continue, but we can optimize later
                pass

        return color

    def kempe_chain_search(conflict, color, rng, time_budget=5.0):
        """
        Try to reduce number of colors using Kempe chains.
        """
        m = len(conflict)
        start_time = time.time()
        
        # Get current number of colors
        current_colors = set(c for c in color if c >= 0)
        if not current_colors:
            return color, 0
        
        max_c = max(current_colors)
        
        # We want to eliminate the highest color index
        # We iterate through colors from highest to lowest
        # For each color C, try to recolor all vertices of color C to other colors
        
        # To make it efficient, we only try to eliminate one color at a time
        # and restart the process if successful.
        
        while time.time() - start_time < time_budget:
            # Find the highest color
            if not current_colors:
                break
            target_color = max(current_colors)
            
            # Find vertices with this color
            vertices_of_color = [v for v in range(m) if color[v] == target_color]
            
            if not vertices_of_color:
                current_colors.discard(target_color)
                continue
            
            # Try to recolor each vertex in vertices_of_color
            # We do this one by one, updating the state
            # If we successfully recolor all, we have eliminated the color
            
            # To avoid infinite loops or getting stuck, we pick a random vertex
            # and try to find a Kempe chain to recolor it.
            
            # Actually, a simpler local search:
            # Pick a random vertex v with color target_color.
            # Try to recolor it to a lower color c < target_color.
            # If c is not in N(v), just recolor.
            # If c is in N(v), try to find a Kempe chain in colors {target_color, c}
            # starting from v's neighbor with color c, and see if we can reach a vertex
            # that allows recoloring v to c.
            
            success = False
            # Shuffle vertices to randomize
            rng.shuffle(vertices_of_color)
            
            for v in vertices_of_color:
                if color[v] != target_color:
                    continue # It might have been recolored in a previous step of this loop? 
                             # No, we are trying to eliminate target_color entirely.
                             # If we recolor one, it's no longer target_color.
                             # But we need to eliminate ALL of them.
                             # So we should try to recolor ONE, then restart the scan?
                             # Or just try to recolor this specific one.
                             # If we recolor v to c, then v is no longer target_color.
                             # We can proceed to the next vertex.
                             # If we get stuck, we break and try a different strategy.
                             
                # Find a color c < target_color that is available or can be made available
                # Available colors are those not in N(v)
                neighbor_colors = {color[u] for u in conflict[v] if color[u] != -1}
                
                # Try all colors c < target_color
                # Sort by some heuristic? e.g., fewer conflicts?
                # For now, random order or increasing order
                candidates = [c for c in range(target_color) if c not in neighbor_colors]
                
                if candidates:
                    # Simple recoloring
                    c = rng.choice(candidates)
                    color[v] = c
                    success = True
                    continue
                
                # No direct candidate. Try Kempe chain.
                # Pick a random neighbor color c < target_color
                # We need c to be in neighbor_colors to use Kempe chain
                # If c is NOT in neighbor_colors, we could have just recolored.
                # So we only consider c in neighbor_colors.
                
                possible_kempe_colors = [c for c in range(target_color) if c in neighbor_colors]
                
                if not possible_kempe_colors:
                    # No lower color is available or can be reached?
                    # This means we are stuck with this vertex.
                    break
                
                c = rng.choice(possible_kempe_colors)
                
                # Find a neighbor u of v with color c
                # We want to find a Kempe chain in the subgraph induced by colors {target_color, c}
                # that starts from some vertex connected to v with color c, 
                # and ends at a vertex that does NOT have a neighbor with color target_color 
                # (other than the one we are trying to recolor? No, the chain is in the rest of the graph).
                
                # Standard Kempe Chain Recoloring:
                # We want to change color of v from target_color to c.
                # v has a neighbor u with color c.
                # Consider the component of the graph induced by vertices with colors {target_color, c} 
                # that contains u.
                # If v is NOT in this component (which is true because v has color target_color 
                # and u has color c, and if they were in the same component, there would be a path 
                # of alternating colors. But wait, v is adjacent to u. 
                # If we swap colors in the component containing u, u becomes target_color.
                # Then v (target_color) and u (target_color) would conflict.
                # So we need the component containing u to NOT contain any vertex adjacent to v 
                # that has color target_color?
                # Actually, the condition is: 
                # Let G_{c, target} be the subgraph induced by vertices with colors c or target_color.
                # Let C_u be the connected component of C_u containing u.
                # If no vertex in C_u is adjacent to v (except u itself, which we handle), 
                # then we can swap colors in C_u.
                # After swap, u has color target_color.
                # v has color target_color.
                # They are adjacent. Conflict!
                
                # Correction:
                # We want to recolor v to c.
                # v is currently target_color.
                # u is a neighbor with color c.
                # If we swap colors in the Kempe chain containing u (colors c and target_color),
                # u becomes target_color.
                # Then v (target_color) and u (target_color) are adjacent. Bad.
                
                # So we need to find a neighbor u with color c such that 
                # the Kempe chain containing u does NOT contain any other neighbor of v with color target_color?
                # No, v is the one being recolored.
                # The issue is that u becomes target_color.
                # So v and u would both be target_color.
                # This implies we CANNOT recolor v to c if there is a neighbor u with color c 
                # in the same Kempe chain as... wait.
                
                # Let's rethink.
                # We want to recolor v to c.
                # If c is not in N(v), we can do it.
                # If c is in N(v), let u be a neighbor with color c.
                # We perform a Kempe chain swap on the component containing u in G_{c, target}.
                # This changes u to target_color.
                # Now v is target_color, u is target_color. Conflict.
                
                # Therefore, simple Kempe chain swap on the neighbor's component doesn't work directly 
                # to recolor v to c if u is in that component.
                
                # However, we can try to find a neighbor u with color c such that 
                # u is NOT in the same component as any other constraint?
                
                # Actually, there is a different move:
                # We can try to recolor v to c by finding a path in G_{c, target} 
                # from u to a vertex w such that w is NOT adjacent to v.
                # If such a path exists, we swap colors along the path.
                # Then u becomes target_color? No.
                # The path alternates c, target, c, target...
                # If we swap, u becomes target_color.
                # Still conflict.
                
                # Wait, the standard Kempe chain argument for 4-color theorem says:
                # If v has neighbors of colors 1, 2, 3, 4.
                # To recolor v to 1, we look at the 1-4 component containing the neighbor of color 1.
                # If the neighbor of color 4 is NOT in this component, we can swap the 1-4 component.
                # Then the neighbor of color 1 becomes 4.
                # Now v has neighbors of colors 2, 3, 4 (old 1), 4 (old 4).
                # Wait, if we swap 1 and 4 in the component containing neighbor 1, 
                # neighbor 1 becomes 4.
                # If neighbor 4 was NOT in that component, it remains 4.
                # So v now has two neighbors of color 4?
                # Yes. But v is being recolored to 1.
                # So v becomes 1.
                # Neighbors are 2, 3, 4, 4.
                # No conflict with 1.
                
                # So the condition is:
                # Let u_c be a neighbor of v with color c.
                # Let u_t be a neighbor of v with color target_color? No, v IS target_color.
                # We are looking for a color c to recolor v to.
                # Let u be a neighbor of v with color c.
                # Consider the component of G_{c, target} containing u.
                # If NO other neighbor of v is in this component, we can swap the component.
                # Then u becomes target_color.
                # v becomes c.
                # Are there conflicts?
                # v is c.
                # u is target_color.
                # Other neighbors of v:
                # - Those with color c: If any other neighbor w has color c, and w is in the component, it becomes target_color.
                #   If w is NOT in the component, it remains c.
                #   If w remains c, v(c) and w(c) conflict!
                #   So we need NO other neighbor of v with color c to be in the component?
                #   Actually, if w has color c and is in the component, it becomes target_color.
                #   Then v(c) and w(target) are fine.
                #   But if w has color c and is NOT in the component, it remains c.
                #   Then v(c) and w(c) conflict.
                #   So we need: All neighbors of v with color c must be in the component containing u?
                #   Or, more simply:
                #   We need to find a color c such that:
                #   1. c is in N(v).
                #   2. Let C be the component of G_{c, target} containing a neighbor u of v with color c.
                #   3. No neighbor of v with color c is OUTSIDE C. (So they all get swapped to target).
                #   4. No neighbor of v with color target is IN C? 
                #      Wait, v is target. Its neighbors with color target are irrelevant to the swap 
                #      unless they are in C. If a neighbor w has color target and is in C, it becomes c.
                #      Then v(c) and w(c) conflict!
                #      So we need: No neighbor of v with color target is IN C.
                
                # Summary of conditions to recolor v (color target) to c:
                # Let N_c = {neighbors of v with color c}
                # Let N_t = {neighbors of v with color target}
                # Pick u in N_c.
                # Let C be the component of G_{c, target} containing u.
                # Condition 1: N_c subset of C. (All neighbors of color c are in C, so they all become target).
                # Condition 2: N_t intersection C is empty. (No neighbors of color target are in C, so they don't become c).
                
                # If these hold, we can swap C and recolor v to c.
                
                # This is a specific check. Let's implement it.
                
                # We need to find such a c and u.
                # We iterate through possible c < target.
                # For each c, we check neighbors with color c.
                # If none, skip (we could have just recolored).
                # If some, pick one u.
                # Find component C of G_{c, target} containing u.
                # Check conditions.
                
                # This is expensive to do for every step.
                # Given time constraints, we might simplify.
                
                # Simplified Local Search:
                # Just try to recolor v to a random lower color c if available.
                # If not, try a random swap of two colors in the whole graph?
                
                # Let's stick to the "Simple Recoloring" if possible, 
                # and if not, try a "Random 1-Swap" (swap colors of two vertices) 
                # that doesn't create conflicts, hoping it opens up a recoloring path.
                
                # Actually, the current code in the prompt does 1-swap.
                # Let's improve the "Simple Recoloring" by trying all lower colors.
                
                # If no simple recoloring, we break and try a different vertex or restart.
                # To avoid getting stuck, we can add a "kick" move: 
                # Randomly swap colors of two non-adjacent vertices.
                
                # Let's implement the Kempe check for the specific c we picked.
                
                # Find component C
                visited = set()
                queue = [u]
                visited.add(u)
                C = set()
                
                while queue:
                    curr = queue.pop()
                    C.add(curr)
                    for nb in conflict[curr]:
                        if color[nb] in (c, target_color) and nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
                
                # Check conditions
                valid = True
                # Condition 1: All neighbors of v with color c are in C
                for nb in conflict[v]:
                    if color[nb] == c and nb not in C:
                        valid = False
                        break
                
                if valid:
                    # Condition 2: No neighbors of v with color target are in C
                    for nb in conflict[v]:
                        if color[nb] == target_color and nb in C:
                            valid = False
                            break
                
                if valid:
                    # Perform swap in C
                    for node in C:
                        if color[node] == c:
                            color[node] = target_color
                        elif color[node] == target_color:
                            color[node] = c
                    
                    # Recolor v
                    color[v] = c
                    success = True
                    break # Break the loop over vertices_of_color? 
                          # We recolor one vertex. We should restart the scan of vertices_of_color 
                          # because the set has changed.
                    # But for simplicity in this loop, we just mark success and continue 
                    # to the next vertex in the shuffled list? 
                    # No, if we recolor v, it's no longer target_color.
                    # The list `vertices_of_color` is static.
                    # So we just proceed to the next vertex in the list.
                    # If we recolor all of them, we are done with this target_color.
                    # If we fail to recolor one, we break and try a different strategy.
                
                if not valid:
                    # Try next candidate c?
                    # For now, just break and try a different vertex or kick.
                    pass

            if success:
                # We recolored at least one vertex.
                # We should update current_colors? 
                # No, we just continue the loop.
                # The next iteration will re-evaluate.
                pass
            else:
                # Stuck. Try a random kick: swap colors of two random non-adjacent vertices.
                # Pick two random vertices x, y.
                # If they are not adjacent and have different colors, swap them.
                # This might open up a path.
                
                x = rng.randrange(m)
                y = rng.randrange(m)
                if x != y and x not in conflict[y] and color[x] != color[y]:
                    # Check if swap is valid
                    # x takes color of y, y takes color of x
                    cx, cy = color[x], color[y]
                    
                    # Check x's neighbors
                    ok = True
                    for nb in conflict[x]:
                        if color[nb] == cy and nb != y:
                            ok = False
                            break
                    if ok:
                        for nb in conflict[y]:
                            if color[nb] == cx and nb != x:
                                ok = False
                                break
                    if ok:
                        color[x] = cy
                        color[y] = cx
                        # Don't break, try to recolor again
                    else:
                        break # Stuck, break inner loop
                else:
                    break # Stuck

        # Renumber compactly
        present = sorted(set(c for c in color if c >= 0))
        if not present:
            return color, 0
        remap = {old: new for new, old in enumerate(present)}
        for v in range(m):
            if color[v] >= 0:
                color[v] = remap[color[v]]
        return color, len(present)

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = gen_instance(n, d, seed)
        conflict = build_conflict_graph(edges)
        m = len(edges)
        best_colour = None
        best_used = m
        rng = Random(42)

        # Multiple DSATUR restarts
        n_restarts = 50
        for r in range(n_restarts):
            if time.time() > DEADLINE - 10.0:
                break
            
            # DSATUR is deterministic for a given graph, but we can randomize tie-breaking
            # or just run it once? 
            # DSATUR is usually run once. 
            # To get diversity, we can randomize the initial order or tie-breaking.
            # Here we just run DSATUR.
            # To add randomness, we can shuffle the initial vertex list? 
            # DSATUR picks best vertex dynamically. 
            # We can randomize the tie-breaking in DSATUR by shuffling the uncolored list 
            # before selecting the best? No, it selects based on score.
            # We can add a small random perturbation to the score.
            
            # For now, just run DSATUR.
            # If we want diversity, we can use a randomized greedy with DSATUR-like scoring.
            
            # Let's implement a randomized DSATUR.
            # In the selection step, instead of picking the max, pick from top K?
            
            # Simpler: Just run DSATUR. It's very strong.
            colour = dsatur_coloring(conflict, rng)
            used = len(set(c for c in colour if c >= 0))
            
            if used < best_used:
                best_colour = colour[:]
                best_used = used

        # Local compression on best
        if best_colour is not None:
            best_colour, best_used = kempe_chain_search(conflict, best_colour, rng, time_budget=20.0)

        # More targeted restarts if time permits
        if time.time() < DEADLINE - 10.0:
            for r in range(50):
                if time.time() > DEADLINE - 5.0:
                    break
                
                # Randomized Greedy with DSATUR scoring
                # We'll just use DSATUR again, maybe with a different seed for tie-breaking?
                # Or just run Kempe again on the best?
                
                # Let's try a different initial coloring: Random Greedy
                order = list(range(m))
                rng.shuffle(order)
                colour = [-1] * m
                used = 0
                col_masks = [0] * (best_used + 10) # Pre-allocate
                
                for e in order:
                    placed = False
                    for c in range(used):
                        # Check conflict
                        # Using set for conflict
                        ok = True
                        for nb in conflict[e]:
                            if colour[nb] == c:
                                ok = False
                                break
                        if ok:
                            colour[e] = c
                            placed = True
                            break
                    if not placed:
                        colour[e] = used
                        used += 1
                        if used >= len(col_masks):
                            col_masks.append(0)
                
                if used <= best_used:
                    colour, used = kempe_chain_search(conflict, colour, rng, time_budget=2.0)
                    if used < best_used:
                        best_colour = colour[:]
                        best_used = used

        if best_colour is None:
            best_colour = dsatur_coloring(conflict, rng)

        out.append(best_colour)

    return out
