"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def strong_edge_coloring_local_search(edges, n, time_limit=10):
        num_edges = len(edges)
        if num_edges == 0:
            return []

        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute conflict masks
        conflict_mask = [0] * num_edges
        for i in range(num_edges):
            u, v = edges[i]
            blocking_verts = {u, v}
            blocking_verts.update(adj[u])
            blocking_verts.update(adj[v])
            
            for j in range(num_edges):
                if i == j: continue
                x, y = edges[j]
                if x in blocking_verts or y in blocking_verts:
                    conflict_mask[i] |= (1 << j)
        
        # Initial Greedy Coloring (DSATUR-like: highest degree first)
        degrees = [bin(mask).count('1') for mask in conflict_mask]
        order = sorted(range(num_edges), key=lambda i: -degrees[i])
        
        color_of_edge = [-1] * num_edges
        color_masks = [] # List of bitmasks for each color
        
        def get_smallest_color(i):
            for c in range(len(color_masks)):
                if not (conflict_mask[i] & color_masks[c]):
                    return c
            return len(color_masks)

        for i in order:
            c = get_smallest_color(i)
            if c == len(color_masks):
                color_masks.append(0)
            color_of_edge[i] = c
            color_masks[c] |= (1 << i)
        
        initial_colors = len(color_masks)
        
        # Local Search
        start_time = time.time()
        rng = random.Random(42)
        
        # Helper to check if color c is valid for edge i
        def is_valid(i, c):
            if c >= len(color_masks):
                return False
            return not (conflict_mask[i] & color_masks[c])

        # Helper to assign color c to edge i
        def assign(i, c):
            old_c = color_of_edge[i]
            if old_c == c:
                return
            # Remove from old color
            color_masks[old_c] &= ~(1 << i)
            # Add to new color
            color_of_edge[i] = c
            color_masks[c] |= (1 << i)
            
        # Helper to compact colors
        def compact():
            # Map old color indices to new ones
            # Only keep colors that are not empty
            new_colors = []
            mapping = {}
            for c in range(len(color_masks)):
                if color_masks[c] != 0:
                    new_c = len(new_colors)
                    mapping[c] = new_c
                    new_colors.append(color_masks[c])
            
            if len(new_colors) < len(color_masks):
                # Update color_of_edge
                for i in range(num_edges):
                    color_of_edge[i] = mapping[color_of_edge[i]]
                # Update global color_masks
                color_masks[:] = new_colors
                return True
            return False

        best_colors = len(color_masks)
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # Try to lower colors of edges
            # Iterate in random order to escape local minima
            edge_order = list(range(num_edges))
            rng.shuffle(edge_order)
            
            for i in edge_order:
                current_c = color_of_edge[i]
                if current_c == 0:
                    continue
                    
                # Try to assign a lower color
                found = False
                for c in range(current_c):
                    if is_valid(i, c):
                        assign(i, c)
                        improved = True
                        found = True
                        break
                
                if not found:
                    # Try to move to a higher color temporarily? 
                    # No, let's just try to compact and see if that helps.
                    pass
            
            # Compact colors
            if compact():
                improved = True
                
            if not improved:
                # Stuck. Try a random kick.
                # Pick a random edge and move it to a random valid color (possibly higher)
                i = rng.choice(edge_order)
                current_c = color_of_edge[i]
                
                # Find all valid colors for i
                valid_colors = []
                for c in range(len(color_masks) + 1): # Allow one new color
                    if is_valid(i, c):
                        valid_colors.append(c)
                
                if valid_colors:
                    # Pick a random one different from current if possible
                    if len(valid_colors) > 1:
                        new_c = rng.choice([c for c in valid_colors if c != current_c] or valid_colors)
                    else:
                        new_c = valid_colors[0]
                        
                    if new_c != current_c:
                        assign(i, new_c)
                        # Continue loop to try to lower it again
                        continue
            
            if len(color_masks) < best_colors:
                best_colors = len(color_masks)
                if best_colors == 1: # Theoretical min for non-empty graph? No, min is 1 if no edges, else >=1.
                    break

        return color_of_edge

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        colors = strong_edge_coloring_local_search(edges, n)
        out.append(colors)
    
    return out
# EVOLVE-BLOCK-END
