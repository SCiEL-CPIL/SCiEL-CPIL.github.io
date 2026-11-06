import time
import random
from collections import defaultdict

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    results = []
    deadline = time.time() + 140.0

    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        m = len(edges)
        
        # Build adjacency set for G
        adj_g_set = set()
        for u, v in edges:
            adj_g_set.add((u, v))
            adj_g_set.add((v, u))
            
        edge_list = edges
        edge_index = {e: i for i, e in enumerate(edge_list)}
        
        # Build H = L(G)^2 conflict graph
        h_adj = [set() for _ in range(m)]
        
        for i, e1 in enumerate(edge_list):
            u1, v1 = e1
            for j, e2 in enumerate(edge_list):
                if i == j:
                    continue
                u2, v2 = e2
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    h_adj[i].add(j)
                    h_adj[j].add(i)
                else:
                    if (u1, u2) in adj_g_set or (u1, v2) in adj_g_set or (v1, u2) in adj_g_set or (v1, v2) in adj_g_set:
                        h_adj[i].add(j)
                        h_adj[j].add(i)

        # Helper to check if vertex v can take color c given current coloring
        def can_take_color(v, c, color, color_sets):
            # Check if any neighbor of v in H has color c
            # Optimization: precomputed lists? No, just check h_adj[v]
            for u in h_adj[v]:
                if color[u] == c:
                    return False
            return True

        def dsatur(seed_val=0):
            rng = random.Random(seed_val)
            color = [-1] * m
            sat = [set() for _ in range(m)]
            uncolored = set(range(m))
            deg_h = [len(h_adj[i]) for i in range(m)]
            
            while uncolored:
                best_v = -1
                best_sat = -1
                best_deg = -1
                # Add some randomness to tie-breaking
                candidates = []
                for v in uncolored:
                    s = len(sat[v])
                    d = deg_h[v]
                    candidates.append((s, d, v))
                
                if not candidates:
                    break
                    
                # Sort by saturation, then degree, then random
                candidates.sort(key=lambda x: (-x[0], -x[1], rng.random()))
                v = candidates[0][2]
                
                uncolored.remove(v)
                
                c = 0
                while c in sat[v]:
                    c += 1
                color[v] = c
                
                for u in h_adj[v]:
                    if u in uncolored:
                        sat[u].add(c)
                        
            return color

        # Get initial coloring
        best_col = None
        best_num_colors = m + 1
        
        # Try a few DSATUR seeds
        for s in range(5):
            if time.time() > deadline - 5.0:
                break
            col = dsatur(s)
            num_colors = max(col) + 1 if m > 0 else 0
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_col = col[:]
        
        col = best_col
        num_colors = best_num_colors
        
        # Local Search: Try to reduce colors by removing color classes
        # We maintain color_sets for quick access to vertices of a color
        color_sets = [set() for _ in range(num_colors)]
        for i in range(m):
            color_sets[col[i]].add(i)
        
        active_colors = list(range(num_colors))
        
        max_passes = 50
        pass_count = 0
        stagnant_passes = 0
        
        while len(active_colors) > 1 and pass_count < max_passes and time.time() < deadline - 5.0:
            pass_count += 1
            removed_any = False
            
            # Try to remove each active color
            # Sort active colors by size (smallest first) to try to eliminate small ones
            active_colors.sort(key=lambda c: len(color_sets[c]))
            
            for c_to_remove in active_colors[:]:
                if c_to_remove not in active_colors:
                    continue
                
                # Attempt to move all vertices in c_to_remove to other colors
                vertices_to_move = list(color_sets[c_to_remove])
                # Shuffle to break symmetry
                random.Random(pass_count * 1000 + c_to_remove).shuffle(vertices_to_move)
                
                success = True
                temp_color = col[:] # Backup? Or just track changes?
                # Let's track changes locally
                moved = {}
                
                for v in vertices_to_move:
                    # Try to find a color for v among active_colors except c_to_remove
                    found = False
                    # Prefer colors that are "full" or have less conflict?
                    # Just iterate
                    for c_target in active_colors:
                        if c_target == c_to_remove:
                            continue
                        if can_take_color(v, c_target, col, color_sets):
                            # Move v to c_target
                            moved[v] = c_target
                            found = True
                            break
                    
                    if not found:
                        success = False
                        break
                
                if success:
                    # Apply moves
                    for v, c_new in moved.items():
                        col[v] = c_new
                        color_sets[c_new].add(v)
                    
                    # Clear c_to_remove
                    color_sets[c_to_remove] = set()
                    active_colors.remove(c_to_remove)
                    removed_any = True
                    stagnant_passes = 0
                    break # Restart the pass as active_colors changed
                
            if not removed_any:
                stagnant_passes += 1
                if stagnant_passes > 5:
                    # Shake: Move a random vertex to a random valid color to escape local min
                    # Pick a random vertex
                    v = random.Random(pass_count).randrange(m)
                    current_c = col[v]
                    # Find all valid colors for v
                    valid_cs = []
                    for c in active_colors:
                        if c == current_c:
                            continue
                        if can_take_color(v, c, col, color_sets):
                            valid_cs.append(c)
                    
                    if valid_cs:
                        new_c = random.Random(pass_count + 1).choice(valid_cs)
                        # Move v
                        color_sets[current_c].remove(v)
                        color_sets[new_c].add(v)
                        col[v] = new_c
                        stagnant_passes = 0
                    else:
                        # Cannot move any vertex? Try to move a different one or give up
                        # If we are really stuck, break
                        break

        # Final compression
        unique_colors = sorted(set(col))
        color_map = {c: i for i, c in enumerate(unique_colors)}
        final_col = [color_map[c] for c in col]
        
        results.append(final_col)

    return results
