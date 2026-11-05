"""Strong edge colourings via Minimum Sum Coloring with DSATUR branch-and-bound."""
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

    def build_conflict_bitmasks(edges, n):
        """Build conflict graph as list of bitmasks for fast operations."""
        m = len(edges)
        vertex_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_to_edges[u].append(i)
            vertex_to_edges[v].append(i)
        
        conflict_masks = [0] * m
        
        # Adjacency conflicts: edges sharing a vertex
        for u in range(n):
            inc_edges = vertex_to_edges[u]
            for i in range(len(inc_edges)):
                for j in range(i + 1, len(inc_edges)):
                    e_i, e_j = inc_edges[i], inc_edges[j]
                    conflict_masks[e_i] |= (1 << e_j)
                    conflict_masks[e_j] |= (1 << e_i)
        
        # Distance-2 conflicts: edges joined by an edge
        vertex_adj = [set() for _ in range(n)]
        for u, v in edges:
            vertex_adj[u].add(v)
            vertex_adj[v].add(u)
        
        for i, (u, v) in enumerate(edges):
            for nu in vertex_adj[u]:
                if nu == v:
                    continue
                for j in vertex_to_edges[nu]:
                    if j != i:
                        conflict_masks[i] |= (1 << j)
                        conflict_masks[j] |= (1 << i)
            for nv in vertex_adj[v]:
                if nv == u:
                    continue
                for j in vertex_to_edges[nv]:
                    if j != i:
                        conflict_masks[i] |= (1 << j)
                        conflict_masks[j] |= (1 << i)
        
        return conflict_masks

    def dsatur_greedy(conflict_masks, m, rng, seed_offset=0):
        """DSATUR-based greedy coloring with randomization for restarts."""
        rng_local = random.Random(seed_offset)
        colors = [-1] * m
        color_class_sizes = []  # list of bitmasks for each color class
        used_colors = set()
        
        # Order vertices by degree (static) with some randomization
        degs = [bin(mask).count('1') for mask in conflict_masks]
        order = list(range(m))
        
        if seed_offset == 0:
            order.sort(key=lambda x: -degs[x])
        else:
            # Add noise to degree
            order.sort(key=lambda x: -(degs[x] + rng_local.random() * 5))
        
        for v in order:
            # Find available colors
            blocked = set()
            mask = conflict_masks[v]
            for c, cmask in enumerate(color_class_sizes):
                if mask & cmask:
                    blocked.add(c)
            
            # Pick smallest available color
            c = 0
            while c in blocked:
                c += 1
            
            if c >= len(color_class_sizes):
                color_class_sizes.append(0)
            
            color_class_sizes[c] |= (1 << v)
            colors[v] = c
        
        return colors

    def sum_of_colors(colors):
        """Compute sum of color indices."""
        return sum(colors)

    def local_search_reduce_max_color(colors, conflict_masks, m, rng, time_limit):
        """Try to reduce the number of colors by moving vertices from highest color class."""
        max_c = max(colors) if m > 0 else -1
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for c_remove in range(max_c, -1, -1):
                if time.time() > time_limit:
                    return colors
                # Get vertices with this color
                verts = [i for i in range(m) if colors[i] == c_remove]
                if not verts:
                    continue
                
                # Try to move each vertex to a lower color
                rng.shuffle(verts)
                moved_any = False
                for v in verts:
                    # Find available lower colors
                    available = []
                    mask = conflict_masks[v]
                    for c_new in range(c_remove):
                        # Check if color c_new is used and conflicts with v
                        conflict = False
                        for u in range(m):
                            if colors[u] == c_new and (mask & (1 << u)):
                                conflict = True
                                break
                        if not conflict:
                            available.append(c_new)
                    
                    if available:
                        colors[v] = rng.choice(available)
                        moved_any = True
                
                if moved_any:
                    improved = True
                    max_c = max(colors) if m > 0 else -1
                    break
        
        return colors

    def branch_and_bound_min_sum(conflict_masks, m, time_limit, rng):
        """
        Branch-and-bound for minimum sum coloring.
        Uses DSATUR variable ordering and bitset-based pruning.
        """
        if m == 0:
            return []
        
        # Initial solution from DSATUR greedy
        best_colors = dsatur_greedy(conflict_masks, m, rng, 0)
        best_sum = sum_of_colors(best_colors)
        best_max = max(best_colors) if m > 0 else 0
        
        # Try a few randomized DSATUR restarts
        for restart in range(5):
            if time.time() > time_limit:
                break
            colors = dsatur_greedy(conflict_masks, m, rng, restart + 1)
            s = sum_of_colors(colors)
            if s < best_sum:
                best_sum = s
                best_colors = colors[:]
                best_max = max(colors) if m > 0 else 0
        
        # Local search to improve
        best_colors = local_search_reduce_max_color(best_colors, conflict_masks, m, rng, time_limit)
        best_sum = sum_of_colors(best_colors)
        best_max = max(best_colors) if m > 0 else 0
        
        # Now do branch-and-bound to try to improve further
        # State: (unassigned_mask, colors_array, current_sum)
        # Use DSATUR ordering dynamically
        
        full_mask = (1 << m) - 1
        assigned = [False] * m
        colors_arr = [-1] * m
        
        # Precompute degree ordering for tie-breaking
        degs = [bin(mask).count('1') for mask in conflict_masks]
        
        def bnb(unassigned_mask, current_sum, depth):
            nonlocal best_sum, best_colors, best_max
            
            if time.time() > time_limit:
                return
            
            if unassigned_mask == 0:
                if current_sum < best_sum:
                    best_sum = current_sum
                    best_colors = colors_arr[:]
                    best_max = max(colors_arr) if m > 0 else 0
                return
            
            # Pruning: if current_sum + lower_bound >= best_sum, prune
            # Lower bound: each unassigned vertex gets at least 0, but we can do better
            # Simple lower bound: 0 (can't improve much)
            # Better: count unassigned vertices and assume minimum color 0
            unassigned_count = bin(unassigned_mask).count('1')
            if current_sum + 0 >= best_sum:
                return
            
            # Select next vertex using DSATUR:
            # 1. Maximum saturation degree (number of different colors in neighbors)
            # 2. Tie-break by degree
            
            best_v = -1
            best_sat = -1
            best_deg = -1
            
            for v in range(m):
                if not (unassigned_mask & (1 << v)):
                    continue
                
                # Compute saturation: number of distinct colors among assigned neighbors
                sat_colors = set()
                mask = conflict_masks[v]
                for u in range(m):
                    if (mask & (1 << u)) and colors_arr[u] >= 0:
                        sat_colors.add(colors_arr[u])
                
                sat = len(sat_colors)
                deg = degs[v]
                
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return
            
            v = best_v
            mask = conflict_masks[v]
            
            # Determine available colors: 0 up to some limit
            # We can use colors 0 to best_max (or slightly more)
            blocked_colors = set()
            for u in range(m):
                if (mask & (1 << u)) and colors_arr[u] >= 0:
                    blocked_colors.add(colors_arr[u])
            
            # Try colors in order, but also try to reuse existing colors
            # Collect used colors
            used_colors = set(colors_arr)
            
            # Try existing colors first (in order), then new colors
            candidate_colors = sorted(blocked_colors.symmetric_difference(used_colors))
            # Actually, we want colors NOT in blocked_colors
            # Try colors from 0 up to best_max + 1
            for c in range(best_max + 2):
                if c in blocked_colors:
                    continue
                
                # Assign color c to v
                colors_arr[v] = c
                new_sum = current_sum + c
                
                # Prune
                if new_sum >= best_sum:
                    colors_arr[v] = -1
                    continue
                
                bnb(unassigned_mask & ~(1 << v), new_sum, depth + 1)
                
                colors_arr[v] = -1
                
                # If we found a good solution, we might not need to try higher colors
                # But for correctness, we continue
        
        # Start B&B with all vertices unassigned
        bnb(full_mask, 0, 0)
        
        return best_colors

    def solve_instance(edges, n, d, seed, time_limit):
        m = len(edges)
        if m == 0:
            return []
        
        conflict_masks = build_conflict_bitmasks(edges, n)
        rng = random.Random(seed)
        
        # Phase 1: Get good initial solutions via multiple DSATUR restarts
        best_colors = None
        best_sum = float('inf')
        
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() > time_limit:
                break
            colors = dsatur_greedy(conflict_masks, m, rng, r)
            colors = local_search_reduce_max_color(colors, conflict_masks, m, rng, time_limit)
            s = sum_of_colors(colors)
            if s < best_sum:
                best_sum = s
                best_colors = colors[:]
        
        if best_colors is None:
            best_colors = dsatur_greedy(conflict_masks, m, rng, 0)
        
        # Phase 2: Try branch-and-bound with time limit
        if time.time() < time_limit - 5.0:
            bnb_limit = time.time() + min(20.0, time_limit - time.time() - 5.0)
            bnb_colors = branch_and_bound_min_sum(conflict_masks, m, bnb_limit, rng)
            if bnb_colors is not None:
                bnb_sum = sum_of_colors(bnb_colors)
                if bnb_sum < best_sum:
                    best_colors = bnb_colors[:]
                    best_sum = bnb_sum
        
        return best_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        
        inst_start = time.time()
        inst_limit = min(inst_start + 45.0, deadline)
        
        colours = solve_instance(edges, n, d, seed, inst_limit)
        results.append(colours)
        
        if time.time() > deadline:
            break

    return results
