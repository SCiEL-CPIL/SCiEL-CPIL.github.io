"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

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

    def solve_instance(edges, n, time_limit=35):
        E = len(edges)
        if E == 0:
            return []
        
        # Build adjacency sets for fast neighbor lookup
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Precompute conflict matrix: conflict[i][j] is True if edge i and edge j cannot share a color
        # Two edges conflict if they share a vertex or if any endpoint of one is adjacent to any endpoint of the other.
        # Let edge i = (u1, v1) and edge j = (u2, v2).
        # They conflict if:
        # 1. u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2 (share vertex)
        # 2. u2 in adj[u1] or u2 in adj[v1] or v2 in adj[u1] or v2 in adj[v1] (joined by edge)
        
        # Optimization: Use bitmasks or sets? For N=80, E is small (~200). List of lists is fine.
        conflict = [[False] * E for _ in range(E)]
        for i in range(E):
            u1, v1 = edges[i]
            for j in range(i + 1, E):
                u2, v2 = edges[j]
                # Check shared vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i][j] = True
                    conflict[j][i] = True
                else:
                    # Check if joined by an edge
                    if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                        conflict[i][j] = True
                        conflict[j][i] = True
                    # Note: if they don't share vertex and aren't joined, they are independent.

        # Greedy Initialization
        # Order edges by degree of endpoints (descending) to place constrained edges first
        edge_order = list(range(E))
        # Calculate "weight" for each edge: sum of degrees of its endpoints
        edge_weights = [len(adj[edges[i][0]]) + len(adj[edges[i][1]]) for i in range(E)]
        edge_order.sort(key=lambda x: edge_weights[x], reverse=True)
        
        colors = [-1] * E
        used_colors = 0
        
        # For each color class, maintain the set of edges in it to quickly check conflicts
        # Since we are adding one edge at a time, we can just check against existing edges in candidate color classes.
        
        for idx in edge_order:
            # Find the smallest color c such that no edge in color class c conflicts with edge idx
            assigned = False
            # We don't know max colors yet, so we iterate up to current used_colors
            for c in range(used_colors):
                # Check if edge idx conflicts with any edge already in color c
                # We need a list of edges per color
                # Instead of maintaining lists, we can just scan all edges? No, too slow.
                # Let's maintain a list of edges per color.
                pass
            
            # Actually, maintaining lists is easier for the greedy step.
            # Let's restart the greedy logic with lists.
            
        color_classes = [] # List of lists of edge indices
        colors = [-1] * E
        
        for idx in edge_order:
            placed = False
            # Try existing colors
            for c in range(len(color_classes)):
                # Check if idx conflicts with any edge in color_classes[c]
                ok = True
                for j in color_classes[c]:
                    if conflict[idx][j]:
                        ok = False
                        break
                if ok:
                    color_classes[c].append(idx)
                    colors[idx] = c
                    placed = True
                    break
            if not placed:
                color_classes.append([idx])
                colors[idx] = len(color_classes) - 1
        
        # Local Search: Hill Climbing
        # Objective: Minimize the number of colors (max color index + 1)
        # Strategy: Try to move an edge to a lower color index if possible.
        # If stuck, try to merge two color classes? Or just minimize max color.
        
        # Let's use a standard local search: 
        # 1. Identify the highest color used.
        # 2. Try to move edges from the highest color to lower colors.
        # 3. If successful, the highest color is empty, reducing total colors.
        # 4. Repeat until no more reductions or time limit.
        
        # We need to be able to check conflicts quickly.
        # For a given edge i and color c, check if any edge j in color c conflicts with i.
        
        # To speed up, we can maintain for each color c, a bitmask or set of vertices involved?
        # Induced matching constraint: 
        # If edge (u,v) is in color c, then no other edge in color c can touch u or v, 
        # nor can any edge in color c have an endpoint adjacent to u or v.
        
        # Let's stick to the conflict matrix check, it's O(E) per move.
        
        start_time = time.time()
        max_iterations = 100000
        iterations = 0
        
        while time.time() - start_time < time_limit:
            iterations += 1
            if iterations > max_iterations: break
            
            # Find the highest color index used
            max_c = -1
            for c in range(len(color_classes)):
                if color_classes[c]:
                    max_c = c
            
            if max_c == -1:
                break
                
            # Try to move an edge from color max_c to a lower color
            moved = False
            # Shuffle edges in max_c to avoid bias
            edges_in_max = color_classes[max_c][:]
            random.Random(iterations).shuffle(edges_in_max)
            
            for i in edges_in_max:
                # Try moving i to color c < max_c
                for c in range(max_c):
                    # Check if i can go to c
                    ok = True
                    for j in color_classes[c]:
                        if conflict[i][j]:
                            ok = False
                            break
                    if ok:
                        # Move i from max_c to c
                        color_classes[max_c].remove(i)
                        color_classes[c].append(i)
                        colors[i] = c
                        moved = True
                        break
                if moved:
                    break
            
            if not moved:
                # If we couldn't reduce the max color, try to improve the distribution?
                # Or just stop, as we are minimizing the count.
                # To escape local minima, we could try a random swap between two colors?
                # But that doesn't reduce the count.
                # Let's try a "shake" move: pick two edges from different colors, swap them if valid?
                # This might help rearrange to allow future reductions.
                # However, for this specific problem, greedy + simple reduction is often very strong.
                # Let's try a random re-coloring of a small subset?
                # Given time constraints, let's just break if we can't reduce max color.
                # But we might be stuck at a local minimum where max color is 10, but 8 is possible.
                # A more robust approach: Simulated Annealing on the number of colors?
                # Since the directive asks for a NEW approach, and SA is the current one, 
                # let's try **Iterated Local Search (ILS)** with random restarts if stuck.
                
                # Actually, let's just break. The greedy initialization is usually quite good.
                # If we want to be more robust, we can try to merge colors.
                # Let's try to find two colors c1, c2 such that their union is an induced matching.
                # If so, merge them.
                
                merged = False
                # Try merging colors 0..max_c-1 into max_c? Or any pair?
                # Merging reduces color count by 1.
                # Check if color_classes[c1] U color_classes[c2] is conflict-free.
                # This is O(E^2) per pair.
                
                # We only need to check if we can reduce the TOTAL number of colors.
                # If we can merge any two non-empty colors, the count drops.
                
                for c1 in range(len(color_classes)):
                    if not color_classes[c1]: continue
                    for c2 in range(c1 + 1, len(color_classes)):
                        if not color_classes[c2]: continue
                        
                        # Check if all edges in c1 are compatible with all edges in c2
                        ok = True
                        for i in color_classes[c1]:
                            for j in color_classes[c2]:
                                if conflict[i][j]:
                                    ok = False
                                    break
                            if not ok: break
                        if ok:
                            # Merge c2 into c1
                            color_classes[c1].extend(color_classes[c2])
                            for i in color_classes[c2]:
                                colors[i] = c1
                            color_classes[c2] = []
                            merged = True
                            break
                    if merged:
                        break
                
                if merged:
                    continue
                
                # If no merge and no move, we are stuck.
                # Try a random restart of the local search phase with a perturbation?
                # Or just break.
                break

        # Compact colors
        final_colors = [0] * E
        mapping = {}
        next_c = 0
        for c in range(len(color_classes)):
            if color_classes[c]:
                mapping[c] = next_c
                next_c += 1
        
        for i in range(E):
            final_colors[i] = mapping[colors[i]]
            
        return final_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        colors = solve_instance(edges, n, time_limit=35)
        out.append(colors)
    return out
# EVOLVE-BLOCK-END
