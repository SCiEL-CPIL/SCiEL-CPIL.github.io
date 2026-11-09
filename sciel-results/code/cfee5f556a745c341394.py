import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

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

    def build_incompatibility(edges, adj):
        m = len(edges)
        incompatible = [set() for _ in range(m)]
        # Pre-compute for faster lookup: map edge index to its endpoints
        # And map vertex to set of incident edge indices
        vert_to_edges = defaultdict(set)
        for i, (u, v) in enumerate(edges):
            vert_to_edges[u].add(i)
            vert_to_edges[v].add(i)
        
        for i in range(m):
            u1, v1 = edges[i]
            # Edges sharing a vertex
            for v in (u1, v1):
                for j in vert_to_edges[v]:
                    if j != i:
                        incompatible[i].add(j)
                        incompatible[j].add(i)
            
            # Edges connected by an edge (distance 2)
            # For each endpoint of edge i, look at its neighbors in the original graph
            for v in (u1, v1):
                for w in adj[v]:
                    # w is a vertex adjacent to v.
                    # Any edge incident to w conflicts with edge i, UNLESS that edge is (v, w) itself?
                    # If edge j is (v, w), it shares vertex v with i. Already handled above.
                    # So we only care about edges incident to w that do NOT share a vertex with i.
                    for j in vert_to_edges[w]:
                        if j == i:
                            continue
                        u2, v2 = edges[j]
                        # If j shares a vertex with i, it's already in incompatible[i]
                        # We only add it here if it's a "distance 2" conflict only
                        # Actually, just adding it again to the set is fine (idempotent)
                        # But we can skip if already present to save tiny bit of time? Set add is fast.
                        incompatible[i].add(j)
                        incompatible[j].add(i)
        
        return incompatible

    def dsatur_coloring(incompatible, m):
        """DSATUR: color vertex with highest saturation first, tie-break by degree."""
        colors = [-1] * m
        saturation = [set() for _ in range(m)]
        degree = [len(incompatible[i]) for i in range(m)]
        colored = [False] * m
        
        for _ in range(m):
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in range(m):
                if not colored[v]:
                    sat = len(saturation[v])
                    if sat > best_sat or (sat == best_sat and degree[v] > best_deg):
                        best_v = v
                        best_sat = sat
                        best_deg = degree[v]
            
            if best_v == -1:
                break
            
            used = set(saturation[best_v])
            c = 0
            while c in used:
                c += 1
            colors[best_v] = c
            colored[best_v] = True
            for u in incompatible[best_v]:
                if not colored[u]:
                    saturation[u].add(c)
        
        return colors

    def count_colors(colors):
        return max(colors) + 1 if colors else 0

    def get_max_color(colors):
        return max(colors) if colors else 0

    def is_valid_coloring(colors, incompatible, m):
        # Check if any two edges with same color are incompatible
        color_to_edges = defaultdict(list)
        for i, c in enumerate(colors):
            color_to_edges[c].append(i)
        
        for c, edges_in_class in color_to_edges.items():
            for i in range(len(edges_in_class)):
                e1 = edges_in_class[i]
                for j in range(i + 1, len(edges_in_class)):
                    e2 = edges_in_class[j]
                    if e2 in incompatible[e1]:
                        return False
        return True

    def simulated_annealing(incompatible, initial_colors, m, time_budget):
        best_colors = initial_colors[:]
        best_k = count_colors(best_colors)
        current_colors = best_colors[:]
        
        # Pre-compute list of edges for each color to speed up checks?
        # For now, we rely on the incompatible sets.
        
        rng = random.Random(42)
        start = time.time()
        
        # Temperature schedule
        T = 1.0
        cooling_rate = 0.99995 # Adjust based on expected iterations
        
        # We want to minimize max color.
        # Move types:
        # 1. Recolor an edge with the current max color to a lower valid color.
        # 2. Swap colors of two edges to potentially enable recoloring.
        
        while time.time() - start < time_budget:
            max_c = get_max_color(current_colors)
            
            # Identify edges with max color
            max_edges = [i for i in range(m) if current_colors[i] == max_c]
            
            if not max_edges:
                break
            
            move_made = False
            improvement = False
            
            # Try to reduce max color
            if max_edges:
                v = rng.choice(max_edges)
                # Try to recolor v to a color < max_c
                # We prefer colors that are already used to keep the count low, 
                # but any color < max_c that is valid is good.
                # Let's try all valid colors < max_c
                valid_colors = []
                for c in range(max_c):
                    conflict = False
                    for u in incompatible[v]:
                        if current_colors[u] == c:
                            conflict = True
                            break
                    if not conflict:
                        valid_colors.append(c)
                
                if valid_colors:
                    # Choose a random valid color
                    new_c = rng.choice(valid_colors)
                    current_colors[v] = new_c
                    move_made = True
                    # Check if max color reduced
                    new_max_c = get_max_color(current_colors)
                    if new_max_c < max_c:
                        # Improvement!
                        best_colors = current_colors[:]
                        best_k = new_max_c + 1
                        improvement = True
                    # If new_max_c == max_c, it's a neutral move (shuffling within max color class or moving to lower)
                    # We accept it if it helps future moves? 
                    # Since we are minimizing max color, moving from max to lower is always "good" or neutral.
                    # It reduces the number of edges in the max color class, which might help eliminate it entirely.
                    # So we always accept this move.
                else:
                    # Cannot recolor v to lower color. Try swapping?
                    # Pick another edge u (not necessarily max color) and swap colors of v and u?
                    # If we swap v (max_c) with u (c_u), v becomes c_u. 
                    # This is valid if c_u is valid for v (which it is, since u was using it and they are not adjacent? No, they might be adjacent).
                    # Actually, if we swap, we need to check validity of the new configuration.
                    # Let's try a simple swap with a random edge u.
                    u = rng.randrange(m)
                    if u == v:
                        continue
                    c_v = current_colors[v]
                    c_u = current_colors[u]
                    if c_v == c_u:
                        continue
                    
                    # Check if swap is valid
                    # v takes c_u: must not conflict with any neighbor of v having color c_u
                    ok_v = True
                    for w in incompatible[v]:
                        if w == u: continue
                        if current_colors[w] == c_u:
                            ok_v = False
                            break
                    if not ok_v:
                        continue
                        
                    # u takes c_v: must not conflict with any neighbor of u having color c_v
                    ok_u = True
                    for w in incompatible[u]:
                        if w == v: continue
                        if current_colors[w] == c_v:
                            ok_u = False
                            break
                    if not ok_u:
                        continue
                        
                    # Perform swap
                    current_colors[v] = c_u
                    current_colors[u] = c_v
                    move_made = True
                    
                    new_max_c = get_max_color(current_colors)
                    if new_max_c < best_k - 1: # best_k is count, so max index is best_k-1
                         # Wait, best_k is count. Current max index is new_max_c.
                         # If new_max_c + 1 < best_k, we improved.
                         pass
                    if new_max_c + 1 < best_k:
                        best_colors = current_colors[:]
                        best_k = new_max_c + 1
                        improvement = True
                    # Accept swap if it doesn't increase max color?
                    # If new_max_c <= max_c, it's safe.
                    # If new_max_c > max_c, it's bad. But we only swapped with c_u which was < max_c?
                    # If u had color < max_c, then v (which had max_c) now has < max_c.
                    # u now has max_c. So max color remains max_c.
                    # So swap is neutral regarding max color, but changes distribution.
                    # We accept it.

            if not move_made:
                # Stuck? Try a random valid recolor of any edge to shuffle.
                # Pick a random edge v
                v = rng.randrange(m)
                c_v = current_colors[v]
                # Try to change to a different valid color
                # We can try all colors 0..max_c
                tried = False
                for c in range(get_max_color(current_colors) + 1):
                    if c == c_v: continue
                    conflict = False
                    for u in incompatible[v]:
                        if current_colors[u] == c:
                            conflict = True
                            break
                    if not conflict:
                        current_colors[v] = c
                        tried = True
                        break
                if not tried:
                    # No move possible for v, try another edge
                    pass

            # Cooling
            T *= cooling_rate
            if T < 0.01:
                T = 0.01

        return best_colors

    out = []
    for idx, spec in enumerate((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        remaining = deadline - time.time()
        if remaining <= 2:
            # Fallback: just use greedy
            edges = instance(*spec)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            m = len(edges)
            incompatible = build_incompatibility(edges, adj)
            colors = dsatur_coloring(incompatible, m)
            out.append(colors)
            continue
        
        # Allocate time
        # First instance gets more time, subsequent less
        if idx == 0:
            time_budget = remaining * 0.45
        elif idx == 1:
            time_budget = remaining * 0.35
        else:
            time_budget = remaining * 0.20
        
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        m = len(edges)
        incompatible = build_incompatibility(edges, adj)
        
        # Initial coloring
        colors = dsatur_coloring(incompatible, m)
        
        # Optimize
        colors = simulated_annealing(incompatible, colors, m, time_budget)
        
        out.append(colors)
        
    return out

if __name__ == "__main__":
    result = construct()
    # Quick check
    for i, r in enumerate(result):
        if r:
            print(f"Instance {i}: {max(r)+1} colors")
