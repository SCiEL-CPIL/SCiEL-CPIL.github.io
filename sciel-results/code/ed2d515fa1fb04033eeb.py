import time
import random

def construct():
    start_time = time.time()
    hard_deadline = start_time + 110.0
    rng = random.Random(42)

    def verify(n, colors):
        """Verify that no color class contains a C6."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in range(4):
            adj_list = [[] for _ in range(n)]
            for i in range(n):
                mask = adj[c][i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    adj_list[i].append(j)
                    mask -= lsb
            
            for u in range(n):
                for v in adj_list[u]:
                    if v <= u: continue
                    for w in adj_list[v]:
                        if w == u: continue
                        for x in adj_list[w]:
                            if x == u or x == v: continue
                            for y in adj_list[x]:
                                if y == u or y == v or y == w: continue
                                for z in adj_list[y]:
                                    if z == u or z == v or z == w or z == x: continue
                                    if (adj[c][z] >> u) & 1:
                                        return False
        return True

    def count_c6s(n, colors):
        """Count total number of C6s across all color classes."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        total = 0
        for c in range(4):
            adj_list = [[] for _ in range(n)]
            for i in range(n):
                mask = adj[c][i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    adj_list[i].append(j)
                    mask -= lsb
            
            for u in range(n):
                for v in adj_list[u]:
                    if v <= u: continue
                    for w in adj_list[v]:
                        if w == u: continue
                        for x in adj_list[w]:
                            if x == u or x == v: continue
                            for y in adj_list[x]:
                                if y == u or y == v or y == w: continue
                                for z in adj_list[y]:
                                    if z == u or z == v or z == w or z == x: continue
                                    if (adj[c][z] >> u) & 1:
                                        total += 1
        return total

    def edge_idx(n, i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    def count_c6s_through_edge(n, adj, u, v, c):
        """Count C6s in color c that include edge (u,v). 
        A C6 through (u,v) is a path of length 5 from u to v in adj[c] (excluding the direct edge).
        We count paths u-x1-x2-x3-x4-v where all vertices are distinct.
        """
        # For each path u -> x1 -> x2 -> x3 -> x4 -> v with distinct internal vertices
        count = 0
        # Try all x1 adjacent to u (not v)
        mask_x1 = adj[c][u] & ~(1 << v)
        while mask_x1:
            lsb1 = mask_x1 & -mask_x1
            x1 = lsb1.bit_length() - 1
            # x2 adjacent to x1, not u, not x1
            mask_x2 = adj[c][x1] & ~(1 << u) & ~(1 << x1)
            while mask_x2:
                lsb2 = mask_x2 & -mask_x2
                x2 = lsb2.bit_length() - 1
                # x3 adjacent to x2, not u, x1, x2
                mask_x3 = adj[c][x2] & ~(1 << u) & ~(1 << x1) & ~(1 << x2)
                while mask_x3:
                    lsb3 = mask_x3 & -mask_x3
                    x3 = lsb3.bit_length() - 1
                    # x4 adjacent to x3, not u, x1, x2, x3
                    mask_x4 = adj[c][x3] & ~(1 << u) & ~(1 << x1) & ~(1 << x2) & ~(1 << x3)
                    while mask_x4:
                        lsb4 = mask_x4 & -mask_x4
                        x4 = lsb4.bit_length() - 1
                        # Check if x4 is adjacent to v
                        if (adj[c][x4] >> v) & 1:
                            count += 1
                        mask_x4 -= lsb4
                    mask_x3 -= lsb3
                mask_x2 -= lsb2
            mask_x1 -= lsb1
        return count

    def greedy_construct(n, seed=0):
        """Greedy construction: assign edges in random order, pick a color that doesn't create C6."""
        r = random.Random(seed)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        r.shuffle(edges)
        colors = [-1] * len(edges)
        adj = [[0] * n for _ in range(4)]
        
        for ei, (u, v) in enumerate(edges):
            placed = False
            color_order = list(range(4))
            r.shuffle(color_order)
            for c in color_order:
                # Check if adding (u,v) with color c creates a C6
                # C6 through (u,v): need path of length 5 from u to v in adj[c]
                found = False
                # DFS for path u -> x1 -> x2 -> x3 -> x4 -> v
                stack = [(u, 0, 1 << u)]
                while stack and not found:
                    cur, depth, visited = stack.pop()
                    if depth == 5:
                        if (adj[c][cur] >> v) & 1:
                            found = True
                        continue
                    mask = adj[c][cur] & ~visited
                    while mask:
                        lsb = mask & -mask
                        nxt = lsb.bit_length() - 1
                        if nxt == v and depth < 4:
                            pass  # can't use v before final step
                        elif nxt != v:
                            stack.append((nxt, depth + 1, visited | lsb))
                        mask -= lsb
                
                if not found:
                    colors[ei] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    placed = True
                    break
            
            if not placed:
                return None
        
        # Convert to lexicographic order
        result = [0] * len(edges)
        for ei, (u, v) in enumerate(edges):
            li = edge_idx(n, u, v)
            result[li] = colors[ei]
        return result

    def build_adj(n, colors):
        """Build adjacency bitmasks from colors array."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def local_search(n, colors, max_iters=50000):
        """Local search with incremental C6 counting: try flipping edge colors to reduce C6 count."""
        num_edges = n * (n - 1) // 2
        colors = colors[:]
        adj = build_adj(n, colors)
        
        # Compute initial total C6 count
        current_count = count_c6s(n, colors)
        if current_count == 0:
            return colors
        
        best_colors = colors[:]
        best_count = current_count
        
        for it in range(max_iters):
            if time.time() > hard_deadline:
                break
            
            # Pick a random edge to flip
            ei = rng.randint(0, num_edges - 1)
            
            # Find u, v for this edge index
            # edge_idx(n, u, v) = u*(n-1) - u*(u+1)//2 + (v-u-1)
            # We need to reverse this. Precompute edge list.
            # Actually, let's just iterate to find u,v from ei
            # For efficiency, precompute edge list
            # For now, compute on the fly (slow but correct)
            u = 0
            v = 0
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    if idx == ei:
                        u = i
                        v = j
                        break
                    idx += 1
                if idx == ei:
                    break
            
            old_c = colors[ei]
            new_c = rng.randint(0, 3)
            if new_c == old_c:
                continue
            
            # Compute delta: C6s added by new edge minus C6s removed by old edge
            # C6s removed: count C6s through (u,v) in old_c, excluding the edge itself
            # But we need to be careful: when we remove edge (u,v) from old_c, 
            # some C6s that included this edge are removed.
            # C6s added: count C6s through (u,v) in new_c, with the edge present.
            
            # For old_c: count C6s that include edge (u,v) in the current adj
            # These are paths u-x1-x2-x3-x4-v in adj[old_c] where x1..x4 are distinct and not u,v
            # But note: the edge (u,v) is currently in adj[old_c], so we need to exclude it
            # Actually, count_c6s_through_edge counts paths that don't use the direct edge,
            # so it correctly counts C6s that include the edge.
            
            # Temporarily remove edge from old_c
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            
            # Count C6s through (u,v) in old_c after removal (should be 0 for the edge itself,
            # but other C6s through this edge position are counted)
            # Actually, we want: how many C6s disappear when we remove edge (u,v) from old_c?
            # That's the number of C6s in old_c that include edge (u,v).
            # With the edge removed, those C6s are gone. So delta_removed = count of C6s through (u,v) in old_c (with edge present).
            
            # Let me recompute: before flip, adj[old_c] has edge (u,v).
            # C6s through (u,v) in old_c = count_c6s_through_edge(n, adj, u, v, old_c)
            # But this function assumes the edge is NOT in adj (it looks for paths of length 5).
            # So I need to call it BEFORE removing the edge.
            
            # Let me redo:
            # Restore edge temporarily
            adj[old_c][u] |= (1 << v)
            adj[old_c][v] |= (1 << u)
            
            # Count C6s through (u,v) in old_c (edge present, looking for length-5 paths)
            removed_c6s = count_c6s_through_edge(n, adj, u, v, old_c)
            
            # Now actually remove edge from old_c
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            
            # Add edge to new_c
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            
            # Count C6s through (u,v) in new_c (edge present, looking for length-5 paths)
            added_c6s = count_c6s_through_edge(n, adj, u, v, new_c)
            
            new_count = current_count - removed_c6s + added_c6s
            
            if new_count <= best_count:
                best_count = new_count
                best_colors = colors[:]
                colors[ei] = new_c
                current_count = new_count
            else:
                # Revert
                adj[new_c][u] &= ~(1 << v)
                adj[new_c][v] &= ~(1 << u)
                adj[old_c][u] |= (1 << v)
                adj[old_c][v] |= (1 << u)
                # Keep colors[ei] as old_c
                if new_count > current_count + 2 or rng.random() > 0.1:
                    pass  # already reverted
                else:
                    # Accept worse move for exploration
                    colors[ei] = new_c
                    current_count = new_count
                    # adj is already updated correctly
            
            if best_count == 0:
                return best_colors
        
        return best_colors

    def try_n(n, time_budget):
        """Try to find a valid coloring for n vertices within time_budget."""
        deadline = time.time() + time_budget
        
        # Try multiple greedy + local search attempts
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Greedy construction
            colors = greedy_construct(n, seed=attempt * 7 + 13)
            if colors is None:
                continue
            
            # Check if already valid
            if verify(n, colors):
                return colors
            
            # Local search to improve
            colors = local_search(n, colors, max_iters=10000)
            
            if verify(n, colors):
                return colors
        
        return None

    # Try from n=17 downward
    for n in [17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > hard_deadline - 10:
            break
        
        remaining = hard_deadline - time.time()
        budget = min(remaining - 5, 30.0)
        if budget < 2:
            break
        
        result = try_n(n, budget)
        if result is not None:
            if verify(n, result):
                return {'n': n, 'colors': result}
        
        # If we got a good result for n, try slightly more for n+1 if time allows
        # (already handled by the loop structure)

    # Fallback: small n with simple construction
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # Use a simple pattern that avoids C6 for small n
            colors.append((i + j) % 4)
    if verify(n, colors):
        return {'n': n, 'colors': colors}
    
    # Absolute fallback
    return {'n': 1, 'colors': []}

if __name__ == '__main__':
    construct()
