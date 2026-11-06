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
                # Quick check: BFS/DFS for path of length 5
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

    def local_search(n, colors, max_iters=50000):
        """Local search: try flipping edge colors to reduce C6 count."""
        num_edges = n * (n - 1) // 2
        colors = colors[:]
        
        def get_c6_count(cols):
            return count_c6s(n, cols)
        
        current_count = get_c6_count(colors)
        if current_count == 0:
            return colors
        
        best_colors = colors[:]
        best_count = current_count
        
        for it in range(max_iters):
            if time.time() > hard_deadline:
                break
            
            # Pick a random edge to flip
            ei = rng.randint(0, num_edges - 1)
            old_c = colors[ei]
            new_c = rng.randint(0, 3)
            if new_c == old_c:
                continue
            
            colors[ei] = new_c
            new_count = get_c6_count(colors)
            
            if new_count <= best_count:
                best_count = new_count
                best_colors = colors[:]
            else:
                # Revert with some probability (simulated annealing style)
                # Always revert if worse, but occasionally accept to escape local minima
                if new_count > current_count + 2 or rng.random() > 0.1:
                    colors[ei] = old_c
                else:
                    current_count = new_count
            
            if best_count == 0:
                return best_colors
            
            # Reset current_count tracking
            if colors == best_colors:
                current_count = best_count
        
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
