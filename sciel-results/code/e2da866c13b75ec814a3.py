import time
import random

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def verify(n, colors):
        """Verify that no color class contains a C6."""
        # Build adjacency per color
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        # For each color, check for C6 using DFS from each edge
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[c][i] >> j) & 1:
                        continue
                    # Check if there's a path of length 4 from i to j avoiding the direct edge
                    # i - a - b - d - e - j where all distinct
                    # Use DFS with small depth
                    def has_path(i, j, c, adj, n):
                        # Find path i -> a -> b -> d -> e -> j, all distinct
                        # a in N(i), b in N(a) \ {i}, d in N(b) \ {i,a}, e in N(d) \ {i,a,b}, j in N(e) \ {i,a,b,d}
                        for a in range(n):
                            if a == i or a == j:
                                continue
                            if not (adj[c][i] >> a) & 1:
                                continue
                            for b in range(n):
                                if b == i or b == j or b == a:
                                    continue
                                if not (adj[c][a] >> b) & 1:
                                    continue
                                for d in range(n):
                                    if d == i or d == j or d == a or d == b:
                                        continue
                                    if not (adj[c][b] >> d) & 1:
                                        continue
                                    for e in range(n):
                                        if e == i or e == j or e == a or e == b or e == d:
                                            continue
                                        if not (adj[c][d] >> e) & 1:
                                            continue
                                        if (adj[c][e] >> j) & 1:
                                            return True
                        return False
                    if has_path(i, j, c, adj, n):
                        return False
        return True

    def small_fallback(n):
        """Generate a valid coloring for small n using a simple pattern."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                # Use a pattern that avoids C6 for small n
                colors.append((i * 3 + j * 2) % 4)
        return colors

    def attempt(n, time_budget):
        """
        Backtracking search for 4-coloring of K_n without monochromatic C6.
        Uses iterative DFS with bitmask adjacency.
        """
        local_deadline = time.time() + time_budget
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]

        def has_path_len_5(c, u, v):
            """Check for simple path of length 5 from u to v in color c."""
            # u - x1 - x2 - x3 - x4 - v
            # Iterative DFS with visited mask
            # State: (node, depth_remaining, visited_mask)
            stack = [(u, 5, 1 << u)]
            while stack:
                cur, depth, visited = stack.pop()
                if depth == 0:
                    if cur == v:
                        return True
                    continue
                if depth == 1:
                    if (adj[c][cur] >> v) & 1:
                        return True
                    continue
                mask = adj[c][cur] & ~visited
                while mask:
                    lsb = mask & -mask
                    nxt = lsb.bit_length() - 1
                    stack.append((nxt, depth - 1, visited | lsb))
                    mask -= lsb
            return False

        def is_valid(c, u, v):
            return not has_path_len_5(c, u, v)

        # Iterative backtracking
        # Stack entries: (edge_idx, color_to_try)
        # We maintain current state in colors and adj
        # Use a stack of (idx, next_color) where next_color is the next color to try
        # When we advance, we push the next state

        # Better: use a list as stack of (idx, color)
        # colors[idx] is set when we push, and unset when we pop
        # We need to track which color was assigned at each level

        # Let's use a different approach:
        # stack of (idx, colors_tried_mask)
        # colors_tried_mask: bitmask of colors already tried at this edge

        stack = [(0, 0)]  # (edge_index, colors_tried_mask)
        while stack:
            idx, tried_mask = stack[-1]

            if time.time() > local_deadline:
                return None

            if idx == num_edges:
                return colors[:]

            u, v = edges[idx]
            found = False
            for c in range(4):
                cbit = 1 << c
                if tried_mask & cbit:
                    continue
                if is_valid(c, u, v):
                    # Assign color c to edge idx
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    # Update top of stack with new tried mask
                    stack[-1] = (idx, tried_mask | cbit)
                    # Push next edge
                    stack.append((idx + 1, 0))
                    found = True
                    break

            if not found:
                # Backtrack
                stack.pop()
                if idx > 0:
                    prev_idx = idx - 1
                    prev_u, prev_v = edges[prev_idx]
                    prev_c = colors[prev_idx]
                    colors[prev_idx] = -1
                    adj[prev_c][prev_u] &= ~(1 << prev_v)
                    adj[prev_c][prev_v] &= ~(1 << prev_u)

        return None

    # Try n=17 first with most time
    res = attempt(17, 55)
    if res:
        if verify(17, res):
            return {'n': 17, 'colors': res}

    # Try n=16
    if time.time() < hard_deadline - 35:
        res = attempt(16, 30)
        if res:
            if verify(16, res):
                return {'n': 16, 'colors': res}

    # Try n=15
    if time.time() < hard_deadline - 15:
        res = attempt(15, 12)
        if res:
            if verify(15, res):
                return {'n': 15, 'colors': res}

    # Try n=14
    if time.time() < hard_deadline - 8:
        res = attempt(14, 6)
        if res:
            if verify(14, res):
                return {'n': 14, 'colors': res}

    # Fallback to small n
    for n in [12, 10, 9, 8]:
        if time.time() < hard_deadline - 2:
            res = attempt(n, 3)
            if res:
                if verify(n, res):
                    return {'n': n, 'colors': res}

    # Absolute fallback
    n = 8
    colors = small_fallback(n)
    if verify(n, colors):
        return {'n': n, 'colors': colors}

    n = 7
    colors = small_fallback(n)
    return {'n': n, 'colors': colors}
