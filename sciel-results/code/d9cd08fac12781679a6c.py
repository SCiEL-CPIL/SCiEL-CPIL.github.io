import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    def luw_construct(k, g, d):
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        # Rule generation: for each level c from 2 to d, pick (a, b) with 1 <= a < c, 1 <= b < c
        # The standard LUW rule uses a specific pattern. For k=3, g=16, d=7 works well.
        # Rule: at level c, y[c] = x[c] + y[a]*x[b] (side 0) or x[c] - x[a]*y[b] (side 1)
        # We need to choose a, b such that the resulting graph has girth >= g.
        # Standard approach: use the "cage" rule where a and b are chosen to maximize girth.
        
        # For the standard LUW construction, the rule is typically:
        # At level c, use indices that create a "Latin square" like structure.
        # A common choice: a = c-1, b = 1, but this may not give optimal girth.
        # Better: use the rule from the original LUW paper.
        
        # Let me try a systematic approach: for each c, try different (a,b) pairs
        # and verify the girth.
        
        def build_and_check(rules):
            """Build the LUW graph with given rules and check girth."""
            def neighbours(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0] * (d - 1)
                    for c in range(2, d + 1):
                        a, b = rules[c]
                        if side == 0:
                            y[c] = add(x[c], mul(y[a], x[b]))
                        else:
                            y[c] = sub(x[c], mul(x[a], y[b]))
                    yield 1 - side, tuple(y)

            start = (0, (None,) + (0,) * d)
            index, order = {start: 0}, [start]
            adj = {}
            for vertex in order:
                idx = index[vertex]
                adj[idx] = []
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    adj[idx].append(index[other])
            
            n = len(order)
            
            # Check girth using BFS from each vertex
            # For girth g, we need no cycle shorter than g.
            # Check for cycles of length 4, 6, 8, ..., g-1 (only even cycles matter for bipartite-like structures,
            # but LUW graphs may have odd cycles too)
            # Actually, check all even lengths from 4 to g-1, and odd from 3 to g-1.
            # But for efficiency, just check up to g-1.
            
            # Quick check: BFS from each vertex, track parent, detect back edges
            for u in range(n):
                visited = [-1] * n
                parent = [-1] * n
                visited[u] = 0
                q_bfs = deque([(u, 0)])
                while q_bfs:
                    v, dist = q_bfs.popleft()
                    if dist >= g - 1:
                        continue
                    for w in adj[v]:
                        if visited[w] == -1:
                            visited[w] = dist + 1
                            parent[w] = v
                            q_bfs.append((w, dist + 1))
                        elif w != parent[v] and v != parent[w]:
                            # Found a cycle
                            cycle_len = dist + 1 - visited[w]
                            if cycle_len < g:
                                return False, None
            return True, (n, adj)

        # Generate rules
        # For k=3, g=16, d=7, let's try a known good rule set.
        # The rule at level c: (a, b) where 1 <= a, b < c
        # A standard choice from the literature for cages:
        # c=2: (1,1) - but a < c means a=1, b=1 is the only option
        # c=3: (1,2) or (2,1)
        # c=4: (1,2), (1,3), (2,1), (2,3), (3,1), (3,2)
        # etc.
        
        # Let me try the "greedy" rule: at each level c, use (c-1, 1)
        # This is a common default.
        rules = {}
        for c in range(2, d + 1):
            if c == 2:
                rules[c] = (1, 1)
            else:
                rules[c] = (c - 1, 1)
        
        valid, result = build_and_check(rules)
        if valid:
            n, adj = result
            edges = []
            seen = set()
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        
        # If the default rule doesn't work, try other combinations
        # For d=7, there are 6 levels (2..7), each with (c-1) choices for a and (c-1) choices for b
        # That's a lot of combinations. Let's try a few systematic ones.
        
        # Try (1, c-1) instead
        rules2 = {}
        for c in range(2, d + 1):
            if c == 2:
                rules2[c] = (1, 1)
            else:
                rules2[c] = (1, c - 1)
        
        valid, result = build_and_check(rules2)
        if valid:
            n, adj = result
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        
        # Try mixed rules
        # For c=3: (2,1), c=4: (3,1), c=5: (4,1), c=6: (5,1), c=7: (6,1) - already tried
        # Let me try: c=3: (1,2), c=4: (2,1), c=5: (3,2), c=6: (4,1), c=7: (5,3)
        # This is getting complex. Let me just try all combinations for small d.
        
        if d <= 7:
            import itertools
            # Levels 2 to d
            levels = list(range(2, d + 1))
            # For each level c, possible (a,b) pairs with 1 <= a < c, 1 <= b < c
            choices = []
            for c in levels:
                if c == 2:
                    choices.append([(1, 1)])
                else:
                    choices.append([(a, b) for a in range(1, c) for b in range(1, c)])
            
            # Try all combinations, but limit to a reasonable number
            count = 0
            for combo in itertools.product(*choices):
                if time.time() > deadline:
                    break
                rules_try = {}
                for i, c in enumerate(levels):
                    rules_try[c] = combo[i]
                
                valid, result = build_and_check(rules_try)
                if valid:
                    n, adj = result
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges
                count += 1
                if count > 10000:
                    break
        
        # Fallback: use the original rule from the seed
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        valid, result = build_and_check(rule)
        if valid:
            n, adj = result
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        
        # If nothing works, use a larger d
        return luw_construct(k, g, d + 2)

    if k != 3 or g != 16:
        # For other parameters, use the original approach
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        return luw_construct(k, g, d)

    # For k=3, g=16, use d=7 which gives ~4k vertices
    result = luw_construct(3, 16, 7)
    return result
