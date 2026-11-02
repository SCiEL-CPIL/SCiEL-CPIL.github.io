"""Rewritten program: deterministic voltage/circulant construction with girth verification."""
import time
import itertools
import math

def construct(k, g):
    start_time = time.time()
    DEADLINE = start_time + 130  # safety margin before 150s kill

    # For k=5, g=9 we want a 5-regular graph of girth >= 9.
    # Use circulant graphs on Z_n with connection set S = {a, -a, b, -b, c} or similar.
    # For odd degree 5, we need an odd-sized connection set (no self-inverse pairs except 0).
    # Connection set for 5-regular circulant: {s1, -s1, s2, -s2, s3} where s3 != -s3 (i.e., 2*s3 != 0 mod n)
    # OR {s1, -s1, s2, -s2, 0} but 0 gives a loop, not allowed.
    # So we need: S = {a, -a, b, -b, c} with c not equal to -c mod n, and all distinct.
    # This requires n even? No: if n is odd, -c != c always (unless c=0). If n even, -c = c when 2c=0 mod n.
    # Let's try n odd to avoid the issue. Then S = {a, n-a, b, n-b, c} with all 5 distinct and nonzero.

    # Strategy: try various n and connection sets, verify girth via BFS/non-backtracking.
    # Moore bound for (5,9): lower bound is 2 + 5*(1 + 4 + 4^2 + ... + 4^3) = 2 + 5*(1+4+16+64) = 2+5*85 = 427.
    # Actually for girth 9 (odd), the Moore bound is: 2 + k * sum_{i=0}^{(g-3)/2} (k-1)^i = 2 + 5*(1 + 4 + 16 + 64) = 427.
    # So n >= 427.

    best_n = None
    best_edges = None

    # Try circulant constructions with various n and connection sets
    # For speed, we'll try a range of n values and connection sets.
    
    def verify_girth(n, S, max_cycle_len):
        """Check that no cycle of length < max_cycle_len exists using BFS from each vertex.
        Returns True if girth >= max_cycle_len."""
        # Build adjacency
        adj = [[] for _ in range(n)]
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if j != i:
                    adj[i].append(j)
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Check no loops
        for i in range(n):
            if i in adj[i]:
                return False
        # Check no repeated edges (should be fine for circulant)
        # BFS to find shortest cycle
        # For each vertex, do BFS tracking parent to find shortest cycle
        # More efficient: for each edge, check if there's a path of length max_cycle_len-3 between endpoints
        # Actually, for girth verification, we need to check all cycles of length < max_cycle_len.
        # Use non-backtracking BFS from each vertex.
        
        for start in range(n):
            # BFS with (node, parent, depth)
            visited = {}  # (node, parent) -> depth
            queue = []
            for nb in adj[start]:
                queue.append((start, nb, 1))
                visited[(start, nb)] = 1
            while queue:
                cur, par, depth = queue.pop(0)
                if depth >= max_cycle_len - 1:
                    continue
                for nb in adj[cur]:
                    if nb == par:
                        continue
                    if nb == start:
                        # Found a cycle of length depth + 1
                        if depth + 1 < max_cycle_len:
                            return False
                    else:
                        key = (cur, nb)
                        if key not in visited:
                            visited[key] = depth + 1
                            queue.append((cur, nb, depth + 1))
        return True

    def verify_girth_fast(n, S, max_cycle_len):
        """Faster girth check using BFS from each vertex, looking for shortest cycle."""
        adj = [[] for _ in range(n)]
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if j != i:
                    adj[i].append(j)
        
        # Check degree and no loops
        for i in range(n):
            if len(adj[i]) != k:
                return False
            if i in adj[i]:
                return False
        
        # For each vertex, BFS to find shortest cycle back to it
        for start in range(n):
            # dist[v] = shortest distance from start to v (with no backtracking constraint for simple BFS)
            # But we need cycles, so we track parent
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            found = False
            for qi in range(len(queue)):
                u = queue[qi]
                if dist[u] >= max_cycle_len - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < max_cycle_len:
                            return False
            # Also check edges from start's neighbors back to start
            for v in adj[start]:
                # Already handled by BFS
                pass
        return True

    # Try various n and connection sets
    # For k=5, g=9: n must be at least 427.
    # Try n from 427 upward, and various connection sets.
    
    # Connection set: {a, -a, b, -b, c} mod n
    # Constraints: all 5 values distinct and nonzero mod n.
    # For n odd: -a != a, -b != b, -c != c automatically (if a,b,c nonzero).
    # Need: a != b, a != -b, a != c, a != -c, b != c, b != -c, c != 0, etc.
    
    max_n_to_try = 2000  # limit search
    found = False
    
    for n in range(427, min(max_n_to_try + 1, 10000)):
        if time.time() > DEADLINE - 10:
            break
        if n % 2 == 0:
            continue  # try odd n first for simplicity
        
        # Try various connection sets
        # Heuristic: choose a, b, c as roughly n/5, n/3, n/7 or similar
        candidates = []
        for a in range(1, n // 2):
            if time.time() > DEADLINE - 10:
                break
            for b in range(1, n // 2):
                if b == a or b == n - a:
                    continue
                for c in range(1, n // 2):
                    if c == a or c == n - a or c == b or c == n - b:
                        continue
                    S = [a, n - a, b, n - b, c]
                    # Check all distinct
                    if len(set(S)) != 5:
                        continue
                    # Check no zero
                    if 0 in S:
                        continue
                    candidates.append(S)
        
        # Try a subset of candidates (for speed)
        import random
        random.seed(n)  # deterministic per n
        if len(candidates) > 50:
            candidates = random.sample(candidates, 50)
        
        for S in candidates:
            if time.time() > DEADLINE - 5:
                break
            if verify_girth_fast(n, S, g):
                # Build edges
                edges = []
                for i in range(n):
                    for s in S:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                best_n = n
                best_edges = edges
                found = True
                break
        if found:
            break

    if not found:
        # Fallback: use the original LUW construction
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start_v = (0, (None,) + (0,) * d)
        index, order, edges = {start_v: 0}, [start_v], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    return best_n, best_edges
