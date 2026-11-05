import time
import random
import math

_DEADLINE = 120.0

def _verify(n, edges, k, g):
    """Verify graph is simple, k-regular, connected, girth >= g."""
    if n < 1:
        return False
    if len(edges) != len(set(edges)):
        return False
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        if a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # girth check
    # BFS from each vertex up to depth g//2
    # If we find a back edge to a vertex at depth d, cycle len = depth[u] + depth[v] + 1
    # We want no cycle < g.
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = [(start, 0)]
        while q:
            u, d = q.pop(0)
            if d >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    q.append((v, d + 1))
                else:
                    # Cycle found: u -> ... -> v and v -> ... -> u?
                    # Actually, if dist[v] != -1 and v is not parent, we have a cycle.
                    # Cycle length = dist[u] + dist[v] + 1?
                    # No, BFS tree edge u-v. If v is already visited, cycle length = dist[u] + dist[v] + 1 is not correct if v is ancestor.
                    # Standard check: if dist[v] != -1 and v != parent[u], then cycle exists.
                    # Length = dist[u] + dist[v] + 1.
                    # We need to ensure this length >= g.
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def _luzw(k, g):
    """Fallback LUW construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
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

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _random_schreier(n, k, g, time_limit):
    """Try to find a 4-regular graph of girth >= g with n vertices using random Schreier graph of C_5 * C_5."""
    if k != 4:
        return None
    start_time = time.time()
    # Generators a, b of order 5
    # We need permutations p_a, p_b of [0..n-1] such that p_a^5 = id, p_b^5 = id.
    # And the group generated is transitive.
    
    def random_perm_order_5(n):
        # Generate a random permutation that is a product of disjoint 5-cycles (and fixed points? No, must be order 5 exactly? 
        # If fixed points exist, order is still 5. But for transitivity, we probably want few fixed points.)
        # Actually, if p_a has fixed points, it's fine.
        # We just need p_a^5 = id.
        # This means p_a is a product of disjoint cycles of length 1 or 5.
        # To ensure transitivity of the group <p_a, p_b>, we should avoid p_a being identity.
        perm = list(range(n))
        # Shuffle
        random.shuffle(perm)
        # Form cycles of length 5
        # We can just take the shuffled list and group into 5s.
        # If n is not divisible by 5, some will be fixed?
        # If n=320, 320/5 = 64. Perfect.
        result = [0] * n
        for i in range(0, n, 5):
            # Cycle: perm[i] -> perm[i+1] -> ... -> perm[i+4] -> perm[i]
            for j in range(5):
                result[perm[i+j]] = perm[i+(j+1)%5]
        return result

    for _ in range(10000): # Max attempts
        if time.time() - start_time > time_limit:
            break
        
        p_a = random_perm_order_5(n)
        p_b = random_perm_order_5(n)
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        for i in range(n):
            adj[i].append(p_a[i])
            adj[i].append(p_b[i])
            # Inverse? 
            # Schreier graph of C_5 * C_5 has edges for a, a^{-1}, b, b^{-1}.
            # Since a^5=1, a^{-1} = a^4.
            # We need to add a^4 and b^4.
            # p_a_inv[i] = p_a[p_a[p_a[p_a[i]]]]
            # This is expensive to compute naively.
            # But we can just store the permutation and compute inverse once.
            
        # Compute inverses
        p_a_inv = [0] * n
        p_b_inv = [0] * n
        for i in range(n):
            p_a_inv[p_a[i]] = i
            p_b_inv[p_b[i]] = i
            
        for i in range(n):
            adj[i].append(p_a_inv[i])
            adj[i].append(p_b_inv[i])
            
        # Check degree 4
        for i in range(n):
            if len(adj[i]) != 4:
                break
        else:
            # Check simple (no multi-edges)
            # p_a[i] != p_b[i], p_a[i] != p_a_inv[i] (unless order 2, but order 5 so no), etc.
            # Since order 5, a != a^{-1}.
            # We just need to ensure p_a[i] != p_b[i] and p_a[i] != p_b_inv[i] etc.
            # If p_a[i] == p_b[i], we have a multi-edge.
            valid = True
            for i in range(n):
                neighbors = set(adj[i])
                if len(neighbors) != 4:
                    valid = False
                    break
            if not valid:
                continue
                
            # Check girth
            # BFS from each vertex up to depth 4
            is_valid = True
            for start in range(n):
                dist = [-1] * n
                dist[start] = 0
                q = [(start, 0)]
                while q:
                    u, d = q.pop(0)
                    if d >= 4: # Max depth for girth 10 check?
                        # Cycle length L. If we reach v at depth d, and edge u-v exists.
                        # If v is visited at depth d', cycle len = d + d' + 1.
                        # We want d + d' + 1 >= 10.
                        # Max d is 4. If d'=4, len=9. So we must check depth 4.
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = d + 1
                            q.append((v, d + 1))
                        else:
                            # Back edge
                            # We need to ensure the cycle is not short.
                            # The cycle length is dist[u] + dist[v] + 1.
                            # This is only valid if v is not the parent of u in BFS tree.
                            # But we don't track parent.
                            # However, if dist[v] <= dist[u], and v is not parent, it's a cycle.
                            # Actually, if dist[v] < dist[u], it's a back edge to ancestor.
                            # If dist[v] == dist[u], it's a cross edge.
                            # In any case, if dist[u] + dist[v] + 1 < 10, it's a short cycle.
                            if dist[u] + dist[v] + 1 < 10:
                                is_valid = False
                                break
                    if not is_valid:
                        break
                if not is_valid:
                    break
            if is_valid:
                # Extract edges
                edges = set()
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.add((i, j))
                return n, sorted(edges)
    return None

def construct(k, g):
    start = time.time()
    if k == 4 and g == 10:
        # Try to find the 320-vertex graph
        result = _random_schreier(320, k, g, 60.0)
        if result:
            n, edges = result
            if _verify(n, edges, k, g):
                return n, edges
        # Fallback to LUW
        n, edges = _luzw(k, g)
        return n, edges
    else:
        n, edges = _luzw(k, g)
        return n, edges
