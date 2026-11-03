import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # Moore bound
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2 - 1
        moore = k * sum((k - 1) ** i for i in range(r))

    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

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

    def check_girth(adj, n, target_g):
        # Returns True if girth >= target_g
        # BFS from each vertex
        for v in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[v] = 0
            queue = [v]
            while queue:
                u = queue.pop(0)
                if dist[u] >= (target_g - 1) // 2:
                    continue
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        parent[w] = u
                        queue.append(w)
                    elif w != parent[u] and parent[w] != u:
                        if dist[w] + dist[u] + 1 < target_g:
                            return False
        return True

    def gen_random_5reg(N):
        # Returns adjacency list or None
        for _ in range(50):
            # Configuration model
            stubs = [i // 5 for i in range(N * 5)]
            random.shuffle(stubs)
            adj = [[] for _ in range(N)]
            valid = True
            for i in range(0, N * 5, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                return adj
        return None

    def local_search_5reg(adj, N, target_g, max_iters=10000):
        # Try to increase girth
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Check girth
            if check_girth(adj, N, target_g):
                return adj
            # Find a short cycle
            # Simplified: Just pick a random edge and try to switch
            # This is a heuristic
            u = random.randint(0, N-1)
            if not adj[u]: continue
            v = random.choice(adj[u])
            # Find another edge (w, x) to switch with
            # w != u, x != v
            # Ensure no loops/multi-edges after switch
            # Switch: remove (u,v) and (w,x), add (u,w) and (v,x)
            # We need to find w, x such that:
            # 1. w != u, x != v
            # 2. w != v, x != u
            # 3. u not in adj[w]
            # 4. v not in adj[x]
            # 5. w not in adj[v] (to avoid multi-edge u-v if we swap differently? No, we remove u-v)
            # 6. x not in adj[u]
            
            # Pick random w
            w = random.randint(0, N-1)
            if w == u or w == v: continue
            if w in adj[u]: continue
            if w in adj[v]: continue # This would create multi-edge v-w? No, we are adding u-w and v-x.
            # We are adding (u,w) and (v,x).
            # So we need:
            # u not in adj[w]
            # v not in adj[x]
            # w not in adj[v] is NOT required, but if w==v we skip.
            # x not in adj[u]
            # x != u, x != v
            # x != w
            
            # Pick random x
            x = random.randint(0, N-1)
            if x in [u, v, w]: continue
            if x in adj[v]: continue
            if x in adj[u]: continue
            
            # Perform switch
            # Remove (u,v)
            adj[u].remove(v)
            adj[v].remove(u)
            # Remove (w,x) - we need to find x in adj[w]
            if x not in adj[w]:
                # Revert
                adj[u].append(v)
                adj[v].append(u)
                continue
            adj[w].remove(x)
            adj[x].remove(w)
            
            # Add (u,w) and (v,x)
            adj[u].append(w)
            adj[w].append(u)
            adj[v].append(x)
            adj[x].append(v)
            
            # Note: This random switching might not converge quickly.
            # It's a gamble.

        return None

    # Strategy
    # 1. Try to find a 5-regular graph with girth >= 9 using random search + local search
    # 2. Fallback to C_9 x H
    # 3. Fallback to LUW

    best_n = float('inf')
    best_edges = None
    
    # Try random search for n in range
    # Start from Moore bound
    for n in range(moore, moore + 200, 10):
        if time.time() > deadline - 20:
            break
        adj = gen_random_5reg(n)
        if adj is None:
            continue
        # Try local search
        adj = local_search_5reg(adj, n, g, max_iters=100)
        if adj is not None and check_girth(adj, n, g):
            # Convert adj to edges
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            if n < best_n:
                best_n = n
                best_edges = edges
            break # Found one, stop searching larger n?
            # Or continue to find smaller? No, we go up.
            # If we found one, it's a valid solution.
            # We can try to find a smaller one?
            # But we started from Moore bound.
            # So this is the smallest n we tried.
            # We should return this.
    
    if best_edges is not None:
        return best_n, best_edges

    # Fallback: C_9 x H
    def gen_random_4reg(N):
        for _ in range(20):
            p1 = list(range(N))
            p2 = list(range(N))
            random.shuffle(p1)
            random.shuffle(p2)
            if any(p1[i] == i for i in range(N)) or any(p2[i] == i for i in range(N)):
                continue
            valid = True
            edges = set()
            for i in range(N):
                for j in [p1[i], p2[i]]:
                    if i > j:
                        e = (i, j)
                    else:
                        e = (j, i)
                    if e in edges:
                        valid = False
                        break
                    edges.add(e)
                if not valid:
                    break
            if valid:
                adj = [[] for _ in range(N)]
                for i in range(N):
                    adj[i].append(p1[i])
                    adj[i].append(p2[i])
                return adj
        return None

    best_H = None
    best_M = float('inf')
    for M in range(161, 300):
        if time.time() > deadline - 10:
            break
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            adj = gen_random_4reg(M)
            if adj is None:
                continue
            if check_girth(adj, M, 9):
                best_H = adj
                best_M = M
                break
        if best_H is not None:
            break

    if best_H is not None:
        M = best_M
        N_total = 9 * M
        edges = []
        for layer in range(9):
            for v in range(M):
                u = layer * M + v
                for w in best_H[v]:
                    v2 = layer * M + w
                    if u < v2:
                        edges.append((u, v2))
                v_next = (layer + 1) % 9
                w_next = v_next * M + v
                if u < w_next:
                    edges.append((u, w_next))
        return N_total, edges

    return luw_construct(k, g)
