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

    def check_girth_nonbacktracking(adj, n, target_g):
        """
        Check girth >= target_g using non-backtracking walks.
        For each vertex, do BFS on the line graph of non-backtracking walks.
        If we can reach a vertex at distance d with the same 'parent' constraint,
        we have a cycle of length < target_g.
        
        Actually, simpler: for each directed edge (u,v), do BFS on non-backtracking 
        walks starting from (u,v). If we reach (w,u) at depth < target_g-1, 
        we have a cycle of length < target_g.
        """
        # For each directed edge, check if there's a non-backtracking walk
        # from (u,v) back to (v,u) with length < target_g - 1
        # That would form a cycle of length < target_g
        
        # Build adjacency sets for quick lookup
        adj_set = [set(adj[i]) for i in range(n)]
        
        for u in range(n):
            for v in adj[u]:
                # BFS on non-backtracking walks starting from (u,v)
                # State: (current_vertex, previous_vertex)
                # We want to reach (v, u) with path length < target_g - 1
                # (since we already used edge u->v, total cycle length = path_len + 1)
                
                visited = set()
                visited.add((u, v))
                queue = [(u, v, 0)]  # (current, prev, depth)
                
                while queue:
                    cur, prev, depth = queue.pop(0)
                    if depth >= target_g - 2:
                        continue
                    for nxt in adj_set[cur]:
                        if nxt == prev:
                            continue
                        new_state = (cur, nxt)
                        if new_state == (v, u):
                            # Found a cycle of length depth + 2
                            if depth + 2 < target_g:
                                return False
                        if new_state not in visited:
                            visited.add(new_state)
                            queue.append((nxt, cur, depth + 1))
        return True

    def gen_random_5reg(N):
        # Returns adjacency list or None
        for attempt in range(100):
            # Configuration model with rejection
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
                # Verify degrees
                for i in range(N):
                    if len(adj[i]) != 5:
                        valid = False
                        break
                if valid:
                    return adj
        return None

    def find_short_cycle(adj, n, target_g):
        """Find a cycle of length < target_g, return list of vertices or None."""
        adj_set = [set(adj[i]) for i in range(n)]
        
        for start in range(n):
            # BFS with parent tracking
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            
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
                        cycle_len = dist[u] + dist[w] + 1
                        if cycle_len < target_g:
                            # Reconstruct cycle
                            cycle = []
                            # Path from u to start
                            cur = u
                            while cur != -1:
                                cycle.append(cur)
                                cur = parent[cur]
                            cycle.reverse()
                            # Path from w to start
                            cur = w
                            while cur != -1:
                                if cur not in cycle:
                                    cycle.append(cur)
                                cur = parent[cur]
                            # Actually, let's just return the two paths
                            path1 = []
                            cur = u
                            while cur != start:
                                path1.append(cur)
                                cur = parent[cur]
                            path1.append(start)
                            path1.reverse()
                            
                            path2 = []
                            cur = w
                            while cur != start:
                                path2.append(cur)
                                cur = parent[cur]
                            path2.append(start)
                            
                            return path1 + path2[1:]
        return None

    def local_search_5reg(adj, N, target_g, max_iters=5000):
        """Try to increase girth by eliminating short cycles."""
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Check if girth is already sufficient
            if check_girth_nonbacktracking(adj, N, target_g):
                return adj
            
            # Find a short cycle
            cycle = find_short_cycle(adj, N, target_g)
            if cycle is None:
                # Shouldn't happen if check says girth < target
                continue
            
            # Try to break the cycle by swapping an edge
            # Pick a random edge in the cycle and try to swap it
            cycle_len = len(cycle)
            if cycle_len < 3:
                continue
            
            # Try several random swaps
            found_swap = False
            for _ in range(20):
                # Pick a random edge in the cycle
                idx = random.randint(0, cycle_len - 1)
                u = cycle[idx]
                v = cycle[(idx + 1) % cycle_len]
                
                if v not in adj[u]:
                    continue
                    
                # Find another edge (w, x) to swap with
                # w, x not in {u, v}, w != x
                # u not adjacent to w, v not adjacent to x
                # w not adjacent to v (avoid creating multi-edge? No, we remove u-v)
                # Actually after swap: remove (u,v) and (w,x), add (u,w) and (v,x)
                # Need: w not in adj[u], x not in adj[v], w != u, x != v, w != v, x != u, w != x
                
                attempts = 0
                while attempts < 50:
                    attempts += 1
                    w = random.randint(0, N-1)
                    x = random.randint(0, N-1)
                    if w == u or w == v or x == u or x == v or w == x:
                        continue
                    if w in adj[u] or x in adj[v]:
                        continue
                    if x not in adj[w]:
                        continue
                    # Perform swap
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[w].remove(x)
                    adj[x].remove(w)
                    adj[u].append(w)
                    adj[w].append(u)
                    adj[v].append(x)
                    adj[x].append(v)
                    found_swap = True
                    break
            
            if not found_swap:
                continue
        
        return None

    def gen_random_4reg(N):
        """Generate a random 4-regular graph on N vertices."""
        for attempt in range(50):
            # Use two perfect matchings (derangements)
            # Or use configuration model
            stubs = [i // 4 for i in range(N * 4)]
            random.shuffle(stubs)
            adj = [[] for _ in range(N)]
            valid = True
            for i in range(0, N * 4, 2):
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
                for i in range(N):
                    if len(adj[i]) != 4:
                        valid = False
                        break
                if valid:
                    return adj
        return None

    # Strategy
    best_n = float('inf')
    best_edges = None
    
    # Try random search for n in range starting from Moore bound
    # For k=5, g=9: Moore bound = 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426
    # Try a few values near the Moore bound
    for n in range(moore, moore + 100, 5):
        if time.time() > deadline - 30:
            break
        adj = gen_random_5reg(n)
        if adj is None:
            continue
        # Try local search to increase girth
        adj = local_search_5reg(adj, n, g, max_iters=200)
        if adj is not None and check_girth_nonbacktracking(adj, n, g):
            # Convert adj to edges
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            if n < best_n:
                best_n = n
                best_edges = edges
            break
    
    if best_edges is not None:
        return best_n, best_edges

    # Fallback: C_9 x H where H is 4-regular with girth >= 9
    best_H = None
    best_M = float('inf')
    for M in range(161, 250):
        if time.time() > deadline - 15:
            break
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            adj = gen_random_4reg(M)
            if adj is None:
                continue
            if check_girth_nonbacktracking(adj, M, 9):
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

    # Final fallback: LUW construction
    return luw_construct(k, g)
