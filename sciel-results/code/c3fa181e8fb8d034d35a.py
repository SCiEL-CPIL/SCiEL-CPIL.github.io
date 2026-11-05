import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------- arithmetic helpers ----------
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        if is_prime(n):
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                t = n
                while t % p == 0:
                    t //= p
                return t == 1
        return False

    def is_quadratic_residue(a, q):
        a = a % q
        if a == 0:
            return True
        if not is_prime_power(q):
            return True
        p = None
        for cand in range(2, int(math.isqrt(q)) + 1):
            if q % cand == 0:
                t = q
                while t % cand == 0:
                    t //= cand
                if t == 1:
                    p = cand
                    break
        if p is None:
            p = q
        a_p = a % p
        if a_p == 0:
            return True
        return pow(a_p, (p - 1) // 2, p) == 1

    def modinv(a, m):
        if m <= 1:
            return None
        a = a % m
        if a == 0:
            return None
        try:
            return pow(a, -1, m)
        except (ValueError, TypeError):
            return None

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    # ---------- fast girth check ----------
    def check_girth_fast(adj, n, g):
        for u in range(n):
            visited = set()
            queue = deque()
            for v in adj[u]:
                queue.append((v, u))
                visited.add((v, u))
            depth = 1
            while queue and depth < g:
                new_queue = deque()
                for (cur, prev) in queue:
                    for nxt in adj[cur]:
                        if nxt == prev:
                            continue
                        if nxt == u:
                            return False
                        state = (nxt, cur)
                        if state not in visited:
                            visited.add(state)
                            new_queue.append(state)
                queue = new_queue
                depth += 1
        return True

    def is_connected(adj, n):
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return count == n

    def validate(adj, n, k, g):
        for i in range(n):
            if len(adj[i]) != k:
                return False
        if not is_connected(adj, n):
            return False
        if not check_girth_fast(adj, n, g):
            return False
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- Lazebnik-Ustimenko-Woldar baseline ----------
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

    # ---------- Local Search for Random Regular Graphs ----------
    def find_shortest_cycle(adj, n, max_len):
        """Returns the cycle as a list of vertices if found, else None."""
        for u in range(n):
            # BFS from u
            parent = [-1] * n
            dist = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                cur = queue.popleft()
                if dist[cur] >= max_len:
                    continue
                for nxt in adj[cur]:
                    if dist[nxt] == -1:
                        dist[nxt] = dist[cur] + 1
                        parent[nxt] = cur
                        queue.append(nxt)
                    elif nxt != parent[cur] and dist[nxt] == dist[cur] + 1:
                        # Found a cycle
                        # Reconstruct path
                        cycle = []
                        node = cur
                        while node != nxt:
                            cycle.append(node)
                            node = parent[node]
                        cycle.append(nxt)
                        cycle.reverse()
                        return cycle
        return None

    def local_search_regular(n, k, g, budget):
        """
        Attempts to find a k-regular graph of girth g on n vertices
        using local search (2-switches) to break short cycles.
        """
        if n * k % 2 != 0:
            return None
        
        # Generate initial simple k-regular graph
        adj = None
        attempts = 0
        max_attempts = 50
        while adj is None and attempts < max_attempts:
            attempts += 1
            # Configuration model
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            temp_adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in temp_adj[u]:
                    valid = False
                    break
                temp_adj[u].add(v)
                temp_adj[v].add(u)
            if valid:
                # Check connected
                if is_connected([list(s) for s in temp_adj], n):
                    adj = [list(s) for s in temp_adj]
        
        if adj is None:
            return None

        # Local search
        max_iters = 1000
        for _ in range(max_iters):
            if time.time() > budget:
                break
            
            # Check girth
            if check_girth_fast(adj, n, g):
                # Success
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return (n, edges)
            
            # Find shortest cycle
            cycle = find_shortest_cycle(adj, n, g)
            if cycle is None:
                # Should not happen if girth < g
                break
            
            # Try to break the cycle using 2-switches
            # Pick a random edge in the cycle
            # and a random edge outside the cycle
            # and swap
            found_swap = False
            for _ in range(100): # Try 100 random swaps
                if time.time() > budget:
                    break
                # Pick an edge (u, v) from the cycle
                idx = random.randrange(len(cycle))
                u = cycle[idx]
                v = cycle[(idx + 1) % len(cycle)]
                
                # Pick an edge (x, y) not in the cycle
                # To be efficient, pick a random vertex x not in cycle, and a neighbor y not in cycle
                # and ensure (x,y) is an edge
                # This is hard to do directly.
                # Instead, pick a random edge from the whole graph and check if it's in the cycle.
                # We can just iterate over a few random edges.
                
                # Let's pick a random vertex x
                x = random.randrange(n)
                if x in cycle:
                    continue
                # Pick a random neighbor y of x
                if not adj[x]:
                    continue
                y = random.choice(adj[x])
                if y in cycle:
                    continue
                if x == y:
                    continue
                if y not in adj[x]: # Should be true
                    continue
                
                # Check if (x,y) is already an edge (it is)
                # Check if (u,x) and (v,y) are valid new edges
                # 1. (u,x) not an edge
                # 2. (v,y) not an edge
                # 3. (u,y) not an edge (to avoid creating triangle if u,v adjacent? No, we are removing u-v)
                # 4. (v,x) not an edge
                # 5. u != x, u != y, v != x, v != y
                
                if u == x or u == y or v == x or v == y:
                    continue
                if x in adj[u] or y in adj[u]:
                    continue
                if x in adj[v] or y in adj[v]:
                    continue
                # Also ensure we don't create a multi-edge
                # adj is a list of lists, so we need to check carefully
                
                # Perform swap
                # Remove (u,v)
                adj[u].remove(v)
                adj[v].remove(u)
                # Remove (x,y)
                adj[x].remove(y)
                adj[y].remove(x)
                # Add (u,x)
                adj[u].append(x)
                adj[x].append(u)
                # Add (v,y)
                adj[v].append(y)
                adj[y].append(v)
                
                found_swap = True
                break
            
            if not found_swap:
                # Try another cycle or restart
                continue

        return None

    # ---------- main ----------
    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    # Try to find a smaller graph using local search
    # Start from n_min up to best_n - 1
    # Limit the range to avoid wasting time
    search_limit = min(best_n[0] - 1, n_min + 100)
    
    for curr_n in range(n_min, search_limit + 1):
        if time.time() > deadline - 5.0:
            break
        # Budget for this n
        budget = min(deadline - 5.0 - time.time(), 1.0)
        if budget <= 0:
            break
        
        res = local_search_regular(curr_n, k, g, budget)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
            # If we found a smaller one, we can stop or continue
            # Given the difficulty, stopping is safer
            break

    return best_result

# EVOLVE-BLOCK-END
