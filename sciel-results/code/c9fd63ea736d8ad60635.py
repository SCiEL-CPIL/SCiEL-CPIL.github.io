import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for final check/return
    
    # Fallback: LUW construction from seed
    def luw_construction():
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

    # Girth Checker using BFS from each vertex
    def get_girth(adj):
        n = len(adj)
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif dist[v] != dist[u] + 1:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
            if min_girth < g:
                return min_girth
        return min_girth if min_girth != float('inf') else 2 * n

    # Random K-Regular Graph Generator using configuration model with repair
    def gen_random_k_reg(n, k):
        # Use a better approach: generate random pairings and repair
        for attempt in range(200):
            # Create stubs
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            
            adj = [[] for _ in range(n)]
            valid = True
            
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if valid and all(len(a) == k for a in adj):
                return adj
        return None

    # Count short cycles (triangles and 4-cycles) for penalty
    def count_short_cycles(adj, n):
        count = 0
        # Count triangles
        for u in range(n):
            for i in range(len(adj[u])):
                for j in range(i+1, len(adj[u])):
                    if adj[u][j] in adj[adj[u][i]]:
                        count += 1
        # Count 4-cycles (approximate)
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = set(adj[u]) & set(adj[v])
                    count += len(common)
        return count

    # Optimize using 2-switches
    def optimize(n, k, g, time_limit):
        adj = gen_random_k_reg(n, k)
        if not adj:
            return None
            
        current_penalty = count_short_cycles(adj, n)
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Pick two random edges (u,v) and (w,x)
            u = random.randint(0, n-1)
            if not adj[u]:
                continue
            v = random.choice(adj[u])
            
            w = random.randint(0, n-1)
            if w in (u, v) or not adj[w]:
                continue
            x = random.choice(adj[w])
            if x in (u, v) or w == x:
                continue
            
            # Check if (u,w) and (v,x) are valid new edges
            if w in adj[u] or x in adj[v]:
                continue
            
            # Perform 2-switch
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            
            adj[u].append(w)
            adj[w].append(u)
            adj[v].append(x)
            adj[x].append(v)
            
            new_penalty = count_short_cycles(adj, n)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if current_penalty == 0:
                    girth = get_girth(adj)
                    if girth >= g:
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        return (n, edges)
            else:
                # Reject
                adj[u].remove(w)
                adj[w].remove(u)
                adj[v].remove(x)
                adj[x].remove(v)
                
                adj[u].append(v)
                adj[v].append(u)
                adj[w].append(x)
                adj[x].append(w)
        
        return None

    # Moore bound for lower bound on n
    def moore_bound(k, g):
        if g % 2 == 0:
            d = g // 2
            return k + k * (k-1) * sum((k-1)**i for i in range(d-1))
        else:
            d = (g-1) // 2
            return k + k * sum((k-1)**i for i in range(d))

    lb = moore_bound(k, g)
    
    # Try to find a graph with n close to the lower bound
    # Start from a reasonable estimate and try to optimize
    candidates = []
    # Moore bound is a lower bound, try values above it
    for mult in [1.0, 1.2, 1.5, 2.0, 3.0]:
        n = int(lb * mult)
        if n % 2 == 0 and n * k % 2 == 0:  # n*k must be even for regular graph
            candidates.append(n)
    
    # Also try some fixed reasonable values
    candidates.extend([500, 700, 900, 1100, 1300, 1500])
    candidates = sorted(set(candidates))
    
    best_result = None
    
    for n in candidates:
        if time.time() > deadline - 15:
            break
        
        time_budget = min(20, deadline - time.time() - 10)
        if time_budget < 5:
            break
        
        result = optimize(n, k, g, time_budget)
        if result:
            if best_result is None or result[0] < best_result[0]:
                best_result = result
                if best_result[0] <= lb + 10:  # Close to optimal
                    break
    
    if best_result:
        return best_result
    
    # Fallback to LUW construction
    return luw_construction()
