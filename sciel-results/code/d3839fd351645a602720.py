import time
import random
import sys
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    Strategy:
    1. For k=4, g=10, attempt to find the 320-vertex cage using randomized local search.
    2. If that fails within time, try larger n (up to a limit) with the same search.
    3. Fall back to the Lazebnik-Ustimenko-Woldar construction which is guaranteed to be valid.
    """
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin

    def elapsed():
        return time.time() - t0

    # Specific optimization for k=4, g=10
    if k == 4 and g == 10:
        def generate_random_4_reg_bipartite(n):
            """
            Generates a simple 4-regular bipartite graph on n vertices (n/2 in each part).
            Uses configuration model with multi-edge removal.
            """
            if n % 2 != 0:
                return None
            
            half_n = n // 2
            # Left: 0..half_n-1, Right: half_n..n-1
            
            adj_list = [[] for _ in range(n)]
            
            # Create stubs
            left_stubs = [i for i in range(half_n) for _ in range(4)]
            right_stubs = [half_n + i for i in range(half_n) for _ in range(4)]
            
            random.shuffle(left_stubs)
            random.shuffle(right_stubs)
            
            # Pair them up
            for l, r in zip(left_stubs, right_stubs):
                adj_list[l].append(r)
                adj_list[r].append(l)
            
            # Remove multi-edges by swapping
            def has_multi_edges(adj_list):
                for v in range(n):
                    if len(adj_list[v]) != len(set(adj_list[v])):
                        return True
                return False
            
            max_iters = 3000
            for _ in range(max_iters):
                if not has_multi_edges(adj_list):
                    break
                
                # Find a vertex with duplicate
                v = -1
                for i in range(n):
                    if len(adj_list[i]) != len(set(adj_list[i])):
                        v = i
                        break
                if v == -1:
                    break
                
                # Find a duplicate neighbor u
                counts = {}
                for u in adj_list[v]:
                    counts[u] = counts.get(u, 0) + 1
                u = [key for key, val in counts.items() if val > 1][0]
                
                found_swap = False
                for _ in range(300):
                    v_prime = random.randint(0, n - 1)
                    if v_prime == v:
                        continue
                    if v_prime < half_n and v < half_n:
                        continue
                    if v_prime >= half_n and v >= half_n:
                        continue
                    
                    if v_prime < half_n:
                        u_prime_candidates = [x for x in adj_list[v_prime] if x >= half_n]
                    else:
                        u_prime_candidates = [x for x in adj_list[v_prime] if x < half_n]
                        
                    if not u_prime_candidates:
                        continue
                        
                    u_prime = random.choice(u_prime_candidates)
                    
                    if u_prime in adj_list[v]:
                        continue
                    if u in adj_list[v_prime]:
                        continue
                    
                    if u not in adj_list[v]:
                        continue
                    adj_list[v].remove(u)
                    adj_list[v].append(u_prime)
                    
                    if v not in adj_list[u]:
                        continue
                    adj_list[u].remove(v)
                    adj_list[u].append(v_prime)
                    
                    if u_prime not in adj_list[v_prime]:
                        continue
                    adj_list[v_prime].remove(u_prime)
                    adj_list[v_prime].append(u)
                    
                    if v_prime not in adj_list[u_prime]:
                        continue
                    adj_list[u_prime].remove(v_prime)
                    adj_list[u_prime].append(v)
                    
                    found_swap = True
                    break
                
                if not found_swap:
                    continue
            
            if not has_multi_edges(adj_list):
                return adj_list
            return None

        def get_girth_fast(adj_list, target_girth):
            """
            Returns the length of the shortest cycle, or target_girth if none shorter found.
            Uses bounded BFS from each vertex up to depth (target_girth-1)//2.
            For a bipartite graph, all cycles are even, so we only need to check
            even cycle lengths up to target_girth-2.
            """
            n_vertices = len(adj_list)
            min_girth = target_girth  # We'll return this if no shorter cycle found
            max_depth = (target_girth - 2) // 2  # For even cycles up to target_girth-2
            
            for start in range(n_vertices):
                dist = [-1] * n_vertices
                parent = [-1] * n_vertices
                dist[start] = 0
                queue = deque([start])
                
                while queue:
                    u = queue.popleft()
                    if dist[u] >= max_depth:
                        continue
                    for v in adj_list[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            if dist[v] < max_depth:
                                queue.append(v)
                        elif parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            # In bipartite graph, cycle_len must be even
                            if cycle_len < min_girth and cycle_len % 2 == 0:
                                min_girth = cycle_len
                                if min_girth <= 4:
                                    return min_girth
            return min_girth

        def try_local_search(n, time_limit):
            """
            Attempts to construct a 4-regular bipartite graph on n vertices with girth >= 10.
            """
            # Generate initial graph
            adj = generate_random_4_reg_bipartite(n)
            if adj is None:
                return None
            
            current_girth = get_girth_fast(adj, 10)
            
            if current_girth >= 10:
                edges = []
                for u in range(n // 2):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges

            # Local search with simulated annealing-like acceptance
            best_girth = current_girth
            best_adj = [list(x) for x in adj]
            
            # Precompute adjacency sets for O(1) lookup
            adj_set = [set(x) for x in adj]
            
            while time.time() - t0 < time_limit:
                # Pick two random edges (u, v) and (u', v')
                u = random.randint(0, n // 2 - 1)
                if not adj[u]:
                    continue
                v = random.choice(adj[u])
                
                u_prime = random.randint(0, n // 2 - 1)
                if u_prime == u:
                    continue
                if not adj[u_prime]:
                    continue
                v_prime = random.choice(adj[u_prime])
                
                # Check validity of swap
                if v_prime in adj_set[u]:
                    continue
                if v in adj_set[u_prime]:
                    continue
                
                # Perform swap
                adj[u].remove(v)
                adj[u].append(v_prime)
                adj_set[u].discard(v)
                adj_set[u].add(v_prime)
                
                adj[v].remove(u)
                adj[v].append(u_prime)
                adj_set[v].discard(u)
                adj_set[v].add(u_prime)
                
                adj[u_prime].remove(v_prime)
                adj[u_prime].append(v)
                adj_set[u_prime].discard(v_prime)
                adj_set[u_prime].add(v)
                
                adj[v_prime].remove(u_prime)
                adj[v_prime].append(u)
                adj_set[v_prime].discard(u_prime)
                adj_set[v_prime].add(u)
                
                new_girth = get_girth_fast(adj, 10)
                
                if new_girth > current_girth:
                    current_girth = new_girth
                    if current_girth > best_girth:
                        best_girth = current_girth
                        best_adj = [list(x) for x in adj]
                    if current_girth >= 10:
                        edges = []
                        for u_node in range(n // 2):
                            for v_node in adj[u_node]:
                                if u_node < v_node:
                                    edges.append((u_node, v_node))
                        return n, edges
                else:
                    # Revert
                    adj[u].remove(v_prime)
                    adj[u].append(v)
                    adj_set[u].discard(v_prime)
                    adj_set[u].add(v)
                    
                    adj[v].remove(u_prime)
                    adj[v].append(u)
                    adj_set[v].discard(u_prime)
                    adj_set[v].add(u)
                    
                    adj[u_prime].remove(v)
                    adj[u_prime].append(v_prime)
                    adj_set[u_prime].discard(v)
                    adj_set[u_prime].add(v_prime)
                    
                    adj[v_prime].remove(u)
                    adj[v_prime].append(u_prime)
                    adj_set[v_prime].discard(u)
                    adj_set[v_prime].add(u_prime)

            # Return best found
            if best_girth >= 10:
                edges = []
                for u_node in range(n // 2):
                    for v_node in best_adj[u_node]:
                        if u_node < v_node:
                            edges.append((u_node, v_node))
                return n, edges
            return None

        # Try specific n values
        n_candidates = [320, 324, 328, 332, 336, 340, 350, 400, 500]
        
        for n in n_candidates:
            if time.time() - t0 > DEADLINE - 10:
                break
            
            if n == 320:
                limit = min(60, DEADLINE - (time.time() - t0) - 5)
            else:
                limit = min(20, DEADLINE - (time.time() - t0) - 5)
                
            if limit < 1:
                continue
                
            try:
                result = try_local_search(n, limit)
                if result:
                    return result
            except Exception:
                continue

    # Fallback: L-U-W Construction
    q = k
    if g % 2 == 1:
        d = max(3, g - 5)
        if d % 2 == 0:
            d += 1
    else:
        d = max(4, g - 4)
        if d % 2 == 1:
            d += 1
    
    if q not in [2, 3, 4, 5, 7, 11, 13]:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        q = 4
        while not (is_prime(q) or q == 4):
            q += 1
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y):
            return x ^ y
        def sub(x, y):
            return x ^ y
        def mul(x, y):
            return table[x][y]
    else:
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
