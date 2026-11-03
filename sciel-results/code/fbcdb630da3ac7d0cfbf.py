import time
import random
import sys

def construct(k, g):
    # Target: 4-regular, girth >= 10.
    # Known cage size: 320.
    # Seed score 1.0 implies seed returns n=2048 (LUW).
    # We aim for n=320.
    
    deadline = time.time() + 115
    
    # If k != 4 or g != 10, use a generic fallback (LUW or random)
    if k != 4 or g != 10:
        # Fallback for other cases
        # Use LUW construction as in seed
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

    # Specific case k=4, g=10
    half = 160 # Total n=320
    
    def make_random_4reg_bipartite(half, seed):
        random.seed(seed)
        left_stubs = [(i, s) for i in range(half) for s in range(4)]
        right_stubs = [(i, s) for i in range(half) for s in range(4)]
        random.shuffle(right_stubs)
        
        edges = []
        edge_set = set()
        adj = [set() for _ in range(2 * half)]
        for (li, ls), (ri, rs) in zip(left_stubs, right_stubs):
            u = li
            v = half + ri
            if u == v: # Should not happen in bipartite
                return None, None
            if v in adj[u]:
                return None, None
            edge_set.add((u, v))
            edges.append((u, v))
            adj[u].add(v)
            adj[v].add(u)
        return edges, adj

    def count_short_cycles(adj, n, max_len):
        # Counts cycles of length < max_len
        # For bipartite, only even cycles.
        count = 0
        # 4-cycles
        # For each pair of left nodes u1, u2, count common neighbors.
        # A 4-cycle is (u1, v1, u2, v2).
        # Number of 4-cycles = sum over pairs of left nodes (common_neighbors choose 2)
        # This is O(N^2 * K)
        left_nodes = range(half)
        right_nodes = range(half, 2*half)
        
        # Precompute neighbor sets for speed
        # adj is list of sets
        
        # 4-cycles
        for i in range(half):
            for j in range(i+1, half):
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    count += len(common) * (len(common) - 1) // 2
        
        # 6-cycles
        # A 6-cycle is (u1, v1, u2, v2, u3, v3).
        # This is harder to count directly.
        # We can use BFS from each left node to depth 3.
        # If we reach a left node u3 at depth 2, and u3 is connected to v3 which is connected to start?
        # Actually, just check girth.
        
        return count

    def check_girth(adj, n, min_girth):
        # Returns True if girth >= min_girth
        # BFS from each vertex to depth (min_girth-2)//2
        max_depth = (min_girth - 2) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            
            for d in range(1, max_depth + 1):
                new_queue = []
                for u in queue:
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = d
                            parent[v] = u
                            new_queue.append(v)
                        else:
                            # Check for cycle
                            # In bipartite graph, if we see a visited node v:
                            # If dist[v] == d-1, it's parent (ok) or cross edge (cycle 2d)
                            # If dist[v] < d-1, cycle d + dist[v] + 1
                            if dist[v] < d - 1:
                                cycle_len = d + dist[v] + 1
                                if cycle_len < min_girth:
                                    return False
                            elif dist[v] == d - 1 and parent[u] != v:
                                cycle_len = 2 * d
                                if cycle_len < min_girth:
                                    return False
                queue = new_queue
        return True

    # Simulated Annealing
    best_edges = None
    best_n = 2048 # Fallback
    
    for n_target in [320, 384, 480, 640, 1024]:
        half = n_target // 2
        if time.time() > deadline:
            break
            
        # Try to find graph with this n
        found = False
        for restart in range(50):
            if time.time() > deadline:
                break
                
            edges, adj = make_random_4reg_bipartite(half, restart)
            if not edges:
                continue
                
            if check_girth(adj, n_target, g):
                return n_target, edges
                
            # Local Search
            current_penalty = 1000 # Initialize high
            # We don't have a fast penalty function, so we just do random swaps and check girth
            # This is slow.
            
            # Better: Use a fast proxy.
            # Count 4-cycles.
            def count_4cycles(adj):
                c = 0
                for i in range(half):
                    for j in range(i+1, half):
                        common = adj[i] & adj[j]
                        if len(common) >= 2:
                            c += len(common) * (len(common) - 1) // 2
                return c
            
            cur_4 = count_4cycles(adj)
            
            for step in range(1000):
                if time.time() > deadline:
                    break
                    
                # Pick random edges to swap
                # Pick u1, v1 and u2, v2
                # Ensure u1, u2 are left, v1, v2 are right
                # u1 != u2, v1 != v2
                # (u1, v2) and (u2, v1) not in edges
                
                # To speed up, pick u1, u2 randomly, then pick v1 from adj[u1], v2 from adj[u2]
                # Check if swap is valid.
                
                valid_swap = False
                attempts = 0
                while not valid_swap and attempts < 10:
                    u1 = random.randint(0, half - 1)
                    u2 = random.randint(0, half - 1)
                    if u1 == u2:
                        attempts += 1
                        continue
                    v1 = random.choice(list(adj[u1]))
                    v2 = random.choice(list(adj[u2]))
                    if v1 == v2:
                        attempts += 1
                        continue
                    # Check if (u1, v2) and (u2, v1) are not edges
                    if v2 not in adj[u1] and v1 not in adj[u2]:
                        valid_swap = True
                    else:
                        attempts += 1
                
                if not valid_swap:
                    continue
                
                # Perform swap
                adj[u1].remove(v1)
                adj[u2].remove(v2)
                adj[u1].add(v2)
                adj[u2].add(v1)
                adj[v1].remove(u1)
                adj[v2].remove(u1)
                adj[v1].add(u2)
                adj[v2].add(u2)
                
                new_4 = count_4cycles(adj)
                
                # Accept if new_4 < cur_4 or random (Annealing)
                # For now, Greedy
                if new_4 < cur_4:
                    cur_4 = new_4
                    if check_girth(adj, n_target, g):
                        # Extract edges
                        final_edges = []
                        for i in range(half):
                            for v in adj[i]:
                                if i < v: # Only left-right edges
                                    final_edges.append((i, v))
                        return n_target, final_edges
                else:
                    # Revert
                    adj[u1].remove(v2)
                    adj[u2].remove(v1)
                    adj[u1].add(v1)
                    adj[u2].add(v2)
                    adj[v1].remove(u2)
                    adj[v2].remove(u2)
                    adj[v1].add(u1)
                    adj[v2].add(u1)
                    
        # If not found for this n, continue to next n

    # Fallback: LUW
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
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
