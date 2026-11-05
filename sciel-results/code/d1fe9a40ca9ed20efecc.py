import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Fallback LUW Construction
    def get_luw_graph(d, rule):
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

    def generate_rule_variants(d):
        variants = []
        rule1 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule1[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        variants.append(rule1)
        return variants

    def check_girth_fast(n, edges, min_girth):
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for u in range(n):
            if len(adj[u]) != k:
                return False
        
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        max_depth = min_girth // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # Generate a valid LUW graph as fallback
    best_luw_n = float('inf')
    best_luw_edges = []
    min_d = max(2, (g - 5) // 2)
    for d in range(min_d, min_d + 6):
        if time.time() > deadline - 60:
            break
        for rule in generate_rule_variants(d):
            if time.time() > deadline - 55:
                break
            try:
                n, edges = get_luw_graph(d, rule)
                if n < best_luw_n and n <= 50000:
                    if check_girth_fast(n, edges, g):
                        best_luw_n = n
                        best_luw_edges = edges
                        break
            except Exception:
                pass

    if best_luw_n == float('inf'):
        d = min_d + 6
        rule = generate_rule_variants(d)[0]
        best_luw_n, best_luw_edges = get_luw_graph(d, rule)

    current_best_n = best_luw_n
    current_best_edges = best_luw_edges

    # 2-Switch Optimization
    def gen_random_k_reg(n, k):
        # Configuration model
        for _ in range(50):
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

    def count_3_cycles(adj):
        n = len(adj)
        adj_sets = [set(a) for a in adj]
        count = 0
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    count += len(adj_sets[u] & adj_sets[v])
        return count // 3

    def count_4_cycles(adj):
        n = len(adj)
        adj_sets = [set(a) for a in adj]
        count = 0
        for u in range(n):
            for i in range(len(adj[u])):
                v = adj[u][i]
                for j in range(i+1, len(adj[u])):
                    w = adj[u][j]
                    # v and w share u. Do they share another neighbor?
                    # Common neighbors of v and w, excluding u
                    common = adj_sets[v] & adj_sets[w]
                    if u in common:
                        common.remove(u)
                    count += len(common)
        return count // 2 # Each 4-cycle u-v-x-w-u is counted twice? 
        # Cycle u-v-x-w. 
        # Pairs (u, (v,w)) -> x in N(v) \cap N(w) \ N(u).
        # Pairs (x, (v,w)) -> u in N(v) \cap N(w) \ N(x).
        # So yes, each 4-cycle is counted twice (once for each opposite pair of vertices).
        # Wait, a 4-cycle has 2 pairs of opposite vertices.
        # So count // 2 is correct.

    def count_short_cycles(adj, max_len=8):
        # For simplicity and speed, we only count 3 and 4 in the main loop.
        # If we need to verify girth >= 9, we do a full check.
        c3 = count_3_cycles(adj)
        c4 = count_4_cycles(adj)
        return c3 + c4

    def optimize_graph(n, k, g, time_limit):
        adj = gen_random_k_reg(n, k)
        if not adj:
            return None
            
        current_score = count_short_cycles(adj)
        best_adj = [list(a) for a in adj]
        best_score = current_score
        
        T = 1.0
        cooling = 0.999
        steps = 0
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            steps += 1
            if steps % 1000 == 0:
                T *= cooling
            
            # Pick two random edges (u,v) and (x,y)
            # Flatten edges for easy picking
            # This is inefficient for large n. 
            # Better: pick random u, random v in adj[u], random x != u,v, random y in adj[x] != u,v
            u = random.randint(0, n-1)
            v = random.choice(adj[u])
            
            # Find x, y
            found = False
            for _ in range(100): # Try a few times
                x = random.randint(0, n-1)
                if x == u or x == v:
                    continue
                y = random.choice(adj[x])
                if y == u or y == v:
                    continue
                # Check if (u,x) and (v,y) are valid new edges
                if x in adj[u] or y in adj[v]:
                    continue
                if x in adj[v] or y in adj[u]:
                    continue
                found = True
                break
            
            if not found:
                continue
            
            # Perform switch: remove (u,v), (x,y); add (u,x), (v,y)
            # Check validity again
            if x in adj[u] or y in adj[v] or x in adj[v] or y in adj[u]:
                continue
                
            # Apply
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            new_score = count_short_cycles(adj)
            delta = new_score - current_score
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = [list(a) for a in adj]
                    if best_score == 0:
                        # Check full girth
                        edges = []
                        for i in range(n):
                            for j in adj[i]:
                                if i < j:
                                    edges.append((i, j))
                        if check_girth_fast(n, edges, g):
                            return n, edges
            else:
                # Revert
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                
                adj[u].append(v)
                adj[v].append(u)
                adj[x].append(y)
                adj[y].append(x)

        return None

    # Try to find a smaller graph than LUW
    # Search range: 1000 to min(LUW_n, 1120)
    search_start = min(current_best_n - 1, 1120)
    search_end = max(1000, search_start - 100)
    
    # We want the smallest n, so iterate from search_start down to search_end?
    # No, smaller n is harder. Start from the known best 1116 and go down?
    # Or start from 1116 and go up to LUW_n?
    # The goal is to minimize n.
    # If LUW_n is 1250, we try 1116, 1115, ..., 1000.
    # If we find 1116, we return it.
    # If we don't find 1116, we try 1115?
    # It is more likely to find 1116 than 1000.
    # So iterate n from 1116 down to search_end.
    
    # But if LUW_n < 1116, we iterate from LUW_n down.
    
    for n in range(search_start, search_end - 1, -1):
        if time.time() > deadline - 20:
            break
        
        # Allocate time for this n
        time_per_n = min(5.0, (deadline - 20 - time.time()) / max(1, search_start - search_end))
        
        res = optimize_graph(n, k, g, time_per_n)
        if res:
            n_found, edges_found = res
            if n_found < current_best_n:
                current_best_n = n_found
                current_best_edges = edges_found
            # If we found a valid graph, we can stop if we are at the lower bound or if we want to try smaller?
            # Given time, we should probably stop at the first valid one found in the descending order?
            # No, if we find 1116, we should check 1115?
            # But 1115 might be impossible or very hard.
            # Let's try a few n values.
            
    return current_best_n, current_best_edges
