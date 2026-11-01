import time
import random
import sys
from itertools import product

def construct(k, g):
    deadline = time.time() + 115
    
    if k == 4 and g == 10:
        # Try to find the 320 cage via smart construction
        result = try_find_cage_smart(320, k, g, deadline - time.time() - 5)
        if result:
            return result
        
        # Fallback: LUW construction
        return luw_construct(k, g)
    
    # For other parameters, use LUW or general approach
    return luw_construct(k, g)

def luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction for cage graphs."""
    q = k
    d = max(3, (g + 2) // 2)
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

def try_find_cage_smart(n, k, g, time_limit):
    """Try to find a k-regular graph of girth g with n vertices using structured random search."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Use a structured approach: start with a good random regular graph
    # and use local search to improve girth
    
    def fast_regular_graph(n, k, seed=None):
        """Generate a random k-regular graph using configuration model."""
        if seed is not None:
            random.seed(seed)
        for attempt in range(5000):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            edges = []
            valid = True
            seen = set()
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u,v), max(u,v))
                if key in seen:
                    valid = False
                    break
                seen.add(key)
                edges.append((u, v))
            if valid:
                return edges
        return None
    
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def count_short_cycles(n, adj, g):
        """Count cycles shorter than g using BFS from each vertex."""
        count = 0
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                count += 1
                                break  # Only count one per start vertex
        return count
    
    def verify_girth(n, adj, g):
        """Check if girth >= g using non-backtracking walks."""
        for start in range(n):
            frontier = []
            for nb in adj[start]:
                frontier.append((nb, start))
            dist = 1
            while frontier and dist < g:
                new_frontier = []
                for (cur, prev) in frontier:
                    for nb in adj[cur]:
                        if nb == prev:
                            continue
                        if nb == start:
                            return False
                        new_frontier.append((nb, cur))
                frontier = new_frontier
                dist += 1
        return True
    
    def edge_swap(edges, adj, n, k, g, edge_set, deadline):
        """Perform a single edge swap to try to improve girth."""
        # Find a short cycle
        for start in range(n):
            if time.time() > deadline:
                return None
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            cycle_found = None
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                # Reconstruct cycle
                                path1 = []
                                cur = u
                                while cur != start:
                                    path1.append(cur)
                                    cur = parent[cur]
                                path1.append(start)
                                path1.reverse()
                                
                                path2 = []
                                cur = v
                                while cur != start:
                                    path2.append(cur)
                                    cur = parent[cur]
                                path2.append(start)
                                path2.reverse()
                                
                                i = 0
                                while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
                                    i += 1
                                if i > 0:
                                    cycle = path1[i:] + path2[i-1::-1]
                                    if len(cycle) >= 3:
                                        cycle_found = cycle
                                        break
                if cycle_found:
                    break
            if cycle_found:
                # Try to break this cycle
                L = len(cycle_found)
                for _ in range(50):
                    if time.time() > deadline:
                        return None
                    idx = random.randint(0, L - 1)
                    u = cycle_found[idx]
                    v = cycle_found[(idx + 1) % L]
                    
                    # Find a random edge to swap with
                    a, b = random.choice(edges)
                    if a in (u, v) or b in (u, v):
                        continue
                    
                    new_e1 = (min(u,a), max(u,a))
                    new_e2 = (min(v,b), max(v,b))
                    old_e1 = (min(u,v), max(u,v))
                    old_e2 = (min(a,b), max(a,b))
                    
                    if new_e1 in edge_set or new_e2 in edge_set:
                        continue
                    if new_e1 == old_e1 or new_e2 == old_e2:
                        continue
                    
                    # Perform swap
                    new_edge_set = set(edge_set)
                    new_edge_set.discard(old_e1)
                    new_edge_set.discard(old_e2)
                    new_edge_set.add(new_e1)
                    new_edge_set.add(new_e2)
                    
                    new_edges = list(new_edge_set)
                    new_adj = build_adj(new_edges, n)
                    
                    return new_edges, new_adj, new_edge_set
        return None
    
    # Try multiple random initial graphs
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        edges = fast_regular_graph(n, k, seed=attempt)
        if edges is None:
            continue
        
        adj = build_adj(edges, n)
        edge_set = set((min(u,v), max(u,v)) for u,v in edges)
        
        # Local search
        stall = 0
        for step in range(5000):
            if time.time() > deadline:
                break
            
            if verify_girth(n, adj, g):
                return n, edges
            
            result = edge_swap(edges, adj, n, k, g, edge_set, deadline)
            if result is None:
                stall += 1
                if stall > 50:
                    break
            else:
                edges, adj, edge_set = result
                stall = 0
    
    return None

# Pre-computed 320-vertex cage if we can find one
# For now, rely on the search

if __name__ == "__main__":
    n, edges = construct(4, 10)
    print(f"Found graph with {n} vertices and {len(edges)} edges")
