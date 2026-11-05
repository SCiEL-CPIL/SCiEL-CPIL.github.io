import random
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    # Fallback: LUW construction
    luw_n, luw_edges = luw_construct(k, g)
    
    # Try to find a better graph using local search
    best_n = luw_n
    best_edges = luw_edges
    
    # Try n from 320 to luw_n - 1
    for n in range(320, min(luw_n, 500)):
        if time.time() > deadline:
            break
        result = local_search(k, g, n, deadline)
        if result is not None:
            best_n = n
            best_edges = result
            break
    
    return best_n, best_edges

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

def local_search(k, g, n, deadline):
    # Generate a random k-regular graph
    graph = random_k_regular(n, k)
    if graph is None:
        return None
    
    # Check girth
    if has_girth_at_least(graph, n, g):
        return graph_to_edges(graph, n)
    
    # Local search
    for _ in range(100000):
        if time.time() > deadline:
            break
        # Try a random 2-switch
        new_graph = random_2_switch(graph, n, k)
        if new_graph is not None:
            if has_girth_at_least(new_graph, n, g):
                return graph_to_edges(new_graph, n)
            # Accept with some probability
            if random.random() < 0.001:
                graph = new_graph
    
    return None

def random_k_regular(n, k):
    # Generate a random k-regular graph using the configuration model
    for _ in range(1000):
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        edges = [(stubs[2*i], stubs[2*i+1]) for i in range(n * k // 2)]
        # Check for multi-edges and self-loops
        if len(set(edges)) == len(edges) and all(u != v for u, v in edges):
            # Build adjacency list
            graph = [[] for _ in range(n)]
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            return graph
    return None

def has_girth_at_least(graph, n, g):
    r = (g - 1) // 2
    for start in range(n):
        first_dist = [-1] * n
        first_parent = [-1] * n
        first_dist[start] = 0
        queue = [(start, -1, 0)]
        while queue:
            v, parent, dist = queue.pop(0)
            if dist == r:
                continue
            for w in graph[v]:
                if w == parent:
                    continue
                if first_dist[w] == -1:
                    first_dist[w] = dist + 1
                    first_parent[w] = v
                    queue.append((w, v, dist + 1))
                elif first_dist[w] == dist + 1 and first_parent[w] != v:
                    return False
    return True

def graph_to_edges(graph, n):
    edges = []
    for u in range(n):
        for v in graph[u]:
            if u < v:
                edges.append((u, v))
    return edges

def random_2_switch(graph, n, k):
    # Pick two random edges and swap them
    edges = []
    for u in range(n):
        for v in graph[u]:
            if u < v:
                edges.append((u, v))
    if len(edges) < 2:
        return None
    e1, e2 = random.sample(edges, 2)
    a, b = e1
    c, d = e2
    # Try swapping: (a, c) and (b, d)
    if c not in graph[a] and d not in graph[b]:
        new_graph = [list(neighbors) for neighbors in graph]
        new_graph[a].remove(b)
        new_graph[b].remove(a)
        new_graph[c].remove(d)
        new_graph[d].remove(c)
        new_graph[a].append(c)
        new_graph[c].append(a)
        new_graph[b].append(d)
        new_graph[d].append(b)
        return new_graph
    # Try swapping: (a, d) and (b, c)
    if d not in graph[a] and c not in graph[b]:
        new_graph = [list(neighbors) for neighbors in graph]
        new_graph[a].remove(b)
        new_graph[b].remove(a)
        new_graph[c].remove(d)
        new_graph[d].remove(c)
        new_graph[a].append(d)
        new_graph[d].append(a)
        new_graph[b].append(c)
        new_graph[c].append(b)
        return new_graph
    return None
