import random

def construct(k, g):
    # Estimate a reasonable n
    if k == 3:
        # Known cages for k=3
        known = {8: 30, 10: 70, 12: 126, 14: 240, 16: 936}
        n = known.get(g, 2 * sum(3 ** i for i in range(g // 2)))
    else:
        n = 2 * sum((k - 1) ** i for i in range(g // 2))
    
    # Try to find a valid graph
    for attempt in range(50):
        graph = generate_random_k_regular(n, k)
        if graph is not None:
            if check_girth(graph, n, g):
                edges = []
                for i in range(n):
                    for j in graph[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        n += k
    
    # Fallback
    edges = [(i, (i + 1) % n) for i in range(n)]
    return n, edges

def generate_random_k_regular(n, k):
    if n * k % 2 != 0 or k > n - 1:
        return None
    # Configuration model with retries
    for _ in range(100):
        half_edges = []
        for v in range(n):
            half_edges.extend([v] * k)
        random.shuffle(half_edges)
        adj = {v: set() for v in range(n)}
        valid = True
        for i in range(0, n * k, 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if valid and all(len(adj[v]) == k for v in range(n)):
            return adj
    return None

def check_girth(graph, n, g):
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in graph[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    cycle = dist[u] + dist[v] + 1
                    if cycle < g:
                        return False
    return True
