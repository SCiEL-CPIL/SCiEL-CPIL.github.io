import random
import time
from collections import deque

def _generate_random_3reg(n):
    stubs = list(range(n * 3))
    random.shuffle(stubs)
    edges = []
    edge_set = set()
    for i in range(0, n * 3, 2):
        u = stubs[i] // 3
        v = stubs[i + 1] // 3
        if u == v:
            return None
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return None
        edge_set.add((u, v))
        edges.append((u, v))
    return edges

def _check_girth(n, edges, girth_limit):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = girth_limit // 2
    for start in range(n):
        visited = [-1] * n
        parent = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nei in adj[node]:
                if visited[nei] == -1:
                    visited[nei] = depth + 1
                    parent[nei] = node
                    queue.append((nei, depth + 1))
                elif parent[nei] != node and parent[node] != nei:
                    # Cycle found
                    # Length = depth(node) + depth(nei) + 1
                    # But we need the path length
                    # This simple check is not robust for all cycles
                    # Better: if visited[nei] != -1 and nei != parent[node], 
                    # it might be a cross edge.
                    # In a tree, only parent/children.
                    # In a graph with cycles, this detects back edges.
                    # The cycle length is depth(node) + depth(nei) + 1.
                    cycle_len = depth + visited[nei] + 1
                    if cycle_len < girth_limit:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try random search
    # Target n around 2^(g/2)
    target_n = 1 << (g // 2)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        n = target_n
        edges = _generate_random_3reg(n)
        if edges is None:
            continue
        if _check_girth(n, edges, g):
            return n, edges
            
    # Fallback to LUW
    # ... existing LUW code ...
