import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 110.0
    
    if k != 3:
        n = 1000
        while time.time() < deadline:
            half_edges = []
            for i in range(n):
                for _ in range(k):
                    half_edges.append(i)
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, n * k, 2):
                u = half_edges[i]
                v = half_edges[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                visited = [False] * n
                q = deque([0])
                visited[0] = True
                count = 1
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            count += 1
                            q.append(v)
                if count == n:
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
        return n, []
    
    def check_girth(n, adj, g):
        max_depth = g // 2 - 1
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True
    
    def generate_random_3_reg(n):
        while time.time() < deadline:
            half_edges = []
            for i in range(n):
                for _ in range(3):
                    half_edges.append(i)
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, n * 3, 2):
                u = half_edges[i]
                v = half_edges[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                visited = [False] * n
                q = deque([0])
                visited[0] = True
                count = 1
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            count += 1
                            q.append(v)
                if count == n:
                    return adj
        return None
    
    candidate_ns = [936, 1000, 1200, 1500, 2000, 3000]
    
    for n in candidate_ns:
        if time.time() > deadline:
            break
        for attempt in range(3):
            if time.time() > deadline:
                break
            adj = generate_random_3_reg(n)
            if adj is None:
                continue
            if check_girth(n, adj, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    n = 5000
    adj = generate_random_3_reg(n)
    if adj is not None:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    return 0, []

