"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base = set()
    for e in P['edges']:
        base.add((e[0], e[1]))
        base.add((e[1], e[0]))
    
    cands = [(u, v) for u in range(n) for v in range(u+1, n) if (u,v) not in base and v-u >= 3]
    
    def check(added):
        adj = [set() for _ in range(n)]
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            nbrs = list(adj[u])
            for i in range(len(nbrs)):
                for j in range(i+1, len(nbrs)):
                    if nbrs[j] in adj[nbrs[i]]:
                        return False
        for s in range(n):
            dist = [-1]*n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u]+1
                        if dist[v] > 4:
                            return False
                        q.append(v)
            if -1 in dist:
                return False
        return True
    
    # Try the 12-edge symmetric solution
    sol12 = [(i, i+12) for i in range(12)]
    if check(sol12):
        return [list(e) for e in sol12]
    
    # Try 8-edge solution
    sol8 = [(0,12),(1,13),(2,14),(3,15),(8,20),(9,21),(10,22),(11,23)]
    if check(sol8):
        return [list(e) for e in sol8]
    
    # Try 6-edge solutions
    sol6 = [(0,12),(1,13),(2,14),(9,21),(10,22),(11,23)]
    if check(sol6):
        return [list(e) for e in sol6]
    
    sol6b = [(0,12),(1,13),(2,14),(3,15),(8,20),(9,21)]
    if check(sol6b):
        return [list(e) for e in sol6b]
    
    sol6c = [(0,12),(1,13),(10,22),(11,23),(4,16),(5,17)]
    if check(sol6c):
        return [list(e) for e in sol6c]
    
    # Greedy with time limit
    t0 = time.time()
    current = []
    while time.time() - t0 < 60:
        adj = [set() for _ in range(n)]
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in current:
            adj[u].add(v)
            adj[v].add(u)
        
        best_e = None
        best_d = 999
        for e in cands:
            if e in current:
                continue
            test = current + [e]
            if check(test):
                d = 0
                for s in range(n):
                    dist = [-1]*n
                    dist[s] = 0
                    q = deque([s])
                    while q:
                        u = q.popleft()
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u]+1
                                q.append(v)
                    if -1 in dist:
                        d = 999
                        break
                    d = max(d, max(dist))
                if d < best_d:
                    best_d = d
                    best_e = e
                if best_d <= 4:
                    break
        if best_e is None:
            break
        current.append(best_e)
        if best_d <= 4:
            break
    
    if check(current):
        return [list(e) for e in current]
    return []
# EVOLVE-BLOCK-END

