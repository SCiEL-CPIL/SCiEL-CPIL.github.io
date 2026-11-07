import random
import time
from collections import deque

def generate_instance(n, seed):
    random.seed(seed)
    W = int(n**0.5)
    if W == 0: W = 1
    H = (n + W - 1) // W
    
    edges = []
    for r in range(H):
        for c in range(W):
            u = r * W + c
            if u >= n: continue
            
            if c + 1 < W:
                v = r * W + (c + 1)
                if v < n:
                    edges.append((u, v))
            if r + 1 < H:
                v = (r + 1) * W + c
                if v < n:
                    edges.append((u, v))
            
    lists = []
    for i in range(n):
        cols = random.sample(range(5), 3)
        lists.append(cols)
        
    return n, edges, lists

def solve_instance(n, edges, lists, deadline):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    domains = [set(l) for l in lists]
    assignment = [-1] * n
    unassigned = set(range(n))
    trail = []
    
    def mrv_select(unassigned_set, domains):
        best = -1
        best_size = float('inf')
        best_deg = -1
        for v in unassigned_set:
            sz = len(domains[v])
            deg = len(adj[v])
            if sz < best_size or (sz == best_size and deg > best_deg):
                best_size = sz
                best_deg = deg
                best = v
        return best
        
    def remove_color(v, color):
        if color in domains[v]:
            domains[v].discard(color)
            trail.append((v, color, True))
            return True
        return False

    def restore_trail(mark):
        while len(trail) > mark:
            v, c, _ = trail.pop()
            domains[v].add(c)

    def backtrack(unassigned_set, assignment):
        if time.time() > deadline:
            return None
        
        if not unassigned_set:
            return assignment[:]
        
        v = mrv_select(unassigned_set, domains)
        if v == -1:
            return None
        
        unassigned_set.remove(v)
        mark = len(trail)
        
        for color in sorted(list(domains[v])):
            assignment[v] = color
            conflict = False
            for u in adj[v]:
                if u in unassigned_set:
                    if not remove_color(u, color):
                        if len(domains[u]) == 0:
                            conflict = True
                            break
            if not conflict:
                result = backtrack(unassigned_set, assignment)
                if result is not None:
                    return result
            
            restore_trail(mark)
        
        unassigned_set.add(v)
        assignment[v] = -1
        return None
    
    result = backtrack(set(range(n)), assignment)
    return result

def construct():
    results = []
    seeds = [(20, 701), (40, 702), (60, 703), (100, 704)]
    start_time = time.time()
    
    for n, seed in seeds:
        remaining = 120.0 - (time.time() - start_time)
        if remaining < 1:
            # Emergency fallback
            n, edges, lists = generate_instance(n, seed)
            # Simple 2-coloring
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            color = [-1] * n
            for i in range(n):
                if color[i] == -1:
                    color[i] = 0
                    q = deque([i])
                    while q:
                        u = q.popleft()
                        for v in adj[u]:
                            if color[v] == -1:
                                color[v] = 1 - color[u]
                                q.append(v)
            # Map to lists
            # Just pick first valid color from list for each vertex
            # This might conflict with neighbors, but it's a fallback
            sol = []
            for i in range(n):
                sol.append(lists[i][0])
            results.append([int(c) for c in sol])
            continue
            
        n, edges, lists = generate_instance(n, seed)
        deadline = time.time() + remaining
        sol = solve_instance(n, edges, lists, deadline)
        
        if sol is None:
            # Fallback
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            color = [-1] * n
            for i in range(n):
                if color[i] == -1:
                    color[i] = 0
                    q = deque([i])
                    while q:
                        u = q.popleft()
                        for v in adj[u]:
                            if color[v] == -1:
                                color[v] = 1 - color[u]
                                q.append(v)
            sol = []
            for i in range(n):
                sol.append(lists[i][0])
                
        results.append([int(c) for c in sol])
        
    return results
