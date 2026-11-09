import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 15
    
    def make_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def get_degrees(adj):
        return [len(adj[i]) for i in range(n)]
    
    def is_valid(adj):
        degs = get_degrees(adj)
        for u in range(n):
            for v in adj[u]:
                if v > u:  # avoid double counting
                    for s in adj[u]:
                        if s == v:
                            continue
                        for t in adj[v]:
                            if t == u:
                                continue
                            if s == t:
                                continue
                            if degs[s] == degs[t]:
                                return False
        return True
    
    # Start with K_{7,8}
    A = list(range(7))   # 0..6
    B = list(range(7, 15))  # 7..14
    
    base_edges = []
    for a in A:
        for b in B:
            base_edges.append([a, b])
    
    adj = make_adj(base_edges)
    
    # Verify base is valid
    if not is_valid(adj):
        # Fallback to star
        return [[0, v] for v in range(1, n)]
    
    best_edges = list(base_edges)
    best_count = len(base_edges)
    
    # Try to add edges within A or within B
    candidate_edges = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            candidate_edges.append([A[i], A[j]])
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            candidate_edges.append([B[i], B[j]])
    
    # Greedy: try adding each candidate edge if it maintains validity
    random.shuffle(candidate_edges)
    for edge in candidate_edges:
        if time.time() > deadline:
            break
        u, v = edge
        if v in adj[u]:
            continue
        # Try adding
        adj[u].add(v)
        adj[v].add(u)
        if is_valid(adj):
            best_edges.append(edge)
            best_count += 1
        else:
            adj[u].remove(v)
            adj[v].remove(u)
    
    # Local search: try swapping or adding more
    # Try all possible edges not yet in graph
    all_possible = []
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj[i]:
                all_possible.append([i, j])
    
    random.shuffle(all_possible)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for edge in all_possible:
            if time.time() > deadline:
                break
            u, v = edge
            if v in adj[u]:
                continue
            adj[u].add(v)
            adj[v].add(u)
            if is_valid(adj):
                best_edges.append(edge)
                best_count += 1
                improved = True
                # Remove from all_possible
                all_possible.remove(edge)
            else:
                adj[u].remove(v)
                adj[v].remove(u)
    
    # Convert to proper list of lists with int types
    result = []
    for e in best_edges:
        u, v = e
        if isinstance(u, int) and isinstance(v, int):
            result.append([int(u), int(v)])
        else:
            result.append([int(u), int(v)])
    
    return result
