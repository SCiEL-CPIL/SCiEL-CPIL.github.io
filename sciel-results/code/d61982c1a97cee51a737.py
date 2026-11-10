import itertools
import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Vertices 0..14
    # Base graph: K_{7,8}
    A = list(range(7))
    B = list(range(7, 15))
    
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add(tuple(sorted((a, b))))
            
    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edges):
        deg = get_degrees(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check all paths s-u-v-t
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    # Initial best solution
    best_edges = set(base_edges)
    best_count = len(best_edges)
    
    # Candidate edges to add: Internal to A, Internal to B
    candidates = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            candidates.append(tuple(sorted((A[i], A[j]))))
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            candidates.append(tuple(sorted((B[i], B[j]))))
            
    random.shuffle(candidates)
    
    current_edges = set(base_edges)
    
    # Pass 1: Simple Greedy
    for c in candidates:
        if c in current_edges:
            continue
        test_edges = current_edges | {c}
        if is_valid(test_edges):
            current_edges = test_edges
            
    if len(current_edges) > best_count:
        best_edges = current_edges
        best_count = len(current_edges)
        
    # Random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
            
        rand_candidates = candidates[:]
        random.shuffle(rand_candidates)
        
        curr = set(base_edges)
        
        for c in rand_candidates:
            if c in curr:
                continue
            test = curr | {c}
            if is_valid(test):
                curr = test
                
        if len(curr) > best_count:
            best_edges = curr
            best_count = len(curr)
            
    result = [list(e) for e in best_edges]
    result.sort()
    
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
        
    return result
