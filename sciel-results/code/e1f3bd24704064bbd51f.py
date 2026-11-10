import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 15
    A = list(range(7))
    B = list(range(7, 15))
    
    # Helper to check validity - optimized with degree grouping
    def is_valid(edge_set):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        # For each edge (u,v), check if there exist s in N(u)\{v}, t in N(v)\{u}
        # with s != t and deg[s] == deg[t]
        for (u, v) in edge_set:
            nu = [s for s in adj[u] if s != v]
            nv = [t for t in adj[v] if t != u]
            
            # Group nv by degree
            deg_map = {}
            for t in nv:
                d = deg[t]
                if d not in deg_map:
                    deg_map[d] = []
                deg_map[d].append(t)
            
            for s in nu:
                ds = deg[s]
                if ds in deg_map:
                    # Check if there's a t != s with same degree
                    if len(deg_map[ds]) > 1:
                        return False
                    elif deg_map[ds][0] != s:
                        return False
        
        return True
    
    # Start with K_{7,8}
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((a, b))
    
    # Generate all candidate intra-partition edges
    candidates = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            candidates.append((A[i], A[j]))
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            candidates.append((B[i], B[j]))
    
    best_set = base_edges
    best_count = len(base_edges)
    
    # Multiple random restarts with greedy + local search
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        random.shuffle(candidates)
        current_set = set(base_edges)
        
        # Greedy phase: add edges in random order if valid
        for e in candidates:
            if time.time() > deadline - 3.0:
                break
            test_set = set(current_set)
            test_set.add(e)
            if is_valid(test_set):
                current_set = test_set
        
        # Local search phase: try to add remaining edges
        remaining = [e for e in candidates if e not in current_set]
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            random.shuffle(remaining)
            for e in remaining:
                if time.time() > deadline - 1.0:
                    break
                test_set = set(current_set)
                test_set.add(e)
                if is_valid(test_set):
                    current_set = test_set
                    remaining.remove(e)
                    improved = True
                    break
        
        if len(current_set) > best_count:
            best_set = current_set
            best_count = len(current_set)
        
        # Early termination if we've found a good solution
        if best_count >= 60:
            break
    
    # Convert to required format
    result = []
    for u, v in best_set:
        result.append([int(u), int(v)])
    
    return result
