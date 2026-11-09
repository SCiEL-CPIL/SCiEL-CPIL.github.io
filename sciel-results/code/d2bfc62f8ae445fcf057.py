"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 15}

def _check_validity(adj, n):
    """Check if no simple path s-u-v-t has deg(s)==deg(t)."""
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _count_violations(adj, n):
    """Count number of violating paths."""
    deg = [len(adj[i]) for i in range(n)]
    count = 0
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        count += 1
    return count

def _try_build_random(n, seed=42):
    """Try to build a dense valid graph via randomized local search."""
    rng = random.Random(seed)
    # Start with complete graph
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            adj[i].add(j)
            adj[j].add(i)
    
    # Remove edges to fix violations
    changed = True
    max_iter = 5000
    it = 0
    while changed and it < max_iter:
        changed = False
        it += 1
        # Find a violating edge and remove it
        deg = [len(adj[i]) for i in range(n)]
        for u in range(n):
            for v in list(adj[u]):
                if v <= u:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            # Remove edge (u,v)
                            adj[u].discard(v)
                            adj[v].discard(u)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
    
    # Now try to add back edges
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                if j not in adj[i]:
                    adj[i].add(j)
                    adj[j].add(i)
                    if not _check_validity(adj, n):
                        adj[i].discard(j)
                        adj[j].discard(i)
                    else:
                        improved = True
    
    return [[i, j] for i in range(n) for j in adj[i] if j > i]

def _try_build_structured(n):
    """Try a structured construction with diverse degrees."""
    # Vertex i gets connected to vertices with index difference constraints
    # to create diverse degree patterns
    adj = [set() for _ in range(n)]
    
    # Create a "scale-free" like structure
    # High-index vertices connect to many, low-index to few
    for i in range(n):
        # Vertex i connects to i+1, i+2, ..., up to some limit
        for j in range(i+1, min(n, i + 1 + i)):
            adj[i].add(j)
            adj[j].add(i)
    
    # Check validity
    if _check_validity(adj, n):
        # Try to add more edges
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if j not in adj[i]:
                        adj[i].add(j)
                        adj[j].add(i)
                        if not _check_validity(adj, n):
                            adj[i].discard(j)
                            adj[j].discard(i)
                        else:
                            improved = True
        return [[i, j] for i in range(n) for j in adj[i] if j > i]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    best_edges = None
    best_count = 0
    
    # Try structured construction first
    edges = _try_build_structured(P['n'])
    if edges is not None:
        best_edges = edges
        best_count = len(edges)
    
    # Try multiple random restarts
    for seed in range(50):
        if time.time() > deadline:
            break
        edges = _try_build_random(P['n'], seed=seed)
        if edges and len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    # Fallback to star if nothing works
    if best_edges is None:
        best_edges = [[0, v] for v in range(1, P['n'])]
    
    # Convert to proper format
    result = []
    for e in best_edges:
        u = int(e[0])
        v = int(e[1])
        result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
