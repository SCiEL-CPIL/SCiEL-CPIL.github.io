import time
import random

N = 16
ALL_BITS = (1 << N) - 1

def _groetzsch_edges():
    """
    Grötzsch graph (Mycielski of C5).
    11 vertices, 20 edges.
    Triangle-free, 4-chromatic.
    """
    k = 5
    edges = []
    # Cycle 0-1-2-3-4-0
    for i in range(k):
        edges.append([i, (i + 1) % k])
    # Copy vertices 5-9
    for i in range(k):
        edges.append([i, k + (i + 1) % k])
        edges.append([(i + 1) % k, k + i])
    # Center vertex 10
    for i in range(k):
        edges.append([2 * k, k + i])
    return edges

def _adj_from_edges(edge_list):
    adj = [0] * N
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _edge_count(adj):
    return sum(bin(a).count("1") for a in adj) // 2

def _is_triangle_free_add(adj, u, v):
    return (adj[u] & adj[v]) == 0

def _greedy_fill(adj, core_adj, deadline, rng):
    """
    Greedily add edges to adj, respecting core_adj (edges that must stay) 
    and triangle-freeness.
    """
    # Identify core edges
    core_edges = set()
    for u in range(N):
        m = core_adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if v > u:
                core_edges.add((u, v))
    
    # Candidate edges to add
    candidates = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) not in core_edges and not ((adj[u] >> v) & 1):
                candidates.append((u, v))
    
    rng.shuffle(candidates)
    
    for u, v in candidates:
        if time.time() > deadline:
            break
        if _is_triangle_free_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    return adj

def _local_improve(adj, core_adj, deadline, rng):
    """
    Local search to maximize edges.
    """
    core_edges = set()
    for u in range(N):
        m = core_adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if v > u:
                core_edges.add((u, v))
                
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any missing edge
        missing = []
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1) and (u, v) not in core_edges:
                    missing.append((u, v))
        rng.shuffle(missing)
        
        for u, v in missing:
            if _is_triangle_free_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                improved = True
                break
        
        if not improved:
            # 2. Try to remove a non-core edge and re-fill
            non_core_edges = []
            for u in range(N):
                for v in range(u + 1, N):
                    if ((adj[u] >> v) & 1) and (u, v) not in core_edges:
                        non_core_edges.append((u, v))
                        
            if non_core_edges:
                best_adj = None
                best_cnt = _edge_count(adj)
                
                # Try a few random removals
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    rem_edge = rng.choice(non_core_edges)
                    u, v = rem_edge
                    # Remove
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    # Greedy fill
                    adj2 = [a for a in adj]
                    adj2 = _greedy_fill(adj2, core_adj, deadline, rng)
                    cnt2 = _edge_count(adj2)
                    
                    if cnt2 > best_cnt:
                        best_cnt = cnt2
                        best_adj = adj2
                    
                    # Restore
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                if best_adj:
                    adj = best_adj
                    improved = True

    return adj

def construct():
    start = time.time()
    deadline = start + 110.0
    
    # Use Grötzsch as core
    core_edges = _groetzsch_edges()
    core_adj = _adj_from_edges(core_edges)
    
    best_adj = None
    best_cnt = -1
    
    rng = random.Random(42)
    
    # Multiple trials
    for trial in range(1000):
        if time.time() > deadline - 2.0:
            break
            
        adj = [a for a in core_adj]
        
        # Greedy fill
        adj = _greedy_fill(adj, core_adj, deadline, rng)
        
        # Local improve
        adj = _local_improve(adj, core_adj, deadline, rng)
        
        cnt = _edge_count(adj)
        if cnt > best_cnt:
            best_cnt = cnt
            best_adj = adj
            
    # Fallback
    if best_adj is None:
        best_adj = core_adj
        
    # Convert back to edge list
    out = []
    for u in range(N):
        m = best_adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if v > u:
                out.append([u, v])
                
    return out
