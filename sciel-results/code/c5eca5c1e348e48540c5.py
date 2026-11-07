"""Improved program: Start from K_{7,8} (provably valid, 56 edges) + local search for intra-partition edges."""
# EVOLVE-BLOCK-START
import time
import random

def _check_validity(adj, deg, n, edges):
    """Check no 3-edge path s-u-v-t (s!=t) has deg(s)==deg(t)."""
    for (u, v) in edges:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u:
                    continue
                if s == t:
                    continue
                if deg[s] == deg[t]:
                    return False
    return True

def _try_add_edge(adj, deg, n, edges, edge_set, u, v):
    """Try to add edge (u,v). Returns True if added, False if invalid."""
    new_edges = edges + [[u, v]]
    new_adj = [set() for _ in range(n)]
    new_deg = [0] * n
    for (a, b) in new_edges:
        new_adj[a].add(b)
        new_adj[b].add(a)
        new_deg[a] += 1
        new_deg[b] += 1
    
    if _check_validity(new_adj, new_deg, n, new_edges):
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        edges.append([u, v])
        edge_set.add((u, v))
        return True
    return False

def _try_remove_edge(adj, deg, n, edges, edge_set, u, v):
    """Remove edge (u,v)."""
    adj[u].discard(v)
    adj[v].discard(u)
    deg[u] -= 1
    deg[v] -= 1
    edges.remove([u, v])
    edge_set.discard((u, v))

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    n = 15
    
    # Phase 1: Start with K_{7,8} - provably valid
    # Partition A = {0,1,2,3,4,5,6} (size 7), Partition B = {7,8,9,10,11,12,13,14} (size 8)
    partition_A = list(range(7))
    partition_B = list(range(7, 15))
    
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edges = []
    edge_set = set()
    
    # Add all cross-partition edges (K_{7,8})
    for u in partition_A:
        for v in partition_B:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            edges.append([u, v])
            edge_set.add((u, v))
    
    # Verify K_{7,8} is valid
    assert _check_validity(adj, deg, n, edges), "K_{7,8} should be valid"
    
    best_edges = [list(e) for e in edges]
    best_count = len(edges)  # 56
    
    # Phase 2: Try to add intra-partition edges
    intra_candidates = []
    for i in range(len(partition_A)):
        for j in range(i+1, len(partition_A)):
            intra_candidates.append((partition_A[i], partition_A[j]))
    for i in range(len(partition_B)):
        for j in range(i+1, len(partition_B)):
            intra_candidates.append((partition_B[i], partition_B[j]))
    
    # Try adding each intra-partition edge
    for (u, v) in intra_candidates:
        if time.time() > deadline - 5:
            break
        if (u, v) in edge_set:
            continue
        if _try_add_edge(adj, deg, n, edges, edge_set, u, v):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = [list(e) for e in edges]
    
    # Phase 3: Local search - try swaps (remove one edge, add another)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(edges)):
            if time.time() > deadline - 5:
                break
            u, v = edges[i][0], edges[i][1]
            _try_remove_edge(adj, deg, n, edges, edge_set, u, v)
            
            for (a, b) in intra_candidates:
                if time.time() > deadline - 5:
                    break
                if (a, b) in edge_set:
                    continue
                if _try_add_edge(adj, deg, n, edges, edge_set, a, b):
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = [list(e) for e in edges]
                    improved = True
                    for (c, d) in intra_candidates:
                        if time.time() > deadline - 5:
                            break
                        if (c, d) in edge_set:
                            continue
                        if _try_add_edge(adj, deg, n, edges, edge_set, c, d):
                            if len(edges) > best_count:
                                best_count = len(edges)
                                best_edges = [list(e) for e in edges]
                            improved = True
                            break
                    break
                else:
                    _try_add_edge(adj, deg, n, edges, edge_set, u, v)
                    break
            
            if (u, v) not in edge_set:
                if len(edges) <= best_count:
                    _try_add_edge(adj, deg, n, edges, edge_set, u, v)
    
    # Final validation
    if best_edges:
        adj2 = [set() for _ in range(n)]
        deg2 = [0] * n
        for (u, v) in best_edges:
            adj2[u].add(v)
            adj2[v].add(u)
            deg2[u] += 1
            deg2[v] += 1
        if not _check_validity(adj2, deg2, n, best_edges):
            best_edges = []
            for u in partition_A:
                for v in partition_B:
                    best_edges.append([u, v])
    
    result = [[int(u), int(v)] for (u, v) in best_edges]
    return result
# EVOLVE-BLOCK-END
