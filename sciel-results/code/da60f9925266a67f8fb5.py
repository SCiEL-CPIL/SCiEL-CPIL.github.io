"""Find maximum valid edge set: no 3-edge path s-u-v-t with deg(s)==deg(t)."""
import time
import random
from collections import Counter

# EVOLVE-BLOCK-START
N = 15
ALL_EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
M = len(ALL_EDGES)

def _degrees_from_mask(mask):
    deg = [0] * N
    for k, (u, v) in enumerate(ALL_EDGES):
        if mask >> k & 1:
            deg[u] += 1
            deg[v] += 1
    return deg

def _is_valid_mask(mask):
    deg = _degrees_from_mask(mask)
    for k, (u, v) in enumerate(ALL_EDGES):
        if not (mask >> k & 1):
            continue
        for s in range(N):
            if s == u or s == v:
                continue
            if not (mask >> (s * N + u if s < u else u * N + s) & 1):
                continue
            for t in range(N):
                if t == u or t == v or t == s:
                    continue
                if not (mask >> (t * N + v if t < v else v * N + t) & 1):
                    continue
                if deg[s] == deg[t]:
                    return False
    return True

def _check_validity(adj, deg):
    """Check if adj (list of sets) is valid given deg."""
    for u in range(N):
        for v in adj[u]:
            if u >= v:
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

def _can_add(adj, deg, u, v):
    """Check if adding edge (u,v) keeps the graph valid."""
    # After adding, deg[u] and deg[v] increase by 1
    new_deg_u = deg[u] + 1
    new_deg_v = deg[v] + 1
    
    # We need to check all paths s-u-v-t and s-v-u-t
    # For path s-u-v-t: s in N(u)\{v}, t in N(v)\{u}, s!=t, need deg[s]!=deg[t]
    # But also the new edge itself creates paths through it.
    # Actually, adding edge (u,v) only affects validity of paths that use this edge.
    # A path s-u-v-t uses edge (u,v) if s~u, v~t, s!=t, s!=v, t!=u.
    # Also path s-v-u-t uses edge (u,v) if s~v, u~t, s!=t, s!=u, t!=v.
    
    for s in adj[u]:
        if s == v:
            continue
        for t in adj[v]:
            if t == u or t == s:
                continue
            if deg[s] == deg[t]:
                return False
    
    for s in adj[v]:
        if s == u:
            continue
        for t in adj[u]:
            if t == v or t == s:
                continue
            if deg[s] == deg[t]:
                return False
    
    return True

def _remove_edge(adj, deg, u, v):
    adj[u].discard(v)
    adj[v].discard(u)
    deg[u] -= 1
    deg[v] -= 1

def _add_edge(adj, deg, u, v):
    adj[u].add(v)
    adj[v].add(u)
    deg[u] += 1
    deg[v] += 1

def _local_search(adj, deg, deadline):
    """Try to add any missing edge. Return improved adj, deg, count."""
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for u in range(N):
            for v in range(u+1, N):
                if v in adj[u]:
                    continue
                if _can_add(adj, deg, u, v):
                    _add_edge(adj, deg, u, v)
                    improved = True
                    break
            if improved:
                break
    count = sum(len(adj[u]) for u in range(N)) // 2
    return adj, deg, count

def construct():
    deadline = time.time() + 85.0
    best_count = 0
    best_adj = None
    best_deg = None
    
    # Strategy 1: Start from K_{7,8} bipartite graph
    # Vertices 0-6 in left, 7-14 in right
    adj = [set() for _ in range(N)]
    deg = [0] * N
    for i in range(7):
        for j in range(7, 15):
            adj[i].add(j)
            adj[j].add(i)
    for i in range(7):
        deg[i] = 8
    for j in range(7, 15):
        deg[j] = 7
    
    count = 56
    # Verify validity
    if _check_validity(adj, deg):
        best_count = count
        best_adj = [set(s) for s in adj]
        best_deg = deg[:]
        # Local search
        adj, deg, count = _local_search(adj, deg, deadline - 5)
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
            best_deg = deg[:]
    
    # Strategy 2: Multiple random greedy starts
    for seed in range(30):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed * 7919 + 13)
        adj = [set() for _ in range(N)]
        deg = [0] * N
        edges = ALL_EDGES[:]
        rng.shuffle(edges)
        for (u, v) in edges:
            if _can_add(adj, deg, u, v):
                _add_edge(adj, deg, u, v)
        count = sum(len(adj[u]) for u in range(N)) // 2
        # Local search
        adj, deg, count = _local_search(adj, deg, deadline - 3)
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
            best_deg = deg[:]
    
    # Strategy 3: Perturbation-based local search from best solution
    if best_adj is not None and time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            rng = random.Random(_ * 31 + 7)
            adj = [set(s) for s in best_adj]
            deg = best_deg[:]
            # Remove 3-8 random edges to perturb
            cur_edges = [(u, v) for u in range(N) for v in adj[u] if u < v]
            n_remove = min(len(cur_edges), rng.randint(3, 8))
            to_remove = rng.sample(cur_edges, n_remove)
            for (u, v) in to_remove:
                _remove_edge(adj, deg, u, v)
            count = sum(len(adj[u]) for u in range(N)) // 2
            # Local search
            adj, deg, count = _local_search(adj, deg, deadline - 3)
            if count > best_count:
                best_count = count
                best_adj = [set(s) for s in adj]
                best_deg = deg[:]
    
    # Final polish
    if best_adj is not None and time.time() < deadline - 1:
        best_adj, best_deg, best_count = _local_search(best_adj, best_deg, deadline)
    
    if best_adj is None:
        return []
    
    edges = [[u, v] for u in range(N) for v in best_adj[u] if u < v]
    return edges
# EVOLVE-BLOCK-END
