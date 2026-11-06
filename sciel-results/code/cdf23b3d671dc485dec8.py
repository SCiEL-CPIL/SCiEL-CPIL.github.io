"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _has_q3_fast(adj, n=16):
    """Fast check: Q3 is 3-regular bipartite on 8 vertices with 12 edges.
    Use the fact that in any Q3, each vertex has exactly 3 neighbors within the 8-set."""
    # Quick necessary condition: need at least 8 vertices with degree >= 3
    cands = [v for v in range(n) if adj[v] >= 3]
    if len(cands) < 8:
        return False
    
    # For each 8-subset, check if it induces a 3-regular bipartite graph
    # Optimize: precompute adjacency as bitmasks
    for subset in combinations(cands, 8):
        # Count edges and degrees within subset
        edge_count = 0
        degs = [0] * 8
        for idx, u in enumerate(subset):
            mask = 0
            for jdx in range(idx + 1, 8):
                v = subset[jdx]
                if adj[u] & (1 << v):
                    edge_count += 1
                    degs[idx] += 1
                    degs[jdx] += 1
        if edge_count != 12:
            continue
        if min(degs) < 3 or max(degs) > 3:
            continue
        # Check bipartite
        # Build local adjacency list
        local = [set() for _ in range(8)]
        for idx, u in enumerate(subset):
            for jdx in range(8):
                if idx != jdx and (adj[u] & (1 << subset[jdx])):
                    local[idx].add(jdx)
        color = [-1] * 8
        ok = True
        for s in range(8):
            if color[s] != -1:
                continue
            color[s] = 0
            stack = [s]
            while stack:
                u = stack.pop()
                for v in local[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        stack.append(v)
                    elif color[v] == color[u]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return True
    return False

def _edges_from_adj(adj, n=16):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _count_edges(adj):
    return sum(bin(a).count('1') for a in adj) // 2

def construct():
    n = 16
    deadline = time.time() + 135
    
    def random_adj(p=0.75):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    best_adj = None
    best_count = 0
    
    # Phase 1: Find a Q3-free graph via random + repair
    for restart in range(30):
        if time.time() > deadline:
            break
        
        adj = random_adj(0.7)
        
        # Repair: iteratively remove edges from Q3s
        for repair_iter in range(500):
            if time.time() > deadline:
                break
            if not _has_q3_fast(adj):
                break
            # Remove a random edge
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
            if not all_edges:
                break
            u, v = random.choice(all_edges)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        cnt = _count_edges(adj)
        if not _has_q3_fast(adj) and cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
    
    # Phase 2: Local search - try adding edges
    if best_adj is not None:
        adj = best_adj[:]
        for sa_iter in range(20000):
            if time.time() > deadline:
                break
            
            # Try adding a random missing edge
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
            if not missing:
                break
            i, j = random.choice(missing)
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
            if not _has_q3_fast(adj):
                cnt = _count_edges(adj)
                if cnt > best_count:
                    best_count = cnt
                    best_adj = adj[:]
            else:
                # Revert
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
        
        # Phase 3: Try swapping edges (remove one, add another)
        adj = best_adj[:]
        for swap_iter in range(10000):
            if time.time() > deadline:
                break
            
            existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
            if not existing or not missing:
                break
            
            rem_i, rem_j = random.choice(existing)
            add_i, add_j = random.choice(missing)
            
            # Remove
            adj[rem_i] &= ~(1 << rem_j)
            adj[rem_j] &= ~(1 << rem_i)
            # Add
            adj[add_i] |= (1 << add_j)
            adj[add_j] |= (1 << add_i)
            
            if not _has_q3_fast(adj):
                cnt = _count_edges(adj)
                if cnt > best_count:
                    best_count = cnt
                    best_adj = adj[:]
            else:
                # Revert
                adj[rem_i] |= (1 << rem_j)
                adj[rem_j] |= (1 << rem_i)
                adj[add_i] &= ~(1 << add_j)
                adj[add_j] &= ~(1 << add_i)
    
    if best_adj is None:
        # Fallback: K_{3,13}
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    return _edges_from_adj(best_adj)
# EVOLVE-BLOCK-END
