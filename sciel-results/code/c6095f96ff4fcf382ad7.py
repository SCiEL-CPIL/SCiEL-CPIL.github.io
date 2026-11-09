import time
import random
import numpy as np

N = 20
DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _popcount(x):
    return bin(x).count('1')

def _can_add_edge_fast(adj, u, v):
    """
    Check if adding edge (u,v) to a K3,3-free graph keeps it K3,3-free.
    adj is a list of int bitmasks.
    K3,3 exists iff there are 3 vertices with 3 common neighbors.
    Adding edge (u,v) creates a K3,3 only if there exist a1,a2 in N(v)\{u} 
    and b1,b2 in N(u)\{v} such that {a1,a2} and {b1,b2} form a K2,2 
    in the bipartite graph between them (i.e., all 4 cross-edges exist).
    """
    nu = adj[u]
    nv = adj[v]
    
    # Candidates for B-side (neighbors of u, excluding v)
    b_mask = nu & ~(1 << v)
    # Candidates for A-side (neighbors of v, excluding u)
    a_mask = nv & ~(1 << u)
    
    # Need at least 2 in each to form K3,3 with u,v
    if _popcount(b_mask) < 2 or _popcount(a_mask) < 2:
        return True
    
    # Check for K2,2 in bipartite graph (A, B)
    # A vertex x in A has degree d_x into B. If any two vertices in A
    # share >= 2 common neighbors in B, we have a K2,2.
    # Similarly for B side.
    
    # Convert to lists
    a_list = [i for i in range(N) if (a_mask >> i) & 1]
    b_list = [i for i in range(N) if (b_mask >> i) & 1]
    
    # Check pairs in smaller set
    if len(a_list) <= len(b_list):
        for i in range(len(a_list)):
            a1 = a_list[i]
            adj_a1 = adj[a1]
            for j in range(i + 1, len(a_list)):
                a2 = a_list[j]
                common = adj_a1 & adj[a2] & b_mask
                if _popcount(common) >= 2:
                    return False
    else:
        for i in range(len(b_list)):
            b1 = b_list[i]
            adj_b1 = adj[b1]
            for j in range(i + 1, len(b_list)):
                b2 = b_list[j]
                common = adj_b1 & adj[b2] & a_mask
                if _popcount(common) >= 2:
                    return False
    return True

def _edge_count(adj):
    return sum(_popcount(adj[u]) for u in range(N)) // 2

def _greedy_fill(adj, time_limit):
    """
    Greedily add all possible edges. Multiple passes until no improvement.
    """
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if time.time() - start_time > time_limit:
                    return adj
                if (adj[u] >> v) & 1:
                    continue
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
    return adj

def _fast_local_search(adj, time_limit):
    """
    Local search with greedy fill and targeted 1-swap moves.
    Key optimization: after removing an edge, only check edges that were
    blocked specifically by that edge's removal.
    """
    start_time = time.time()
    current_count = _edge_count(adj)
    best_adj = adj[:]
    best_count = current_count
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Greedy add missing edges
        for u in range(N):
            for v in range(u + 1, N):
                if time.time() - start_time > time_limit:
                    return best_adj
                if (adj[u] >> v) & 1:
                    continue
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_count += 1
                    improved = True
        
        # Phase 2: Try 1-swap moves
        if time.time() - start_time > time_limit:
            return best_adj
        
        # Collect all edges for swap attempts
        edges_in_graph = []
        for u in range(N):
            for v in range(u + 1, N):
                if (adj[u] >> v) & 1:
                    edges_in_graph.append((u, v))
        
        random.shuffle(edges_in_graph)
        
        for (u, v) in edges_in_graph:
            if time.time() - start_time > time_limit:
                return best_adj
            
            # Temporarily remove edge (u,v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Try to find a missing edge that can now be added
            found = False
            for u2 in range(N):
                for v2 in range(u2 + 1, N):
                    if time.time() - start_time > time_limit:
                        # Revert and return
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        return best_adj
                    if (adj[u2] >> v2) & 1:
                        continue
                    if (u2 == u and v2 == v):
                        continue
                    if _can_add_edge_fast(adj, u2, v2):
                        # Accept the swap
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        found = True
                        break
                if found:
                    break
            
            if not found:
                # Revert: put back (u,v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Phase 3: After swaps, try greedy add again
        if time.time() - start_time > time_limit:
            return best_adj
        
        for u in range(N):
            for v in range(u + 1, N):
                if time.time() - start_time > time_limit:
                    return best_adj
                if (adj[u] >> v) & 1:
                    continue
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_count += 1
                    improved = True
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        if not improved:
            break
            
    return best_adj

def _random_start(density):
    """Generate a random graph with given edge probability, filtered for K3,3."""
    adj = [0] * N
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if random.random() < density:
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
    return adj

def _structured_start():
    """
    Create a structured K3,3-free graph.
    Uses a circulant base + greedy completion.
    """
    adj = [0] * N
    
    # Start with circulant C(20, {1,2,3}) - 30 edges
    for i in range(N):
        for d in [1, 2, 3]:
            j = (i + d) % N
            if i < j:
                if _can_add_edge_fast(adj, i, j):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
    
    # Try adding more edges greedily
    for i in range(N):
        for j in range(i + 1, N):
            if (adj[i] >> j) & 1:
                continue
            if _can_add_edge_fast(adj, i, j):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    return adj

def _algebraic_start():
    """
    Try a construction based on quadratic residues or similar algebraic structure.
    For n=20, try connecting i-j mod 20 where difference is in a set S.
    """
    adj = [0] * N
    
    # Try different difference sets
    diff_sets = [
        [1, 2, 3, 5, 7],
        [1, 2, 3, 4, 7],
        [1, 2, 3, 5, 8],
        [1, 2, 4, 5, 7],
        [1, 3, 4, 5, 7],
        [1, 2, 3, 6, 7],
    ]
    
    best_adj = [0] * N
    best_count = 0
    
    for ds in diff_sets:
        adj = [0] * N
        for i in range(N):
            for d in ds:
                j = (i + d) % N
                if i < j:
                    if _can_add_edge_fast(adj, i, j):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        # Greedy fill
        for i in range(N):
            for j in range(i + 1, N):
                if (adj[i] >> j) & 1:
                    continue
                if _can_add_edge_fast(adj, i, j):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        count = _edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    return best_adj

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_adj = [0] * N
    best_count = 0
    
    # Strategy 1: Structured starts
    for start_func in [_structured_start, _algebraic_start]:
        if time.time() > deadline - 20:
            break
        adj = start_func()
        adj = _greedy_fill(adj, 3.0)
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count
    
    # Strategy 2: Random starts with various densities
    densities = [0.7, 0.8, 0.6, 0.5, 0.75, 0.65, 0.85, 0.55]
    random.shuffle(densities)
    
    for density in densities:
        if time.time() > deadline - 15:
            break
            
        adj = _random_start(density)
        
        # Greedy fill
        remaining = deadline - time.time() - 5
        if remaining < 1:
            break
        adj = _greedy_fill(adj, min(remaining * 0.25, 2.5))
        
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count
        
        # Fast local search
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
        adj = _fast_local_search(adj, min(remaining * 0.6, 6.0))
        
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count
            
    # If we still have time, do more local search on the best found
    remaining = deadline - time.time() - 1
    if remaining > 2:
        adj = best_adj[:]
        adj = _fast_local_search(adj, remaining)
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count

    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
