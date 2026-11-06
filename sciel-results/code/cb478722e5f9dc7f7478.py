import random
import time
import sys
import math
from itertools import combinations

def _paley_graph(q):
    """Return adjacency bitmasks for Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _popcount(x):
    return bin(x).count("1")

def _count_triangles_mask(adj, mask):
    """Count triangles in subgraph induced by vertices in mask."""
    # Get list of vertices in mask
    verts = []
    m = mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        verts.append(i)
        m ^= lsb
        
    count = 0
    n_v = len(verts)
    for i in range(n_v):
        u = verts[i]
        # Neighbors of u within mask
        neigh_u_in_mask = adj[u] & mask
        # Iterate over neighbors v > u in verts? 
        # Simpler: iterate all pairs in verts
        for j in range(i + 1, n_v):
            v = verts[j]
            if not (neigh_u_in_mask & (1 << v)):
                continue
            # Count common neighbors of u and v within mask
            common = adj[u] & adj[v] & mask
            # We only want to count each triangle once.
            # If we sum popcount(common) for all edges (u,v), each triangle is counted 3 times.
            # So we can sum and divide by 3 at the end.
            count += _popcount(common)
    return count // 3

def _count_ind_triangles_mask(adj, mask):
    """Count independent triangles in subgraph induced by vertices in mask."""
    # Independent triangle in G is triangle in G_complement.
    # Complement adjacency: ~adj[i] (masked to n bits)
    # Let's create complement adj on the fly?
    # Or just check non-edges.
    # For a set of 3 vertices to be independent, there must be no edges between them.
    # This is equivalent to counting triangles in the complement graph.
    
    # Let's build complement masks for the vertices in mask
    # Actually, we can just use the same logic on complement adj.
    # comp_adj[i] = (~adj[i]) & full_mask
    
    verts = []
    m = mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        verts.append(i)
        m ^= lsb
        
    count = 0
    n_v = len(verts)
    full_mask = (1 << len(adj)) - 1
    
    for i in range(n_v):
        u = verts[i]
        # Non-neighbors of u within mask
        # adj[u] contains neighbors. ~adj[u] contains non-neighbors.
        # Mask to n bits: full_mask ^ adj[u]
        non_neigh_u_in_mask = (full_mask ^ adj[u]) & mask
        for j in range(i + 1, n_v):
            v = verts[j]
            # Check if v is a non-neighbor of u
            if (non_neigh_u_in_mask & (1 << v)) == 0:
                continue
            # Count common non-neighbors of u and v within mask
            # non_neigh_v_in_mask = (full_mask ^ adj[v]) & mask
            common_non = (full_mask ^ adj[u]) & (full_mask ^ adj[v]) & mask
            count += _popcount(common_non)
    return count // 3

def _calc_delta_cost_mask(adj, u, v, add_edge):
    """
    Calculate change in cost if we flip edge (u, v) using bitmasks.
    """
    n = len(adj)
    full_mask = (1 << n) - 1
    
    # Common neighbors
    common_neigh = adj[u] & adj[v]
    # Common non-neighbors
    common_non_neigh = (full_mask ^ adj[u]) & (full_mask ^ adj[v]) & ~(1 << u) & ~(1 << v)
    
    tri_neigh = _count_triangles_mask(adj, common_neigh)
    ind_tri_non_neigh = _count_ind_triangles_mask(adj, common_non_neigh)
    
    if add_edge:
        return tri_neigh - ind_tri_non_neigh
    else:
        return -tri_neigh + ind_tri_non_neigh

def _total_cost_mask(adj):
    """Calculate total cost using bitmasks."""
    n = len(adj)
    cost = 0
    full_mask = (1 << n) - 1
    
    # Count K5s
    # For each vertex v, count K4s in N(v).
    # Sum over v, divide by 5.
    k5_count = 0
    for v in range(n):
        mask_neigh = adj[v]
        # Count K4s in mask_neigh
        # This is still O(k^4) or similar.
        # Let's just use the combinations method for initial cost, it's only done once.
        pass
        
    # Revert to combinations for initial cost as it's simpler and only done once.
    # Convert adj back to sets for this? Or just use bit checks.
    for comb in combinations(range(n), 5):
        is_k5 = True
        for i in range(5):
            for j in range(i+1, 5):
                if not (adj[comb[i]] & (1 << comb[j])):
                    is_k5 = False
                    break
            if not is_k5:
                break
        if is_k5:
            cost += 1
            
        is_ind = True
        for i in range(5):
            for j in range(i+1, 5):
                if (adj[comb[i]] & (1 << comb[j])):
                    is_ind = False
                    break
            if not is_ind:
                break
        if is_ind:
            cost += 1
            
    return cost

def _run_sa_mask(adj, target_n, time_limit, start_temp=1.0, end_temp=0.1):
    n = len(adj)
    current_cost = _total_cost_mask(adj)
    best_adj = list(adj)
    best_cost = current_cost
    
    start_time = time.time()
    t = start_temp
    
    while time.time() - start_time < time_limit:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        add_edge = not (adj[u] & (1 << v))
        
        delta = _calc_delta_cost_mask(adj, u, v, add_edge)
        
        if delta < 0:
            if add_edge:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj = list(adj)
                if best_cost == 0:
                    return best_adj, 0
        else:
            prob = math.exp(-delta / t)
            if random.random() < prob:
                if add_edge:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                current_cost += delta
        
        t = max(t * 0.999, end_temp)
        
    return best_adj, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 310
    
    for target_n in [42, 41, 40, 39, 38]:
        if time.time() > deadline - 10:
            break
            
        base_n = 37
        adj = _paley_graph(base_n)
        
        # Extend to target_n
        for i in range(base_n, target_n):
            adj.append(0)
            for j in range(base_n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            for j in range(base_n, i):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        remaining_time = deadline - time.time()
        sa_time = min(60.0, remaining_time - 5)
        if sa_time <= 0:
            continue
            
        best_adj, best_cost = _run_sa_mask(adj, target_n, sa_time)
        
        if best_cost == 0:
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if best_adj[i] & (1 << j):
                        edges.append((i, j))
            return target_n, edges
            
    # Fallback
    adj = _paley_graph(37)
    n = 37
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append((i, j))
    return n, edges
