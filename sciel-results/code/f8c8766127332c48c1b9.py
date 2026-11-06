import numpy as np
import time
import random
import math

def count_c4(adj):
    n = adj.shape[0]
    A2 = adj @ adj
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = A2[mask].astype(np.int64)
    return int(np.sum(vals * (vals - 1)) // 2)

def count_c4_fast(adj):
    n = adj.shape[0]
    A2 = adj @ adj
    vals = A2[np.triu_indices(n, k=1)].astype(np.int64)
    return int(np.sum(vals * (vals - 1)) // 2)

def c4_delta_for_swap(adj, e_rm, e_add):
    u1, v1 = e_rm
    u2, v2 = e_add
    lost = 0
    nu1 = np.where(adj[u1] == 1)[0]
    nv1 = np.where(adj[v1] == 1)[0]
    nu1_mask = nu1 != v1
    nv1_mask = nv1 != u1
    nu1_f = nu1[nu1_mask]
    nv1_f = nv1[nv1_mask]
    if len(nu1_f) < len(nv1_f):
        for w in nu1_f:
            lost += np.sum(adj[w, nv1_f])
    else:
        for x in nv1_f:
            lost += np.sum(adj[x, nu1_f])
    gained = 0
    nu2 = np.where(adj[u2] == 1)[0]
    nv2 = np.where(adj[v2] == 1)[0]
    if len(nu2) < len(nv2):
        for w in nu2:
            gained += np.sum(adj[w, nv2])
    else:
        for x in nv2:
            gained += np.sum(adj[x, nu2])
    return gained - lost

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 145

    # Build base graph: Orthogonal Polarity Graph of PG(2,5)
    points = []
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    for z in range(5):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    adj = np.zeros((n, n), dtype=np.int8)
    edge_set = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q = points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edge_set.add((i, j))
    
    current_edges = len(edge_set)
    edges_to_add = m - current_edges
    
    if edges_to_add > 0:
        all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        min_c4s = float('inf')
        best_edge = None
        for u, v in all_possible_edges:
            nu = np.where(adj[u] == 1)[0]
            nv = np.where(adj[v] == 1)[0]
            count = 0
            if len(nu) < len(nv):
                for w in nu:
                    count += np.sum(adj[w, nv])
            else:
                for x in nv:
                    count += np.sum(adj[x, nu])
            if count < min_c4s:
                min_c4s = count
                best_edge = (u, v)
        if best_edge is not None:
            u, v = best_edge
            adj[u, v] = 1
            adj[v, u] = 1
            edge_set.add(best_edge)
            added = 1
            while added < edges_to_add and time.time() < deadline:
                all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
                min_c4s_iter = float('inf')
                best_edge_iter = None
                for u, v in all_possible_edges:
                    nu = np.where(adj[u] == 1)[0]
                    nv = np.where(adj[v] == 1)[0]
                    count = 0
                    if len(nu) < len(nv):
                        for w in nu:
                            count += np.sum(adj[w, nv])
                    else:
                        for x in nv:
                            count += np.sum(adj[x, nu])
                    if count < min_c4s_iter:
                        min_c4s_iter = count
                        best_edge_iter = (u, v)
                if best_edge_iter:
                    u, v = best_edge_iter
                    adj[u, v] = 1
                    adj[v, u] = 1
                    edge_set.add(best_edge_iter)
                    added += 1
                else:
                    break

    # Convert to bitmask representation for fast delta computation
    # adj_bits[i] is a bitmask where bit j is set iff (i,j) is an edge
    adj_bits = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i, j]:
                mask |= (1 << j)
        adj_bits[i] = mask
    
    # Precompute popcount lookup for 16-bit chunks
    popcount_table = [0] * 65536
    for i in range(1, 65536):
        popcount_table[i] = popcount_table[i >> 1] + (i & 1)
    
    def popcount(x):
        # Count bits in a 31-bit integer
        return popcount_table[x & 0xFFFF] + popcount_table[(x >> 16) & 0xFFFF]
    
    # Compute initial C4 count using the adjacency matrix
    current_c4 = count_c4_fast(adj)
    best_c4 = current_c4
    best_adj_bits = adj_bits[:]
    best_edge_set = set(edge_set)
    
    # Helper: count C4s that use a specific edge (u,v) in the current bitmask graph
    def c4s_using_edge(u, v, bits):
        # A C4 using edge (u,v): u-w-x-v where w in N(u)\{v}, x in N(v)\{u}, (w,x) in E
        # Count = sum_{w in N(u)\{v}} |N(w) ∩ N(v)\{u}|
        # But we need to be careful: the path u-w-x-v forms a C4 iff w != x and (w,x) is an edge
        # Since we're counting pairs (w,x) with w in N(u)\{v}, x in N(v)\{u}, (w,x) in E, w != x
        # adj[w,x]=1 and w != x is automatic since no self-loops
        nu = bits[u] & ~(1 << v)
        nv = bits[v] & ~(1 << u)
        count = 0
        # Iterate over set bits in nu
        w = nu
        while w:
            lowbit = w & (-w)
            wi = lowbit.bit_length() - 1
            w ^= lowbit
            # Count common neighbors of wi and v, excluding u
            common = bits[wi] & nv
            # Exclude u from common (but u shouldn't be in nv since (u,v) is not in N(v)\{u}... wait)
            # Actually nv = N(v) \ {u}, so u is already excluded
            count += popcount(common)
        return count
    
    # Helper: count C4s that would be created by adding edge (u,v) to the current graph
    def c4s_created_by_edge(u, v, bits):
        # Adding (u,v) creates C4s of form u-w-x-v where w in N(u), x in N(v), (w,x) in E, w != x
        # Since (u,v) is not currently an edge, w != v and x != u are automatic
        nu = bits[u]
        nv = bits[v]
        count = 0
        w = nu
        while w:
            lowbit = w & (-w)
            wi = lowbit.bit_length() - 1
            w ^= lowbit
            common = bits[wi] & nv
            count += popcount(common)
        return count
    
    # Phase 1: Exhaustive 1-edge-swap local search with bitmask delta
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges_list = list(edge_set)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        
        for e_rm in current_edges_list:
            if time.time() > deadline:
                break
            u1, v1 = e_rm
            lost = c4s_using_edge(u1, v1, adj_bits)
            for e_add in non_edges:
                u2, v2 = e_add
                gained = c4s_created_by_edge(u2, v2, adj_bits)
                delta = gained - lost
                if delta < 0:
                    # Apply the swap
                    adj_bits[u1] &= ~(1 << v1)
                    adj_bits[v1] &= ~(1 << u1)
                    adj_bits[u2] |= (1 << v2)
                    adj_bits[v2] |= (1 << u2)
                    edge_set.remove(e_rm)
                    edge_set.add(e_add)
                    current_c4 += delta
                    if current_c4 < best_c4:
                        best_c4 = current_c4
                        best_adj_bits = adj_bits[:]
                        best_edge_set = set(edge_set)
                    improved = True
                    break
    
    # Phase 2: Simulated annealing with bitmask representation
    adj_bits = best_adj_bits[:]
    edge_set = set(best_edge_set)
    current_c4 = best_c4
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    current_edges_list = list(edge_set)
    steps = 0
    while T > T_min and time.time() < deadline:
        steps += 1
        idx_rm = random.randint(0, len(current_edges_list) - 1)
        e_rm = current_edges_list[idx_rm]
        u1, v1 = e_rm
        
        while True:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            e_add = (u2, v2)
            if e_add not in edge_set:
                break
        
        # Compute delta using bitmask
        lost = c4s_using_edge(u1, v1, adj_bits)
        gained = c4s_created_by_edge(u2, v2, adj_bits)
        delta = gained - lost
        
        # Apply swap temporarily
        adj_bits[u1] &= ~(1 << v1)
        adj_bits[v1] &= ~(1 << u1)
        adj_bits[u2] |= (1 << v2)
        adj_bits[v2] |= (1 << u2)
        
        new_c4 = current_c4 + delta
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_c4 = new_c4
            edge_set.remove(e_rm)
            edge_set.add(e_add)
            current_edges_list[idx_rm] = e_add
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj_bits = adj_bits[:]
                best_edge_set = set(edge_set)
        else:
            # Revert
            adj_bits[u1] |= (1 << v1)
            adj_bits[v1] |= (1 << u1)
            adj_bits[u2] &= ~(1 << v2)
            adj_bits[v2] &= ~(1 << u2)
        
        T *= cooling_rate
    
    # Phase 3: Final exhaustive local search pass with bitmask
    adj_bits = best_adj_bits[:]
    edge_set = set(best_edge_set)
    current_c4 = best_c4
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges_list = list(edge_set)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        
        for e_rm in current_edges_list:
            if time.time() > deadline:
                break
            u1, v1 = e_rm
            lost = c4s_using_edge(u1, v1, adj_bits)
            for e_add in non_edges:
                u2, v2 = e_add
                gained = c4s_created_by_edge(u2, v2, adj_bits)
                delta = gained - lost
                if delta < 0:
                    adj_bits[u1] &= ~(1 << v1)
                    adj_bits[v1] &= ~(1 << u1)
                    adj_bits[u2] |= (1 << v2)
                    adj_bits[v2] |= (1 << u2)
                    edge_set.remove(e_rm)
                    edge_set.add(e_add)
                    current_c4 += delta
                    if current_c4 < best_c4:
                        best_c4 = current_c4
                        best_adj_bits = adj_bits[:]
                        best_edge_set = set(edge_set)
                    improved = True
                    break
    
    # Return the best found
    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
