import numpy as np
import time
import random
import math

def count_c4(adj):
    """
    Count the number of 4-cycles in the graph represented by adjacency matrix adj.
    Uses the formula: C4 = (1/2) * sum_{u<v} C(common_neighbors(u,v), 2)
    where common_neighbors(u,v) = (A^2)_{uv} for u != v.
    """
    n = adj.shape[0]
    A2 = adj @ adj
    # For each pair u < v, count common neighbors, then C(common, 2)
    # A2[u,v] gives number of length-2 paths from u to v = common neighbors
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            cn = A2[u, v]
            if cn >= 2:
                total += cn * (cn - 1)
    return total // 2

def count_c4_delta(adj, edge_remove, edge_add):
    """
    Compute the change in C4 count when removing edge_remove and adding edge_add.
    edge_remove = (u1, v1), edge_add = (u2, v2)
    Assumes edge_remove is currently in graph and edge_add is not.
    """
    # Quick full recount is fast enough for n=31
    # But let's try incremental for speed
    u1, v1 = edge_remove
    u2, v2 = edge_add
    
    # Remove edge
    adj[u1, v1] = 0
    adj[v1, u1] = 0
    c4_after_remove = count_c4(adj)
    
    # Add edge
    adj[u2, v2] = 1
    adj[v2, u2] = 1
    c4_after_add = count_c4(adj)
    
    # Restore
    adj[u1, v1] = 1
    adj[v1, u1] = 1
    adj[u2, v2] = 0
    adj[v2, u2] = 0
    
    return c4_after_add - c4_after_remove

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # 1. Construct the base graph: Orthogonal Polarity Graph of PG(2,5)
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
    current_c4 = count_c4(adj)
    
    # 2. Greedy addition of remaining edges with minimum C4 increase
    def count_new_c4s(u, v, adj):
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        count = 0
        if len(nu) < len(nv):
            for w in nu:
                count += np.sum(adj[w, nv])
        else:
            for x in nv:
                count += np.sum(adj[x, nu])
        return count

    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
    
    edges_to_add = m - current_edges
    for _ in range(edges_to_add):
        if time.time() > deadline:
            break
        best_edge = None
        min_c4s = float('inf')
        
        for u, v in all_possible_edges:
            c4s = count_new_c4s(u, v, adj)
            if c4s < min_c4s:
                min_c4s = c4s
                best_edge = (u, v)
        
        if best_edge is not None:
            u, v = best_edge
            adj[u, v] = 1
            adj[v, u] = 1
            edge_set.add(best_edge)
            all_possible_edges.remove(best_edge)
    
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_edge_set = set(edge_set)
    
    # 3. Local Search with Simulated Annealing
    T = 3.0
    T_min = 0.01
    cooling_rate = 0.998
    
    current_edges_list = list(edge_set)
    
    while T > T_min and time.time() < deadline:
        # Pick a random edge to remove
        idx_rm = random.randint(0, len(current_edges_list) - 1)
        e_rm = current_edges_list[idx_rm]
        u1, v1 = e_rm
        
        # Pick a random non-edge to add
        attempts = 0
        while attempts < 100:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 != v2:
                if u2 > v2:
                    u2, v2 = v2, u2
                e_add = (u2, v2)
                if e_add not in edge_set:
                    break
            attempts += 1
        else:
            continue
        
        # Apply swap
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        
        new_c4 = count_c4(adj)
        delta = new_c4 - current_c4
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_c4 = new_c4
            edge_set.remove(e_rm)
            edge_set.add(e_add)
            current_edges_list[idx_rm] = e_add
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edge_set = set(edge_set)
        else:
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0
        
        T *= cooling_rate
    
    # 4. Final hill climbing pass
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current_edges_list)):
                if time.time() > deadline:
                    break
                e_rm = current_edges_list[i]
                u1, v1 = e_rm
                # Try swapping with a few random non-edges
                for _ in range(5):
                    u2 = random.randint(0, n-1)
                    v2 = random.randint(0, n-1)
                    if u2 != v2:
                        if u2 > v2:
                            u2, v2 = v2, u2
                        e_add = (u2, v2)
                        if e_add not in edge_set:
                            adj[u1, v1] = 0
                            adj[v1, u1] = 0
                            adj[u2, v2] = 1
                            adj[v2, u2] = 1
                            new_c4 = count_c4(adj)
                            if new_c4 < best_c4:
                                current_c4 = new_c4
                                best_c4 = new_c4
                                best_edge_set = set(edge_set)
                                edge_set.remove(e_rm)
                                edge_set.add(e_add)
                                current_edges_list[i] = e_add
                                improved = True
                            else:
                                adj[u1, v1] = 1
                                adj[v1, u1] = 1
                                adj[u2, v2] = 0
                                adj[v2, u2] = 0
    
    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
