import numpy as np
import time
import random
import math

def count_c4(adj):
    """
    Count the number of 4-cycles in the graph represented by adjacency matrix adj.
    """
    n = adj.shape[0]
    A2 = adj @ adj
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = A2[mask].astype(np.int64)
    return int(np.sum(vals * (vals - 1)) // 2)

def count_c4_fast(adj):
    """
    Fast C4 count using only the upper triangle of A^2.
    """
    n = adj.shape[0]
    A2 = adj @ adj
    vals = A2[np.triu_indices(n, k=1)].astype(np.int64)
    return int(np.sum(vals * (vals - 1)) // 2)

def c4_delta_for_swap(adj, e_rm, e_add):
    """
    Compute the change in C4 count when removing edge e_rm and adding edge e_add.
    
    C4s involving edge (u,v): number of length-3 paths from u to v not using (u,v).
    When we remove (u1,v1), we lose C4s that use (u1,v1).
    When we add (u2,v2), we gain C4s that use (u2,v2).
    
    But C4s can use both edges if they're in the same 4-cycle, so we need to be careful.
    For a swap where e_rm and e_add are disjoint or share at most one vertex, 
    the delta is approximately:
      delta = -c4s_using(e_rm) + c4s_using(e_add)
    where c4s_using(e) = number of common neighbors' neighbors pairs.
    
    More precisely: a 4-cycle containing edge (u,v) corresponds to a path u-w-x-v 
    where w in N(u)\\{v}, x in N(v)\\{u}, and (w,x) in E.
    The number of such paths = sum over w in N(u)\\{v} of |N(w) ∩ N(v)\\{u}|
    But we must exclude cases where w=x (which would be a triangle, not a 4-cycle)
    and where the path uses the edge (u,v) itself.
    
    For a non-edge (u,v), the number of 4-cycles created by adding (u,v) is:
      sum_{w in N(u)} sum_{x in N(v)} [w != x and (w,x) in E and w != v and x != u]
    Since (u,v) is not an edge, w != v and x != u are automatic if we define
    N(u) as the current neighbors.
    
    For an existing edge (u,v), the number of 4-cycles that use (u,v) is:
      sum_{w in N(u)\\{v}} sum_{x in N(v)\\{u}} [w != x and (w,x) in E]
    """
    u1, v1 = e_rm
    u2, v2 = e_add
    
    # Count C4s using edge (u1, v1) in the current graph
    # These are lost when we remove (u1, v1)
    lost = 0
    nu1 = np.where(adj[u1] == 1)[0]
    nv1 = np.where(adj[v1] == 1)[0]
    # Exclude v1 from nu1 and u1 from nv1
    nu1_mask = nu1 != v1
    nv1_mask = nv1 != u1
    nu1_f = nu1[nu1_mask]
    nv1_f = nv1[nv1_mask]
    
    if len(nu1_f) < len(nv1_f):
        for w in nu1_f:
            # Count neighbors of w that are in nv1_f, excluding w itself (but w is not in nv1_f unless w==v1 which is excluded)
            common = np.sum(adj[w, nv1_f])
            # Subtract 1 if w is in nv1_f (i.e., w is adjacent to v1), but w != v1 is already ensured
            # Actually, adj[w, nv1_f] counts edges from w to each x in nv1_f
            # We need w != x, so if w is in nv1_f, we count an extra 1 (the self-loop which isn't there)
            # adj[w, w] = 0, so no issue. But if w is in nv1_f, then (w, v1) is an edge, 
            # and we're counting the pair (w, w) which shouldn't exist.
            # Actually adj[w, nv1_f] sums adj[w, x] for x in nv1_f. If w is in nv1_f, 
            # then one term is adj[w, w] = 0. So no overcounting.
            lost += common
    else:
        for x in nv1_f:
            lost += np.sum(adj[x, nu1_f])
    
    # Count C4s that would be created by adding edge (u2, v2)
    # In the graph WITHOUT (u1,v1) but WITH (u2,v2)
    # For simplicity, compute in the current graph (with (u1,v1) present, without (u2,v2))
    # The C4s created by adding (u2,v2) use paths u2-w-x-v2 where w in N(u2), x in N(v2), (w,x) in E
    # We need to be careful: if the path uses edge (u1,v1), it won't exist after removal.
    # For a first-order approximation, compute in current graph.
    gained = 0
    nu2 = np.where(adj[u2] == 1)[0]
    nv2 = np.where(adj[v2] == 1)[0]
    # u2 and v2 are not adjacent (it's a non-edge), so no need to exclude
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
    deadline = start_time + 135  # 15s margin from 150s limit

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

    # Phase 1: Exhaustive 1-edge-swap local search with incremental C4
    current_c4 = count_c4_fast(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edge_set = set(edge_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges_list = list(edge_set)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        
        for e_rm in current_edges_list:
            if time.time() > deadline:
                break
            for e_add in non_edges:
                # Use incremental delta computation
                delta = c4_delta_for_swap(adj, e_rm, e_add)
                new_c4 = current_c4 + delta
                
                if new_c4 < best_c4:
                    # Verify with exact count
                    u1, v1 = e_rm
                    u2, v2 = e_add
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1
                    
                    exact_c4 = count_c4_fast(adj)
                    
                    if exact_c4 < best_c4:
                        best_c4 = exact_c4
                        best_adj = adj.copy()
                        best_edge_set = set(edge_set)
                        edge_set.remove(e_rm)
                        edge_set.add(e_add)
                        current_c4 = exact_c4
                        improved = True
                        adj = best_adj.copy()
                        break
                    else:
                        # Reject
                        adj[u1, v1] = 1
                        adj[v1, u1] = 1
                        adj[u2, v2] = 0
                        adj[v2, u2] = 0

    # Phase 2: Simulated annealing for further refinement
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    current_edges_list = list(edge_set)
    current_c4 = best_c4
    adj = best_adj.copy()
    edge_set = set(best_edge_set)
    
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
        
        # Apply swap
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        
        new_c4 = count_c4_fast(adj)
        delta = new_c4 - current_c4
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_c4 = new_c4
            edge_set.remove(e_rm)
            edge_set.add(e_add)
            current_edges_list[idx_rm] = e_add
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
        else:
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            
        T *= cooling_rate

    # Phase 3: Final exhaustive local search pass
    adj = best_adj.copy()
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
            for e_add in non_edges:
                u1, v1 = e_rm
                u2, v2 = e_add
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                
                new_c4 = count_c4_fast(adj)
                
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_adj = adj.copy()
                    best_edge_set = set(edge_set)
                    edge_set.remove(e_rm)
                    edge_set.add(e_add)
                    current_c4 = new_c4
                    improved = True
                    adj = best_adj.copy()
                    break
                else:
                    adj[u1, v1] = 1
                    adj[v1, u1] = 1
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0

    # Return the best found
    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
