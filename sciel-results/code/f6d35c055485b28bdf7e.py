"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_N = 16
_E = _N * (_N - 1) // 2  # 120 edges
_DEADLINE_MARGIN = 20  # seconds to spare before the 150s kill

# Precompute Q3 structure
_Q3_EDGES = []
for i in range(8):
    for j in range(i + 1, 8):
        if (i ^ j) in (1, 2, 4):
            _Q3_EDGES.append((i, j))

_Q3_NEIGHBORS = [[] for _ in range(8)]
for i in range(8):
    for bit in (1, 2, 4):
        j = i ^ bit
        if j > i:
            _Q3_NEIGHBORS[i].append(j)
            _Q3_NEIGHBORS[j].append(i)

# Precompute: for each qv, bitmask of Q3 neighbors
_Q3_QV_NEIGHBOR_MASK = [0] * 8
for i in range(8):
    mask = 0
    for j in _Q3_NEIGHBORS[i]:
        mask |= (1 << j)
    _Q3_QV_NEIGHBOR_MASK[i] = mask

# Precompute: for each qv, list of Q3 neighbors that come before it in the backtracking order
# Backtracking order: 0, 1, 2, 3, 4, 5, 6, 7
_Q3_PRIOR_NEIGHBORS = [[] for _ in range(8)]
for i in range(8):
    for j in _Q3_NEIGHBORS[i]:
        if j < i:
            _Q3_PRIOR_NEIGHBORS[i].append(j)


def _find_q3_fast(adj_bits):
    """
    Finds a Q3 subgraph if one exists.
    Returns a list of (u, v) edges in the graph corresponding to the Q3 edges, or None.
    adj_bits: list of 16 integers, each a 16-bit mask of neighbors.
    """
    assigned = [-1] * 8
    used_mask = 0
    full_mask = (1 << _N) - 1
    
    def backtrack(qv):
        nonlocal used_mask
        if qv == 8:
            return True
        
        candidates = (~used_mask) & full_mask
        
        # Filter: must be adjacent to all already-assigned Q3 neighbors of qv
        for pv in _Q3_PRIOR_NEIGHBORS[qv]:
            pgv = assigned[pv]
            candidates &= adj_bits[pgv]
        
        if candidates == 0:
            return False
        
        c = candidates
        while c:
            lsb = c & (-c)
            gv = lsb.bit_length() - 1
            c ^= lsb
            
            assigned[qv] = gv
            used_mask |= (1 << gv)
            if backtrack(qv + 1):
                return True
            used_mask &= ~(1 << gv)
            assigned[qv] = -1
        
        return False
    
    for gv0 in range(_N):
        assigned[0] = gv0
        used_mask = 1 << gv0
        if backtrack(1):
            # Reconstruct edges
            edges = []
            for i in range(8):
                for j in _Q3_NEIGHBORS[i]:
                    if i < j:
                        edges.append((assigned[i], assigned[j]))
            return edges
        used_mask = 0
        assigned[0] = -1
    
    return None


def _has_q3_with_edge(adj_bits, u, v):
    """
    Check if adding edge (u,v) to the current graph creates a Q3.
    Only checks Q3 embeddings that include edge (u,v).
    """
    # Q3 vertices: 0=000, 1=001, 2=010, 3=011, 4=100, 5=101, 6=110, 7=111
    # Fix edge (u,v) as Q3 edge (0,1)
    # Remaining: 2,3,4,5,6,7
    # 
    # Constraints:
    # 2: adj to u, 3, 6
    # 3: adj to v, 2, 7
    # 4: adj to u, 5, 6
    # 5: adj to v, 4, 7
    # 6: adj to 2, 4, 7
    # 7: adj to 3, 5, 6
    
    order = [2, 4, 3, 5, 6, 7]
    assigned = {}
    used_mask = (1 << u) | (1 << v)
    full_mask = (1 << _N) - 1
    
    # For each qv in order, which previously-assigned qvs it must connect to, and external constraint
    constraints = {
        2: ((), u),      # no prior q3 neighbors, must connect to u
        4: ((), u),      # no prior q3 neighbors, must connect to u
        3: ((2,), v),    # must connect to qv2 and v
        5: ((4,), v),    # must connect to qv4 and v
        6: ((2, 4), None),  # must connect to qv2 and qv4
        7: ((3, 5), None),  # must connect to qv3 and qv5
    }
    
    def backtrack(idx):
        nonlocal used_mask
        if idx == len(order):
            # Final check: edge 6-7
            return (adj_bits[assigned[6]] >> assigned[7]) & 1
        
        qv = order[idx]
        prior_qvs, ext = constraints[qv]
        
        candidates = (~used_mask) & full_mask
        
        if ext is not None:
            candidates &= adj_bits[ext]
        
        for nqv in prior_qvs:
            candidates &= adj_bits[assigned[nqv]]
        
        if candidates == 0:
            return False
        
        c = candidates
        while c:
            lsb = c & (-c)
            gv = lsb.bit_length() - 1
            c ^= lsb
            
            assigned[qv] = gv
            used_mask |= (1 << gv)
            if backtrack(idx + 1):
                return True
            used_mask &= ~(1 << gv)
            del assigned[qv]
        
        return False
    
    return backtrack(0)


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_MARGIN
    
    all_edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    
    best_edges = []
    best_count = 0
    best_adj = [0] * _N
    
    def update_best(edges, adj):
        nonlocal best_edges, best_count, best_adj
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            best_adj = adj[:]
            
    def local_search(edges, adj, time_limit):
        """
        Improve the graph by adding missing edges and swapping.
        """
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
        missing = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
        random.shuffle(missing)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add edges
            for i in range(len(missing)):
                if time.time() > time_limit - 1:
                    break
                u, v = missing[i]
                if (u, v) in edge_set:
                    continue
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if not _has_q3_with_edge(adj, u, v):
                    edges.append([u, v])
                    edge_set.add((u, v))
                    missing.pop(i)
                    i -= 1
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
            # Try swaps: remove 1, add 2
            if time.time() < time_limit - 2:
                for e_idx in range(len(edges)):
                    if time.time() > time_limit - 1:
                        break
                    u_rem, v_rem = edges[e_idx]
                    
                    # Temporarily remove
                    adj[u_rem] &= ~(1 << v_rem)
                    adj[v_rem] &= ~(1 << u_rem)
                    edge_set.discard((u_rem, v_rem))
                    
                    added = []
                    for u_add, v_add in missing:
                        if (u_add, v_add) in edge_set:
                            continue
                        if (u_add, v_add) == (u_rem, v_rem):
                            continue
                        
                        adj[u_add] |= (1 << v_add)
                        adj[v_add] |= (1 << u_add)
                        
                        if not _has_q3_with_edge(adj, u_add, v_add):
                            added.append((u_add, v_add))
                            edge_set.add((u_add, v_add))
                            # Remove from missing list for consistency? 
                            # We'll rebuild missing if we accept
                        else:
                            adj[u_add] &= ~(1 << v_add)
                            adj[v_add] &= ~(1 << u_add)
                        
                        if len(added) == 2:
                            break
                    
                    if len(added) == 2:
                        # Accept swap
                        edges.pop(e_idx)
                        for u_a, v_a in added:
                            edges.append([u_a, v_a])
                        # Rebuild missing
                        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
                        missing = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
                        random.shuffle(missing)
                        improved = True
                        break
                    else:
                        # Revert
                        adj[u_rem] |= (1 << v_rem)
                        adj[v_rem] |= (1 << u_rem)
                        edge_set.add((u_rem, v_rem))
                        # Revert added edges
                        for u_a, v_a in added:
                            adj[u_a] &= ~(1 << v_a)
                            adj[v_a] &= ~(1 << u_a)
                            edge_set.discard((u_a, v_a))
                        
    # Phase 1: Greedy edge addition with multiple orderings
    num_trials = 10
    
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        if trial == 0:
            edge_order = all_edges[:]
        elif trial == 1:
            edge_order = all_edges[::-1]
        else:
            random.seed(trial * 12345 + 67890)
            edge_order = all_edges[:]
            random.shuffle(edge_order)
        
        adj = [0] * _N
        edges = []
        
        for u, v in edge_order:
            if time.time() > deadline - 5:
                break
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not _has_q3_with_edge(adj, u, v):
                edges.append([u, v])
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        update_best(edges, adj)
        
        if time.time() < deadline - 15:
            local_search(edges, adj, deadline - 10)
            update_best(edges, adj)

    # Phase 2: Greedy removal from K16
    if time.time() < deadline - 20:
        adj = [0] * _N
        for u in range(_N):
            for v in range(u + 1, _N):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        edges = all_edges[:]
        
        # Remove edges to make Q3-free
        # Limit iterations to avoid infinite loop if stuck
        max_iter = 50
        for _ in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            q3_edges = _find_q3_fast(adj)
            if not q3_edges:
                break
            
            # Remove one edge from the Q3
            # Choose edge with highest degree sum? Or random?
            # Random is faster
            u_rem, v_rem = random.choice(q3_edges)
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            edges = [[u, v] for u, v in edges if not (u == u_rem and v == v_rem) and not (u == v_rem and v == u_rem)]
            
        update_best(edges, adj)
        
        if time.time() < deadline - 10:
            local_search(edges, adj, deadline - 5)
            update_best(edges, adj)

    # Phase 3: Final polish
    if time.time() < deadline - 5:
        adj = best_adj[:]
        edges = best_edges[:]
        local_search(edges, adj, deadline - 2)
        update_best(edges, adj)
    
    return best_edges
# EVOLVE-BLOCK-END
