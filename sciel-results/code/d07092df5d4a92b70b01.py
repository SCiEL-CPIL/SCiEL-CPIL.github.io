"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_N = 16
_E = _N * (_N - 1) // 2  # 120 edges
_DEADLINE = 140  # seconds, 10s safety margin before 150s kill

# Precompute Q3 structure
_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# Precompute Q3 edge orientations with better ordering
_Q3_EDGE_ORIENTS = []
for qa, qb in _Q3_EDGES:
    for (gu_q, gv_q) in [(qa, qb), (qb, qa)]:
        unassigned = [i for i in range(8) if i != gu_q and i != gv_q]
        def _cc(x, a, b):
            c = 0
            if _Q3_ADJ[x] & (1 << a): c += 1
            if _Q3_ADJ[x] & (1 << b): c += 1
            return c
        unassigned.sort(key=lambda x: _cc(x, gu_q, gv_q), reverse=True)
        _Q3_EDGE_ORIENTS.append((gu_q, gv_q, tuple(unassigned)))

def _count_edges(adj):
    return sum(bin(adj[i]).count('1') for i in range(_N)) // 2

def _edge_list(adj):
    es = []
    for u in range(_N):
        for v in range(u+1, _N):
            if (adj[u] >> v) & 1:
                es.append([u, v])
    return es

def _has_q3_after_add(adj, u, v, deadline):
    """Check if adding edge (u,v) to current adj creates a Q3 subgraph."""
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

    for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
        for (gu, gv) in [(u, v), (v, u)]:
            assignment = [-1] * 8
            assignment[qa] = gu
            assignment[qb] = gv
            used_mask = (1 << gu) | (1 << gv)

            def bt(idx, used_mask):
                if time.time() > deadline - 2:
                    return False
                if idx == 6:
                    for a, b in _Q3_EDGES:
                        ga = assignment[a]
                        gb = assignment[b]
                        if not (adj[ga] & (1 << gb)):
                            return False
                    return True

                x = unassigned[idx]
                candidates = ((1 << _N) - 1) & ~used_mask
                for y in range(8):
                    if assignment[y] != -1 and (_Q3_ADJ[x] & (1 << y)):
                        hy = assignment[y]
                        candidates &= adj[hy]

                while candidates:
                    bit = candidates & (-candidates)
                    g = bit.bit_length() - 1
                    candidates ^= bit

                    assignment[x] = g
                    new_used = used_mask | (1 << g)
                    if bt(idx + 1, new_used):
                        return True
                    assignment[x] = -1
                return False

            if bt(0, used_mask):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                return True

    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    return False

def _try_add_edge(adj, u, v, deadline):
    """Try to add edge (u,v) to adj, return True if successful."""
    if adj[u] & (1 << v):
        return True
    if not _has_q3_after_add(adj, u, v, deadline):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True
    return False

def _find_q3_witness(adj, u, v):
    """Find a Q3 embedding that would be created by adding edge (u,v). Returns list of edges in the Q3."""
    for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
        for (gu, gv) in [(u, v), (v, u)]:
            assignment = [-1] * 8
            assignment[qa] = gu
            assignment[qb] = gv
            used_mask = (1 << gu) | (1 << gv)

            def bt(idx, used_mask):
                if idx == 6:
                    for a, b in _Q3_EDGES:
                        ga = assignment[a]
                        gb = assignment[b]
                        if not (adj[ga] & (1 << gb)):
                            return False
                    return True

                x = unassigned[idx]
                candidates = ((1 << _N) - 1) & ~used_mask
                for y in range(8):
                    if assignment[y] != -1 and (_Q3_ADJ[x] & (1 << y)):
                        hy = assignment[y]
                        candidates &= adj[hy]

                while candidates:
                    bit = candidates & (-candidates)
                    g = bit.bit_length() - 1
                    candidates ^= bit

                    assignment[x] = g
                    new_used = used_mask | (1 << g)
                    if bt(idx + 1, new_used):
                        return True
                    assignment[x] = -1
                return False

            if bt(0, used_mask):
                # Found a Q3 witness
                q3_edges = []
                for a, b in _Q3_EDGES:
                    ga = assignment[a]
                    gb = assignment[b]
                    if ga < gb:
                        q3_edges.append((ga, gb))
                    else:
                        q3_edges.append((gb, ga))
                return q3_edges
    return None

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE

    full_mask = (1 << _N) - 1

    best_edges = None
    best_count = -1

    # Strategy 1: Start with K3,13
    adj = [0] * _N
    for a in range(3):
        for b in range(3, _N):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    cnt = _count_edges(adj)
    best_count = cnt
    best_edges = _edge_list(adj)

    # Greedy add with multiple restarts
    missing = [(i, j) for i in range(_N) for j in range(i+1, _N) if not (adj[i] & (1 << j))]
    
    for attempt in range(30):
        if time.time() > deadline - 15:
            break
        
        # Reset to K3,13
        adj = [0] * _N
        for a in range(3):
            for b in range(3, _N):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        
        cnt = _count_edges(adj)
        
        if attempt == 0:
            pass
        else:
            random.shuffle(missing)
            
        for u, v in missing:
            if time.time() > deadline - 15:
                break
            if _try_add_edge(adj, u, v, deadline):
                cnt += 1
        
        if cnt > best_count:
            best_count = cnt
            best_edges = _edge_list(adj)

    # Strategy 2: Local search with edge swaps
    adj = [0] * _N
    for u, v in best_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    current_count = _count_edges(adj)
    
    # Phase 2a: Try to add all possible missing edges
    missing = [(i, j) for i in range(_N) for j in range(i+1, _N) if not (adj[i] & (1 << j))]
    random.shuffle(missing)
    for u, v in missing:
        if time.time() > deadline - 20:
            break
        if _try_add_edge(adj, u, v, deadline):
            current_count += 1
    
    if current_count > best_count:
        best_count = current_count
        best_edges = _edge_list(adj)

    # Phase 2b: Local search - try edge swaps (remove one, add two)
    if time.time() < deadline - 25:
        for iteration in range(100):
            if time.time() > deadline - 10:
                break
            
            # Try random edge removals followed by greedy re-add
            adj = [0] * _N
            for u, v in best_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            current_count = _count_edges(adj)
            
            # Remove 1-2 random edges
            edge_list_copy = [(u, v) for u, v in best_edges]
            random.shuffle(edge_list_copy)
            num_remove = random.randint(1, 2)
            for u, v in edge_list_copy[:num_remove]:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_count -= 1
            
            # Greedy re-add
            missing = [(i, j) for i in range(_N) for j in range(i+1, _N) if not (adj[i] & (1 << j))]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline - 10:
                    break
                if _try_add_edge(adj, u, v, deadline):
                    current_count += 1
            
            if current_count > best_count:
                best_count = current_count
                best_edges = _edge_list(adj)

    # Phase 3: Targeted edge-swap using Q3 witnesses
    if time.time() < deadline - 15:
        adj = [0] * _N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        current_count = _count_edges(adj)
        
        missing = [(i, j) for i in range(_N) for j in range(i+1, _N) if not (adj[i] & (1 << j))]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 5:
                break
            
            # Find the Q3 witness that blocks this edge
            witness = _find_q3_witness(adj, u, v)
            if witness is None:
                # No Q3 found, try adding directly
                if _try_add_edge(adj, u, v, deadline):
                    current_count += 1
                    continue
            
            # Try removing specific edges from the witness to unblock
            for rem_u, rem_v in witness:
                if time.time() > deadline - 5:
                    break
                if rem_u == u and rem_v == v:
                    continue  # Don't remove the edge we're trying to add
                
                # Check if removing this edge allows us to add (u,v)
                test_adj = adj[:]
                test_adj[rem_u] &= ~(1 << rem_v)
                test_adj[rem_v] &= ~(1 << rem_u)
                
                if not _has_q3_after_add(test_adj, u, v, deadline):
                    # Remove the blocking edge and add the new one
                    adj[rem_u] &= ~(1 << rem_v)
                    adj[rem_v] &= ~(1 << rem_u)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    # current_count stays the same (removed 1, added 1)
                    break
    
    # Phase 4: Final polish - try to add any remaining edges
    adj = [0] * _N
    for u, v in best_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    current_count = _count_edges(adj)
    
    missing = [(i, j) for i in range(_N) for j in range(i+1, _N) if not (adj[i] & (1 << j))]
    random.shuffle(missing)
    for u, v in missing:
        if time.time() > deadline - 5:
            break
        if _try_add_edge(adj, u, v, deadline):
            current_count += 1
    
    if current_count > best_count:
        best_count = current_count
        best_edges = _edge_list(adj)

    return best_edges
# EVOLVE-BLOCK-END
