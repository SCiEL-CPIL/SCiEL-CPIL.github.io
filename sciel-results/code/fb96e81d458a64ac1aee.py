import time
import itertools
import random

_N = 16
_E = _N * (_N - 1) // 2  # 120 edges
_DEADLINE_MARGIN = 10

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

def _has_q3(adj_bits):
    assigned = [-1] * 8
    used_mask = 0
    
    def backtrack(qv):
        nonlocal used_mask
        if qv == 8:
            return True
        for gv in range(_N):
            if (used_mask >> gv) & 1:
                continue
            ok = True
            for pv in range(qv):
                if pv in _Q3_NEIGHBORS[qv]:
                    if not (adj_bits[gv] >> assigned[pv]) & 1:
                        ok = False
                        break
            if ok:
                assigned[qv] = gv
                used_mask |= (1 << gv)
                if backtrack(qv + 1):
                    return True
                used_mask &= ~(1 << gv)
                assigned[qv] = -1
        return False
    
    return backtrack(0)

def _get_all_edges():
    return [(u, v) for u in range(_N) for v in range(u + 1, _N)]

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_MARGIN
    
    all_edges = _get_all_edges()
    # Map edge to index
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Try removing k edges
    for k in range(0, 5):
        if time.time() > deadline:
            break
            
        # If k=0, just check K16
        if k == 0:
            adj_bits = [0] * _N
            for u, v in all_edges:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
            if not _has_q3(adj_bits):
                return all_edges
            continue
            
        # Try all combinations of k edges to remove
        # To speed up, we can sample if k is large
        # But for k=1,2,3,4 it's manageable
        # k=1: 120
        # k=2: 7140
        # k=3: 280840
        # k=4: 8.2M (too many, maybe sample or skip)
        
        if k > 3:
             # Fallback to heuristic if k=4 is too slow
             # Or just sample
             pass
             
        for removed in itertools.combinations(range(_E), k):
            if time.time() > deadline:
                break
                
            adj_bits = [0] * _N
            # Build graph with removed edges
            for i, (u, v) in enumerate(all_edges):
                if i in removed:
                    continue
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
            if not _has_q3(adj_bits):
                # Construct result list
                result = []
                for i, (u, v) in enumerate(all_edges):
                    if i not in removed:
                        result.append([u, v])
                return result
                
    # Fallback to current logic if not found
    # ... (copy current logic)
    
    return []
