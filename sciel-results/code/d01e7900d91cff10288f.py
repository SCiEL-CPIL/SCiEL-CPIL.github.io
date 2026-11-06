"""Rewritten: linear scan with fast Hamilton cycle feasibility + direct construction."""
import time
import sys

_N = 24
_EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

_NEDGES = len(_EDGE_ORDER)

# Precompute edge -> position (1-based)
_EDGE_POS = {}
for _i, (_a, _b) in enumerate(_EDGE_ORDER, 1):
    _EDGE_POS[(_a, _b)] = _i
    _EDGE_POS[(_b, _a)] = _i

sys.setrecursionlimit(10000)


def _score_cycle(cycle):
    n = len(cycle)
    max_pos = 0
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        pos = _EDGE_POS[(a, b)]
        if pos > max_pos:
            max_pos = pos
    return max_pos


def _build_adj_masks(K):
    """Build adjacency bitmasks for edges with position <= K."""
    masks = [0] * _N
    for i, (u, v) in enumerate(_EDGE_ORDER, 1):
        if i > K:
            break
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks


def _check_feasibility_fast(masks):
    """Quick necessary-condition check: every vertex has degree >= 2, graph is connected."""
    for v in range(_N):
        if bin(masks[v]).count('1') < 2:
            return False
    # Connectivity
    start = -1
    for i in range(_N):
        if masks[i] != 0:
            start = i
            break
    if start == -1:
        return False
    visited = 1 << start
    stack = [start]
    while stack:
        u = stack.pop()
        m = masks[u] & ~visited
        while m:
            lowbit = m & -m
            v = lowbit.bit_length() - 1
            m ^= lowbit
            visited |= (1 << v)
            stack.append(v)
    return visited == (1 << _N) - 1


def _find_ham_cycle(masks, deadline):
    """Find a Hamilton cycle in the graph defined by masks.
    Returns a list of _N vertices or None.
    Uses DFS with aggressive pruning."""
    full_mask = (1 << _N) - 1
    start = 0
    start_deadline = time.time()
    
    # Precompute degrees
    degs = [bin(masks[v]).count('1') for v in range(_N)]
    
    # Get neighbors of 0 sorted by degree (ascending - prefer low-degree vertices)
    def get_neighbors_sorted(v):
        m = masks[v]
        nbrs = []
        while m:
            lowbit = m & -m
            u = lowbit.bit_length() - 1
            m ^= lowbit
            nbrs.append(u)
        nbrs.sort(key=lambda u: degs[u])
        return nbrs
    
    n0_neighbors = get_neighbors_sorted(0)
    if len(n0_neighbors) < 2:
        return None
    
    path = [0] * _N
    path[0] = 0
    
    for n1 in n0_neighbors:
        if time.time() > deadline:
            return None
        
        path[1] = n1
        v_mask = (1 << 0) | (1 << n1)
        
        # Stack-based DFS: (current_vertex, depth, neighbor_index)
        stack = [(n1, 1, 0)]
        
        while stack:
            if time.time() > start_deadline + 40:
                return None
            
            cur, d, nidx = stack[-1]
            
            if d == _N - 1:
                # All vertices placed. Check if cur connects to 0.
                if masks[cur] & (1 << 0):
                    return list(path[:d + 1])
                # Backtrack
                stack.pop()
                if not stack:
                    break
                pc, pd, pnidx = stack[-1]
                stack[-1] = (pc, pd, pnidx + 1)
                v_mask &= ~(1 << cur)
                path[d] = 0
                continue
            
            cur_neighbors = get_neighbors_sorted(cur)
            found_next = False
            
            while nidx < len(cur_neighbors):
                nxt = cur_neighbors[nidx]
                nidx += 1
                stack[-1] = (cur, d, nidx)
                
                if v_mask & (1 << nxt):
                    continue
                
                # Pruning 1: if nxt is 0 and d < N-1, skip (can't close cycle early)
                if nxt == 0 and d < _N - 1:
                    continue
                
                # Pruning 2: connectivity check on unvisited vertices
                # After adding nxt, check if all unvisited vertices are reachable
                # from the current path's endpoints through unvisited vertices
                new_v_mask = v_mask | (1 << nxt)
                unvisited = full_mask ^ new_v_mask
                
                if unvisited:
                    # Check connectivity of unvisited subgraph + connection to path
                    # The unvisited vertices must form a connected structure that
                    # can be reached from either end of the current path
                    # Simplified: check if unvisited graph is connected
                    # and has at least one vertex adjacent to the path
                    reach = 1 << nxt
                    bfs_stack = [nxt]
                    while bfs_stack:
                        u = bfs_stack.pop()
                        m = masks[u] & unvisited
                        while m:
                            lowbit = m & -m
                            w = lowbit.bit_length() - 1
                            m ^= lowbit
                            if not (reach & (1 << w)):
                                reach |= (1 << w)
                                bfs_stack.append(w)
                    # All unvisited must be reachable from nxt
                    if reach != unvisited | (1 << nxt):
                        continue
                
                # Pruning 3: degree constraint - vertices that will have only 2 edges
                # in the final cycle must already have their 2 edges determined
                # This is complex; skip for speed
                
                # Move to nxt
                v_mask |= (1 << nxt)
                path[d + 1] = nxt
                stack.append((nxt, d + 1, 0))
                found_next = True
                break
            
            if not found_next:
                stack.pop()
                if not stack:
                    break
                pc, pd, pnidx = stack[-1]
                stack[-1] = (pc, pd, pnidx + 1)
                v_mask &= ~(1 << cur)
                path[d] = 0
        
        # Try next n1

    return None


def _two_opt_improve(cycle, deadline):
    """Improve cycle score via 2-opt moves."""
    n = len(cycle)
    best_cycle = list(cycle)
    best_score = _score_cycle(best_cycle)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                
                a = best_cycle[i]
                b = best_cycle[(i + 1) % n]
                c = best_cycle[j]
                d = best_cycle[(j + 1) % n]
                
                # 2-opt: reverse segment i+1..j
                # New edges: (a, c) and (b, d)
                if a == c or b == d or a == d or b == c:
                    continue
                
                if (a, c) not in _EDGE_POS or (b, d) not in _EDGE_POS:
                    continue
                
                pos_ac = _EDGE_POS[(a, c)]
                pos_bd = _EDGE_POS[(b, d)]
                
                if pos_ac > best_score or pos_bd > best_score:
                    continue
                
                # Build new cycle
                new_cycle = best_cycle[:i+1] + best_cycle[i+1:j+1][::-1] + best_cycle[j+1:]
                new_score = _score_cycle(new_cycle)
                
                if new_score < best_score:
                    best_cycle = new_cycle
                    best_score = new_score
                    improved = True
                    break
            if improved:
                break
    
    return best_cycle, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin
    
    # Start from a reasonable lower bound
    # A Hamilton cycle needs at least 24 edges, and each vertex needs degree >= 2
    # So minimum K is at least 24 (but likely higher)
    # Upper bound: score of identity cycle
    identity = list(range(_N))
    id_score = _score_cycle(identity)
    
    best_cycle = identity
    best_score = id_score
    
    # Linear scan from low K to high K
    # Start from K=24 and go up
    # But first, find a good lower bound by checking when all vertices have degree >= 2
    
    # Find minimum K where all vertices have degree >= 2
    min_K = 24
    for K in range(24, 100):
        masks = _build_adj_masks(K)
        ok = True
        for v in range(_N):
            if bin(masks[v]).count('1') < 2:
                ok = False
                break
        if ok:
            min_K = K
            break
    
    # Now scan from min_K upward
    for K in range(min_K, min(min_K + 80, _NEDGES + 1)):
        if time.time() > deadline - 5:
            break
        
        masks = _build_adj_masks(K)
        
        # Quick feasibility check
        if not _check_feasibility_fast(masks):
            continue
        
        # Try to find a Hamilton cycle
        cycle = _find_ham_cycle(masks, deadline - 5)
        
        if cycle is not None and len(cycle) == _N and len(set(cycle)) == _N:
            s = _score_cycle(cycle)
            if s < best_score:
                best_cycle = cycle
                best_score = s
                # If we found a cycle with max edge <= K, and K is the current threshold,
                # the actual score might be less than K
                # Try to improve with 2-opt
                if time.time() < deadline - 3:
                    polished, polished_score = _two_opt_improve(best_cycle, deadline - 2)
                    if polished_score < best_score:
                        best_cycle = polished
                        best_score = polished_score
                # We found a valid cycle; break since we're scanning in order
                # and this is the first feasible K
                break
    
    # Final polish
    if time.time() < deadline - 1:
        polished, polished_score = _two_opt_improve(best_cycle, deadline)
        if polished_score < best_score:
            best_cycle = polished
            best_score = polished_score
    
    return best_cycle
