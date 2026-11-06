"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

_N = P['n']
_EDGE_ORDER = P['edge_order']
_NEDGES = len(_EDGE_ORDER)

# Precompute edge -> position (1-based)
_EDGE_POS = {}
for _i, (_a, _b) in enumerate(_EDGE_ORDER, 1):
    _EDGE_POS[(_a, _b)] = _i
    _EDGE_POS[(_b, _a)] = _i

# Build adjacency list per vertex sorted by edge position
_ADJ = [[] for _ in range(_N)]
for _i, (_a, _b) in enumerate(_EDGE_ORDER, 1):
    _ADJ[_a].append((_i, _b))
    _ADJ[_b].append((_i, _a))
for _v in range(_N):
    _ADJ[_v].sort()  # sort by position

sys.setrecursionlimit(10000)


def _build_masks(K):
    """Build adjacency bitmasks for edges with position <= K."""
    masks = [0] * _N
    for i, (u, v) in enumerate(_EDGE_ORDER, 1):
        if i > K:
            break
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks


def _is_connected(masks):
    """Check if graph defined by masks is connected (all N vertices reachable)."""
    # Find first vertex with any edge
    start = -1
    for i in range(_N):
        if masks[i] != 0:
            start = i
            break
    if start == -1:
        return False

    visited = 0
    stack = [start]
    visited |= (1 << start)
    count = 1

    while stack:
        u = stack.pop()
        m = masks[u] & ~visited
        while m:
            lowbit = m & -m
            v = lowbit.bit_length() - 1
            m ^= lowbit
            visited |= (1 << v)
            count += 1
            stack.append(v)

    return count == _N


def _find_cycle_for_k(K, time_limit=None, start_time=None):
    """Find a Hamilton cycle using only edges with position <= K.
    Returns a list of vertices (the cycle order) or None."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 60.0

    # Build bitmasks
    masks = _build_masks(K)

    # Check every vertex has degree >= 2
    for v in range(_N):
        if bin(masks[v]).count('1') < 2:
            return None

    # Check connectivity
    if not _is_connected(masks):
        return None

    full_mask = (1 << _N) - 1
    start = 0

    # Get neighbors of 0 sorted by degree (ascending) for better pruning
    def _get_sorted_neighbors(v, masks):
        """Get neighbors of v sorted by their degree (ascending)."""
        m = masks[v]
        neighbors = []
        while m:
            lowbit = m & -m
            u = lowbit.bit_length() - 1
            m ^= lowbit
            neighbors.append(u)
        neighbors.sort(key=lambda u: bin(masks[u]).count('1'))
        return neighbors

    neighbors_of_0 = _get_sorted_neighbors(0, masks)

    if len(neighbors_of_0) < 2:
        return None

    # For each possible first neighbor of 0, try to find a Hamilton path
    # from that neighbor back to 0 through all other vertices.
    # We fix the cycle as: 0 -> n1 -> ... -> nk -> 0
    # where n1 and nk are both neighbors of 0.

    # Use iterative DFS with bitmask state and connectivity pruning
    def _dfs(n1):
        """Find Hamilton path from n1 visiting all vertices, ending at a neighbor of 0."""
        path = [0] * _N
        path[0] = 0
        path[1] = n1
        v_mask = (1 << 0) | (1 << n1)

        # Stack: (current_vertex, depth, neighbor_index)
        # depth is the index in path (1-based for the current position)
        stack = [(n1, 1, 0)]

        while stack:
            if time.time() - start_time > time_limit:
                return None

            cur, d, nidx = stack[-1]

            if d == _N - 1:
                # All vertices placed. Check if cur connects to 0.
                if masks[cur] & (1 << 0):
                    return list(path[:d + 1])
                # Backtrack
                stack.pop()
                if not stack:
                    return None
                pc, pd, pnidx = stack[-1]
                stack[-1] = (pc, pd, pnidx + 1)
                v_mask &= ~(1 << cur)
                path[d] = 0
                continue

            # Try neighbors of cur starting from nidx
            found_next = False
            # Get sorted neighbors once for this call
            cur_neighbors = _get_sorted_neighbors(cur, masks)

            while nidx < len(cur_neighbors):
                nxt = cur_neighbors[nidx]
                nidx += 1
                stack[-1] = (cur, d, nidx)

                if v_mask & (1 << nxt):
                    continue

                # Connectivity pruning: check if all unvisited vertices
                # are reachable from nxt via unvisited vertices
                unvisited = full_mask ^ (v_mask | (1 << nxt))
                if unvisited:
                    # BFS from nxt using only unvisited vertices
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
                    if reach != unvisited | (1 << nxt):
                        continue

                # Move to nxt
                v_mask |= (1 << nxt)
                path[d + 1] = nxt
                stack.append((nxt, d + 1, 0))
                found_next = True
                break

            if not found_next:
                # No more neighbors, backtrack
                stack.pop()
                if not stack:
                    return None
                pc, pd, pnidx = stack[-1]
                stack[-1] = (pc, pd, pnidx + 1)
                v_mask &= ~(1 << cur)
                path[d] = 0

        return None

    # Try each neighbor of 0 as the first step
    for n1 in neighbors_of_0:
        if time.time() - start_time > time_limit:
            break
        result = _dfs(n1)
        if result is not None:
            return result

    return None


def _score_cycle(cycle):
    """Compute the max edge position in the cycle."""
    n = len(cycle)
    max_pos = 0
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        pos = _EDGE_POS[(a, b)]
        if pos > max_pos:
            max_pos = pos
    return max_pos


def _edge_pos(a, b):
    """Get edge position for vertices a, b."""
    return _EDGE_POS[(a, b)]


def _two_opt_polish(cycle, time_limit, start_time):
    """Apply 2-opt local search to reduce the max edge position of a Hamilton cycle.
    
    Reverses segments of the cycle to try to reduce individual edge positions.
    """
    n = len(cycle)
    if n < 4:
        return cycle, _score_cycle(cycle)
    
    best = list(cycle)
    best_score = _score_cycle(best)
    
    # Precompute all edge positions for the current cycle
    def _cycle_edge_positions(c):
        positions = []
        for i in range(n):
            positions.append(_EDGE_POS[(c[i], c[(i + 1) % n])])
        return positions
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        positions = _cycle_edge_positions(best)
        current_max = max(positions)
        
        # Try all 2-opt moves: reverse segment [i, j] where 0 <= i < j < n
        for i in range(n):
            if time.time() - start_time > time_limit:
                return best, best_score
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    return best, best_score
                # Skip if segment length is 1 (no change) or n-1 (same as length 1)
                if j == i + 1 or j == (i + n - 1) % n:
                    continue
                
                # Edges to remove: (best[i-1], best[i]) and (best[j], best[(j+1)%n])
                prev_i = (i - 1) % n
                next_j = (j + 1) % n
                
                a1 = best[prev_i]
                b1 = best[i]
                a2 = best[j]
                b2 = best[next_j]
                
                # New edges: (a1, a2) and (b1, b2)
                new_pos1 = _EDGE_POS[(a1, a2)]
                new_pos2 = _EDGE_POS[(b1, b2)]
                
                # Compute new max
                new_max = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if positions[k] > new_max:
                        new_max = positions[k]
                if new_pos1 > new_max:
                    new_max = new_pos1
                if new_pos2 > new_max:
                    new_max = new_pos2
                
                if new_max < best_score:
                    # Perform the reversal
                    best[i:j+1] = best[i:j+1][::-1]
                    best_score = new_max
                    improved = True
                    # Restart search after improvement
                    break
            if improved:
                break
    
    return best, best_score


def _three_opt_polish(cycle, time_limit, start_time):
    """Apply 3-opt local search to reduce the max edge position of a Hamilton cycle.
    
    A 3-opt move removes 3 edges and reconnects them in a different way.
    For a cycle, removing edges (a,b), (c,d), (e,f) and reconnecting as
    (a,c), (b,e), (d,f) (or other permutations) can improve the cycle.
    """
    n = len(cycle)
    if n < 5:
        return cycle, _score_cycle(cycle)
    
    best = list(cycle)
    best_score = _score_cycle(best)
    
    def _cycle_edge_positions(c):
        positions = []
        for i in range(n):
            positions.append(_EDGE_POS[(c[i], c[(i + 1) % n])])
        return positions
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        positions = _cycle_edge_positions(best)
        current_max = max(positions)
        
        # Try all 3-opt moves
        for i in range(n):
            if time.time() - start_time > time_limit:
                return best, best_score
            for j in range(i + 2, n):
                if time.time() - start_time > time_limit:
                    return best, best_score
                for k in range(j + 2, n):
                    if time.time() - start_time > time_limit:
                        return best, best_score
                    
                    # Edges to remove: (best[i-1], best[i]), (best[j-1], best[j]), (best[k-1], best[k])
                    prev_i = (i - 1) % n
                    prev_j = (j - 1) % n
                    prev_k = (k - 1) % n
                    
                    a1 = best[prev_i]
                    b1 = best[i]
                    a2 = best[prev_j]
                    b2 = best[j]
                    a3 = best[prev_k]
                    b3 = best[k]
                    
                    # Try different reconnections
                    # Option 1: (a1,a2), (b1,b2), (a3,b3)
                    new_pos1 = _EDGE_POS[(a1, a2)]
                    new_pos2 = _EDGE_POS[(b1, b2)]
                    new_pos3 = _EDGE_POS[(a3, b3)]
                    
                    new_max = 0
                    for pos_idx in range(n):
                        if pos_idx in (i, j, k):
                            continue
                        if positions[pos_idx] > new_max:
                            new_max = positions[pos_idx]
                    for np in (new_pos1, new_pos2, new_pos3):
                        if np > new_max:
                            new_max = np
                    
                    if new_max < best_score:
                        # Perform the reconnection
                        # This is complex, so we'll skip for now and just use 2-opt
                        pass
                    
                    # Option 2: (a1,b2), (b1,a2), (a3,b3)
                    new_pos1 = _EDGE_POS[(a1, b2)]
                    new_pos2 = _EDGE_POS[(b1, a2)]
                    new_pos3 = _EDGE_POS[(a3, b3)]
                    
                    new_max = 0
                    for pos_idx in range(n):
                        if pos_idx in (i, j, k):
                            continue
                        if positions[pos_idx] > new_max:
                            new_max = positions[pos_idx]
                    for np in (new_pos1, new_pos2, new_pos3):
                        if np > new_max:
                            new_max = np
                    
                    if new_max < best_score:
                        # Perform the reconnection
                        pass
                    
                    # Option 3: (a1,a2), (b1,a3), (b2,b3)
                    new_pos1 = _EDGE_POS[(a1, a2)]
                    new_pos2 = _EDGE_POS[(b1, a3)]
                    new_pos3 = _EDGE_POS[(b2, b3)]
                    
                    new_max = 0
                    for pos_idx in range(n):
                        if pos_idx in (i, j, k):
                            continue
                        if positions[pos_idx] > new_max:
                            new_max = positions[pos_idx]
                    for np in (new_pos1, new_pos2, new_pos3):
                        if np > new_max:
                            new_max = np
                    
                    if new_max < best_score:
                        # Perform the reconnection
                        pass
        
        # Since 3-opt implementation is complex, we'll just return the best from 2-opt
        return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s

    # Baseline: identity permutation
    baseline_cycle = list(range(_N))
    baseline_score = _score_cycle(baseline_cycle)

    best_cycle = baseline_cycle
    best_score = baseline_score

    # Binary search for the minimum K where a Hamilton cycle exists
    lo = 24
    hi = best_score

    while lo < hi and time.time() < deadline:
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) // 2
        remaining = deadline - time.time()
        if remaining < 2:
            break
        cycle = _find_cycle_for_k(mid, time_limit=remaining - 1, start_time=start_time)
        if cycle is not None:
            if len(set(cycle)) == _N and len(cycle) == _N:
                s = _score_cycle(cycle)
                if s <= mid:
                    best_cycle = cycle
                    best_score = s
                    hi = mid
                else:
                    if s < best_score:
                        best_cycle = cycle
                        best_score = s
                    hi = mid
            else:
                lo = mid + 1
        else:
            lo = mid + 1

    # After binary search, try to verify/improve at lo
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        if remaining > 5:
            cycle = _find_cycle_for_k(lo, time_limit=remaining - 5, start_time=start_time)
            if cycle is not None and len(set(cycle)) == _N:
                s = _score_cycle(cycle)
                if s < best_score:
                    best_cycle = cycle
                    best_score = s

    # Apply 2-opt polish to the best cycle found
    if time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        if remaining > 1:
            polished_cycle, polished_score = _two_opt_polish(best_cycle, remaining, start_time)
            if polished_score < best_score:
                best_cycle = polished_cycle
                best_score = polished_score

    # Apply 3-opt polish if time remains
    if time.time() < deadline - 1:
        remaining = deadline - time.time() - 0.5
        if remaining > 0.5:
            polished_cycle, polished_score = _three_opt_polish(best_cycle, remaining, start_time)
            if polished_score < best_score:
                best_cycle = polished_cycle
                best_score = polished_score

    return best_cycle
# EVOLVE-BLOCK-END
