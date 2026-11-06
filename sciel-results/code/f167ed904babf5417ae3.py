import time
import sys

# EVOLVE-BLOCK-START
N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions (1-based)
_edge_pos = {}
for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
    _edge_pos[(min(_u, _v), max(_u, _v))] = _i

def _edge_idx(u, v):
    return _edge_pos[(min(u, v), max(u, v))]

def _cycle_score(cycle):
    return max(_edge_idx(cycle[i], cycle[(i + 1) % N]) for i in range(N))

# Precompute, for each vertex, its incident edges sorted by position
# adj_by_pos[v] = list of (neighbor, position) sorted by position
_adj_by_pos = [[] for _ in range(N)]
for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
    _adj_by_pos[_u].append((_v, _i))
    _adj_by_pos[_v].append((_u, _i))
for _v in range(N):
    _adj_by_pos[_v].sort(key=lambda x: x[1])

# Lower bound: for each vertex, the 2nd-smallest incident edge index
# (every vertex must have degree >= 2 in the prefix graph)
_lb = 0
for _v in range(N):
    # _adj_by_pos[_v] is sorted by position
    if len(_adj_by_pos[_v]) >= 2:
        _lb = max(_lb, _adj_by_pos[_v][1][1])
    else:
        _lb = max(_lb, _adj_by_pos[_v][-1][1])

def _greedy_cycle():
    """Greedy: always pick the unvisited neighbor with the smallest edge index."""
    visited = [False] * N
    visited[0] = True
    current = 0
    cycle = [0]
    for _ in range(N - 1):
        best_next = -1
        best_idx = float('inf')
        for j in range(N):
            if not visited[j]:
                idx = _edge_idx(current, j)
                if idx < best_idx:
                    best_idx = idx
                    best_next = j
        if best_next == -1:
            return None
        visited[best_next] = True
        cycle.append(best_next)
        current = best_next
    return cycle

def _build_adj_mask(K):
    """Return list of N bitmasks using only edges with position <= K."""
    masks = [0] * N
    for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
        if _i > K:
            break
        masks[_u] |= (1 << _v)
        masks[_v] |= (1 << _u)
    return masks

def _dfs_hamilton(masks, deadline):
    """
    Iterative DFS with bitmasks to find a Hamilton cycle starting at 0.
    Uses Warnsdorff-style ordering: prefer neighbors with fewer unvisited options.
    Returns the cycle (list of vertices) or None.
    """
    full_mask = (1 << N) - 1
    start = 0
    start_mask = 1 << start

    # Iterative DFS with explicit stack
    # Each stack entry: (vertex, visited_mask, path_tuple)
    # We use a list for path for mutability
    path = [start]
    visited = start_mask
    stack = [(start, visited, 0)]  # (vertex, visited_mask, next_neighbor_idx)

    # Precompute neighbor lists from masks for each vertex
    # Actually, we iterate via bit operations

    # For Warnsdorff ordering, we need to try neighbors in order of
    # their unvisited-degree (smallest first). We'll compute this on the fly.

    def get_neighbors_ordered(v, vis):
        """Get unvisited neighbors of v, sorted by unvisited-degree (ascending)."""
        avail = masks[v] & ~vis
        if avail == 0:
            return []
        neighbors = []
        tmp = avail
        while tmp:
            lsb = tmp & (-tmp)
            nb = lsb.bit_length() - 1
            # Count unvisited neighbors of nb
            cnt = (masks[nb] & ~vis & ~(1 << v)).bit_count()
            neighbors.append((cnt, nb))
            tmp ^= lsb
        neighbors.sort(key=lambda x: x[0])
        return [nb for _, nb in neighbors]

    # Iterative DFS
    # We track for each vertex in the path the index of the next neighbor to try
    # This is complex; let's use a simpler recursive-like approach with a manual stack.

    # Actually, for N=24, let's use a clean iterative approach:
    # Stack of (vertex, visited_mask, path, neighbor_iterator_state)
    # Simpler: use recursion with sys.setrecursionlimit, but iterative is safer.

    # Let me use a clean iterative DFS:
    # stack entries: (vertex, visited_mask, path_list)
    # We pop, extend path, push children.

    # But that's DFS with backtracking. Let me do it properly:

    def dfs_iterative():
        # State: (current_vertex, visited_mask, path)
        # We'll simulate recursion with a stack of iterators
        path = [start]
        visited = start_mask

        # For each level, we need to remember which neighbors we've tried
        # Use a stack of (vertex, visited_mask, neighbor_list, next_idx)
        neighbor_lists = []
        next_indices = []

        v = start
        vis = visited

        while True:
            if len(path) == N:
                # Check closing edge
                if masks[v] & (1 << start):
                    return path[:]
                # Can't close, backtrack
                if not path:
                    return None
                path.pop()
                v = path[-1] if path else start
                vis = 0
                # Recompute visited from path
                vis = 0
                for p in path:
                    vis |= (1 << p)
                # Continue backtracking
                if neighbor_lists:
                    neighbor_lists.pop()
                    next_indices.pop()
                continue

            if not neighbor_lists or next_indices[-1] >= len(neighbor_lists[-1]):
                # Need to get neighbors for current vertex
                if not path:
                    return None
                v = path[-1]
                vis = 0
                for p in path:
                    vis |= (1 << p)
                nbrs = get_neighbors_ordered(v, vis)
                neighbor_lists.append(nbrs)
                next_indices.append(0)

            # Try next neighbor
            idx = next_indices[-1]
            nbrs = neighbor_lists[-1]
            if idx < len(nbrs):
                nb = nbrs[idx]
                next_indices[-1] = idx + 1
                path.append(nb)
                # Continue loop
            else:
                # Backtrack
                path.pop()
                if not path:
                    return None
                neighbor_lists.pop()
                next_indices.pop()
                if not neighbor_lists:
                    # All backtracked
                    # Recompute
                    v = path[-1]
                    vis = 0
                    for p in path:
                        vis |= (1 << p)
                    nbrs = get_neighbors_ordered(v, vis)
                    neighbor_lists.append(nbrs)
                    next_indices.append(0)

    result = dfs_iterative()
    return result

def _improve_cycle(cycle, max_k, time_limit=0.5):
    """2-opt local search to reduce the maximum edge index."""
    current_cycle = list(cycle)
    current_score = _cycle_score(current_cycle)
    if current_score > max_k:
        return current_cycle, current_score

    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N - 1):
            for j in range(i + 1, N):
                if i == 0 and j == N - 1:
                    continue
                a = current_cycle[i - 1]
                b = current_cycle[i]
                c = current_cycle[j]
                d = current_cycle[(j + 1) % N]

                if _edge_idx(a, c) > max_k or _edge_idx(b, d) > max_k:
                    continue

                new_cycle = current_cycle[:i] + current_cycle[i:j + 1][::-1] + current_cycle[j + 1:]
                new_score = _cycle_score(new_cycle)

                if new_score < current_score:
                    current_cycle = new_cycle
                    current_score = new_score
                    improved = True
    return current_cycle, current_score

def _polish_cycle(cycle, time_limit=1.0):
    """
    Final polish: repeatedly apply 2-opt moves that reduce the max edge index.
    No K constraint — just minimize the score.
    """
    current_cycle = list(cycle)
    current_score = _cycle_score(current_cycle)
    start_time = time.time()

    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N - 1):
            for j in range(i + 1, N):
                if i == 0 and j == N - 1:
                    continue
                a = current_cycle[i - 1]
                b = current_cycle[i]
                c = current_cycle[j]
                d = current_cycle[(j + 1) % N]

                new_cycle = current_cycle[:i] + current_cycle[i:j + 1][::-1] + current_cycle[j + 1:]
                new_score = _cycle_score(new_cycle)

                if new_score < current_score:
                    current_cycle = new_cycle
                    current_score = new_score
                    improved = True
    return current_cycle, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    # 1. Greedy initial cycle
    best_cycle = _greedy_cycle()
    if best_cycle is None:
        best_cycle = list(range(N))
    best_K = _cycle_score(best_cycle)

    # 2. Improve with local search
    best_cycle, best_K = _improve_cycle(best_cycle, best_K, time_limit=1.0)

    # 3. Binary search for minimal K
    lo = _lb
    hi = best_K
    last_feasible_cycle = best_cycle
    last_feasible_K = best_K

    while lo < hi:
        if time.time() > deadline - 5:
            break

        mid = (lo + hi) // 2

        # Quick degree check
        masks = _build_adj_mask(mid)
        feasible = True
        for u in range(N):
            if masks[u].bit_count() < 2:
                feasible = False
                break

        if feasible:
            cycle = _dfs_hamilton(masks, deadline - 5)
            if cycle is not None:
                score = _cycle_score(cycle)
                if score < last_feasible_K:
                    last_feasible_K = score
                    last_feasible_cycle = cycle
                hi = mid
            else:
                lo = mid + 1
        else:
            lo = mid + 1

    # 4. Final polish
    if time.time() < deadline - 3:
        last_feasible_cycle, last_feasible_K = _polish_cycle(last_feasible_cycle, time_limit=2.0)

    # 5. Verify
    final_score = _cycle_score(last_feasible_cycle)
    return last_feasible_cycle
# EVOLVE-BLOCK-END
