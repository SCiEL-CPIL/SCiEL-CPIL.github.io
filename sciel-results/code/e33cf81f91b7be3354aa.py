import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
TOTAL_EDGES = N * (N - 1) // 2  # 276

# arrival index (0-based) for each edge
_arrival = {}
for _i, (_a, _b) in enumerate(P['edge_order']):
    if _a > _b:
        _a, _b = _b, _a
    _arrival[(_a, _b)] = _i

def _edge_idx(a, b):
    if a > b:
        a, b = b, a
    return _arrival[(a, b)]

# adjacency bitmasks for incremental graph
# adj_mask[v] is a bitmask of neighbors present in current prefix
# We'll maintain this as we grow k.

def _has_ham_cycle(adj, n, deadline):
    """Try to find a Hamilton cycle in graph given by adjacency bitmasks.
    Returns a list of vertices (cycle order) or None.
    Uses backtracking with pruning.
    """
    # Fix start vertex as 0 to break rotational symmetry.
    # We'll build a path starting at 0, must end at a neighbor of 0, and visit all.
    path = [0]
    visited = 1  # bitmask, vertex 0 set
    remaining = ((1 << n) - 1) ^ 1  # all except 0

    # For backtracking, we track:
    # - path: list of vertices in order
    # - visited: bitmask of visited vertices
    # - For each vertex in path (except last), it has degree 2 in the cycle (prev and next)
    #   The last vertex will need to connect back to 0.
    #   So for the last vertex in path, it needs an edge to 0 (not yet used).
    #   For intermediate vertices, both edges are used.
    #   Vertex 0 has one edge used (to path[1]), needs one more (to path[-1]).

    # Pruning:
    # 1. The unvisited subgraph must allow a path from path[-1] to a neighbor of 0, visiting all remaining.
    # 2. Any vertex that must have degree 2 in the cycle: if it's not in path, it needs at least 2 edges
    #    to (path vertices that can connect to it) or to other unvisited vertices.
    #    More precisely: for each unvisited vertex v, the number of edges from v to (path ∪ {unvisited})
    #    that are "available" must be >= 2, unless v is the last vertex (which connects to 0).
    #    Actually, every vertex in the cycle has degree 2. For unvisited vertices, both edges must go to
    #    vertices in the final cycle. Since 0 is already in the cycle, and the path vertices are in the cycle,
    #    an unvisited vertex v can connect to: path vertices (but only path[-1] can connect to v as the
    #    "next" in path, and 0 can connect to v if v is the last), or other unvisited vertices.
    #    This is complex. Let's use a simpler but effective pruning:
    #    - Connectivity: all unvisited vertices must be reachable from path[-1] in the subgraph induced by
    #      unvisited ∪ {path[-1]} ∪ {0} (since path[-1] must connect to next, and eventually to 0).
    #    - Degree constraint: every unvisited vertex must have degree >= 2 in the graph induced by
    #      (unvisited ∪ {0} ∪ {path[-1]}). Wait, that's not quite right either.

    # Let me use a cleaner formulation:
    # We're building a Hamiltonian path from 0 to some vertex w, where w is adjacent to 0.
    # The path visits all vertices exactly once.
    # At any point, path = [v0, v1, ..., vk] where v0=0.
    # Constraints:
    # - vk must be able to reach a neighbor of 0 through the remaining unvisited vertices.
    # - For each unvisited vertex u, it must have at least 2 neighbors in (unvisited ∪ {vk} ∪ {0}).
    #   Because in the final cycle, u has degree 2, and its neighbors are either in the path (only vk
    #   can be adjacent to u as the "next" vertex) or 0 (if u is the last vertex) or other unvisited.
    #   Actually, u's two neighbors in the cycle are among: {vk} (if u is next after vk), {0} (if u is last),
    #   and other unvisited vertices. So u needs at least 2 edges to (unvisited ∪ {vk} ∪ {0}), but we need
    #   to be careful: if u is the vertex right after vk, then one of its edges goes to vk, and the other
    #   goes to something else. If u is the last vertex, one edge goes to 0.
    #   Simplest safe prune: for each unvisited u, deg(u) in (unvisited ∪ {vk} ∪ {0}) >= 2.
    #   This is necessary but not sufficient.

    import sys
    sys.setrecursionlimit(10000)

    def dfs(k, path, visited):
        if time.time() > deadline:
            return None
        if len(path) == n:
            # Check if last vertex is adjacent to 0
            last = path[-1]
            if (adj[last] & 1) != 0:  # vertex 0 is bit 0
                return path[:]
            return None

        last = path[-1]
        # Candidates: unvisited neighbors of last
        cand_mask = adj[last] & ~visited
        # Try each candidate
        # Order by degree (fewest options first) for better pruning
        cands = []
        cm = cand_mask
        while cm:
            lsb = cm & (-cm)
            v = lsb.bit_length() - 1
            cands.append(v)
            cm ^= lsb

        # Sort by degree in remaining graph (unvisited ∪ {last} ∪ {0})
        remaining_mask = ((1 << n) - 1) ^ visited
        def deg_in_remaining(v):
            # Count neighbors of v in (remaining_mask | (1 << last) | 1)
            mask = adj[v] & (remaining_mask | (1 << last) | 1)
            return bin(mask).count('1')

        cands.sort(key=deg_in_remaining)

        for nxt in cands:
            new_visited = visited | (1 << nxt)
            new_path = path + [nxt]
            # Prune: check degree constraint for all unvisited vertices
            rem = ((1 << n) - 1) ^ new_visited
            valid = True
            if rem:
                # For each unvisited vertex, check it has >= 2 neighbors in (rem ∪ {nxt} ∪ {0})
                check_mask = rem | (1 << nxt) | 1
                rm = rem
                while rm:
                    lsb = rm & (-rm)
                    u = lsb.bit_length() - 1
                    if bin(adj[u] & check_mask).count('1') < 2:
                        valid = False
                        break
                    rm ^= lsb
            if not valid:
                continue
            # Connectivity check: all unvisited must be reachable from nxt in subgraph (rem ∪ {nxt} ∪ {0})
            if rem:
                # BFS/DFS from nxt in subgraph induced by (rem ∪ {nxt} ∪ {0})
                sub_mask = rem | (1 << nxt) | 1
                seen = 1 << nxt
                stack = [nxt]
                while stack:
                    cur = stack.pop()
                    nbrs = adj[cur] & sub_mask & ~seen
                    nm = nbrs
                    while nm:
                        lsb = nm & (-nm)
                        w = lsb.bit_length() - 1
                        seen |= (1 << w)
                        stack.append(w)
                        nm ^= lsb
                if seen != sub_mask:
                    continue
            result = dfs(k + 1, new_path, new_visited)
            if result is not None:
                return result
        return None

    return dfs(0, [0], 1)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin

    # Start with a reasonable lower bound on k.
    # Each vertex needs degree >= 2 in the cycle, so we need at least 2N edges... but that's not tight.
    # Let's just start from k=1 and go up, but that's slow. Instead, binary search or linear from a guess.
    # For 24 vertices, the hitting time for Hamilton cycle in random edge process is around 2N ln N ≈ 2*24*3.17 ≈ 152.
    # But this is a specific shuffle, so let's try a range.

    best_cycle = None
    best_k = None

    # Build adjacency incrementally
    # adj[v] is a bitmask
    adj = [0] * N

    # Try k from a lower bound upward
    # Lower bound: need at least 2N edges for 2-regular, so k >= 2N = 48. But Hamilton cycle needs specific structure.
    # Let's try from k=40 upward, but use the fact that we can test incrementally.

    # Actually, let's do a smarter approach: try k values in increasing order, and for each,
    # check if Hamilton cycle exists. We can cache the adjacency state.

    # First, let's try a range of k values.
    # For efficiency, let's try k = 40, 45, 50, 55, 60, ... and binary search once we find one.

    # Actually, let's just iterate k from 1 to TOTAL_EDGES, but stop early if we find a cycle.
    # To speed up, let's first do a quick check with a higher k to see if it's feasible, then narrow down.

    # Let's try k from 40 to 276
    for k in range(40, TOTAL_EDGES + 1):
        if time.time() > deadline:
            break
        # Add edge k-1 to adjacency
        a, b = P['edge_order'][k - 1]
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

        # Quick check: every vertex must have degree >= 2
        min_deg = min(bin(adj[v]).count('1') for v in range(N))
        if min_deg < 2:
            continue

        # Check connectivity (necessary for Hamilton cycle)
        # BFS from 0
        seen = 1
        stack = [0]
        while stack:
            cur = stack.pop()
            nbrs = adj[cur] & ~seen
            nm = nbrs
            while nm:
                lsb = nm & (-nm)
                w = lsb.bit_length() - 1
                seen |= (1 << w)
                stack.append(w)
                nm ^= lsb
        if seen != (1 << N) - 1:
            continue

        # Now try to find Hamilton cycle
        result = _has_ham_cycle(adj, N, deadline)
        if result is not None:
            best_cycle = result
            best_k = k
            break

    if best_cycle is not None:
        return best_cycle
    else:
        # Fallback: return identity (shouldn't happen if we have time)
        return list(range(N))
# EVOLVE-BLOCK-END
