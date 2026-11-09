import time
import sys
from collections import deque

# Precompute edge positions
N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge position map: (min(u,v), max(u,v)) -> 1-based position
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def get_pos(u, v):
    a, b = min(u, v), max(u, v)
    return edge_pos[(a, b)]

def cycle_score(cycle):
    """Max 1-based arrival index of the cycle's edges (with implicit closing edge)."""
    m = len(cycle)
    best = 0
    for i in range(m):
        u = cycle[i]
        v = cycle[(i + 1) % m]
        p = get_pos(u, v)
        if p > best:
            best = p
    return best

# ---------------------------------------------------------------------------
# Fast 2-factor check using Hopcroft-Karp (only called O(log N) times)
# ---------------------------------------------------------------------------
def find_2factor(adj, n):
    """Return a 2-factor as successor list (matchL) or None.
    Uses Hopcroft-Karp perfect matching on bipartite graph."""
    # Quick degree check
    for v in range(n):
        if len(adj[v]) < 2:
            return None

    matchL = [-1] * n
    matchR = [-1] * n
    dist = [-1] * n

    def bfs():
        q = deque()
        found = False
        for u in range(n):
            if matchL[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = matchR[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = matchR[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                matchL[u] = v
                matchR[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if matchL[u] == -1 and dfs(u):
                matching += 1
    if matching != n:
        return None
    return matchL

def extract_cycles_from_2factor(matchL, n):
    """Extract cycles from a 2-factor successor list."""
    visited = [False] * n
    cycles = []
    for s in range(n):
        if not visited[s]:
            cyc = []
            u = s
            while not visited[u]:
                visited[u] = True
                cyc.append(u)
                u = matchL[u]
            cycles.append(cyc)
    return cycles

# ---------------------------------------------------------------------------
# Fast 2-opt with precomputed edge positions
# ---------------------------------------------------------------------------
def two_opt_refine(cycle, n, max_iter=5000, deadline=None):
    """Improve a Hamilton cycle by 2-opt to minimize max edge position."""
    cycle = cycle[:]
    m = len(cycle)

    # Precompute edge positions for current cycle
    edge_positions = [get_pos(cycle[i], cycle[(i + 1) % m]) for i in range(m)]

    def compute_max(pos_list):
        return max(pos_list)

    cur = compute_max(edge_positions)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        if deadline is not None and time.time() > deadline:
            break
        for i in range(m - 1):
            for j in range(i + 2, m):
                if j == m - 1 and i == 0:
                    continue
                a, b = cycle[i], cycle[i + 1]
                c, d = cycle[j], cycle[(j + 1) % m]
                for (e1, e2) in [((a, c), (b, d)), ((a, d), (b, c))]:
                    p1 = get_pos(e1[0], e1[1])
                    p2 = get_pos(e2[0], e2[1])
                    # Quick check: if both new edges are >= cur, no improvement
                    if p1 >= cur and p2 >= cur:
                        continue
                    # Compute new max: max of all edges except removed ones, plus new ones
                    # Remove edges at positions i and j
                    newmax = 0
                    for k in range(m):
                        if k == i or k == j:
                            continue
                        if edge_positions[k] > newmax:
                            newmax = edge_positions[k]
                    if p1 > newmax:
                        newmax = p1
                    if p2 > newmax:
                        newmax = p2
                    if newmax < cur:
                        # Apply 2-opt: reverse segment i+1..j
                        cycle[i + 1:j + 1] = reversed(cycle[i + 1:j + 1])
                        cur = newmax
                        # Recompute edge positions
                        edge_positions = [get_pos(cycle[k], cycle[(k + 1) % m]) for k in range(m)]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return cycle

# ---------------------------------------------------------------------------
# Backtracking Hamilton cycle finder with Warnsdorff heuristic
# ---------------------------------------------------------------------------
def find_hamilton_cycle(adj, n, deadline, start=0):
    """Find a Hamilton cycle in the given adjacency list using backtracking."""
    sys.setrecursionlimit(100000)
    
    # Sort neighbors by degree (Warnsdorff) then by edge position
    neigh = []
    for v in range(n):
        lst = list(adj[v])
        lst.sort(key=lambda x: (len(adj[x]), get_pos(v, x)))
        neigh.append(lst)

    path = [start]
    used = [False] * n
    used[start] = True

    def bt():
        if time.time() > deadline:
            return None
        if len(path) == n:
            last = path[-1]
            if start in adj[last]:
                return path[:]
            return None
        last = path[-1]
        for nxt in neigh[last]:
            if not used[nxt]:
                # Pruning: if 2 vertices remain, check feasibility
                if len(path) == n - 2:
                    remaining = [v for v in range(n) if not used[v]]
                    r = remaining[0]
                    if r not in adj[nxt] or start not in adj[r]:
                        continue
                used[nxt] = True
                path.append(nxt)
                res = bt()
                if res is not None:
                    return res
                path.pop()
                used[nxt] = False
        return None

    return bt()

# ---------------------------------------------------------------------------
# Merge two cycles from a 2-factor using edge swaps
# ---------------------------------------------------------------------------
def merge_two_cycles(A, B, adj):
    """Try to merge two cycles by swapping two edges. Returns merged cycle or None."""
    la, lb = len(A), len(B)
    for ia in range(la):
        a1, a2 = A[ia], A[(ia + 1) % la]
        for ib in range(lb):
            b1, b2 = B[ib], B[(ib + 1) % lb]
            # Option 1: (a1,b1) and (a2,b2)
            if b1 in adj[a1] and b2 in adj[a2]:
                # Build merged cycle: A from a1 to a2, then B from b1 to b2
                newcyc = []
                idx = ia
                while True:
                    newcyc.append(A[idx])
                    if A[idx] == a2:
                        break
                    idx = (idx + 1) % la
                idx = ib
                while True:
                    newcyc.append(B[idx])
                    if B[idx] == b2:
                        break
                    idx = (idx + 1) % lb
                return newcyc
            # Option 2: (a1,b2) and (a2,b1)
            if b2 in adj[a1] and b1 in adj[a2]:
                newcyc = []
                idx = ia
                while True:
                    newcyc.append(A[idx])
                    if A[idx] == a2:
                        break
                    idx = (idx + 1) % la
                idx = ib
                while True:
                    newcyc.append(B[idx])
                    if B[idx] == b1:
                        break
                    idx = (idx + 1) % lb
                return newcyc
    return None

def merge_all_cycles(cycles, adj, n, deadline):
    """Merge all cycles in a 2-factor into one Hamilton cycle."""
    if len(cycles) == 1:
        return cycles[0][:]
    
    cur_cycles = [c[:] for c in cycles]
    changed = True
    while changed and len(cur_cycles) > 1 and time.time() < deadline:
        changed = False
        for i in range(len(cur_cycles)):
            for j in range(i + 1, len(cur_cycles)):
                merged = merge_two_cycles(cur_cycles[i], cur_cycles[j], adj)
                if merged is not None:
                    cur_cycles[i] = merged
                    del cur_cycles[j]
                    changed = True
                    break
            if changed:
                break
    if len(cur_cycles) == 1:
        return cur_cycles[0][:]
    return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n = N
    total_edges = n * (n - 1) // 2  # 276

    # Build full adjacency
    full_adj = [[] for _ in range(n)]
    for (u, v) in EDGE_ORDER:
        full_adj[u].append(v)
        full_adj[v].append(u)

    best_cycle = None
    best_score = total_edges + 1

    # Binary search for minimal T where 2-factor exists
    lo, hi = n, total_edges
    feasible_T = total_edges

    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        adj = [[] for _ in range(n)]
        for (u, v), p in edge_pos.items():
            if p <= mid:
                adj[u].append(v)
                adj[v].append(u)
        matchL = find_2factor(adj, n)
        if matchL is not None:
            feasible_T = mid
            hi = mid - 1
        else:
            lo = mid + 1

    # Now try to find actual Hamilton cycles starting from feasible_T
    for T in range(feasible_T, min(feasible_T + 20, total_edges + 1)):
        if time.time() > deadline:
            break
        
        adj = [[] for _ in range(n)]
        for (u, v), p in edge_pos.items():
            if p <= T:
                adj[u].append(v)
                adj[v].append(u)

        # Strategy 1: Direct backtracking
        cycle = find_hamilton_cycle(adj, n, deadline)
        if cycle is not None:
            cycle = two_opt_refine(cycle, n, max_iter=3000, deadline=deadline)
            s = cycle_score(cycle)
            if s < best_score:
                best_score = s
                best_cycle = cycle
            if best_score <= T:
                break  # Can't do better than T for this threshold

        # Strategy 2: 2-factor + merge
        if time.time() > deadline:
            break
        matchL = find_2factor(adj, n)
        if matchL is not None:
            cycles = extract_cycles_from_2factor(matchL, n)
            if len(cycles) == 1:
                # Already a Hamilton cycle!
                c = cycles[0][:]
                s = cycle_score(c)
                if s < best_score:
                    best_score = s
                    best_cycle = c
            elif len(cycles) > 1:
                merged = merge_all_cycles(cycles, adj, n, deadline)
                if merged is not None:
                    merged = two_opt_refine(merged, n, max_iter=2000, deadline=deadline)
                    s = cycle_score(merged)
                    if s < best_score:
                        best_score = s
                        best_cycle = merged

        # Early exit if we hit the theoretical minimum for this T
        if best_score <= T:
            break

    # Final polish
    if best_cycle is not None and time.time() < deadline:
        best_cycle = two_opt_refine(best_cycle, n, max_iter=10000, deadline=deadline)
        s = cycle_score(best_cycle)
        if s < best_score:
            best_score = s

    if best_cycle is not None:
        return best_cycle
    return list(range(n))
