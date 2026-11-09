"""Minimize distinct colors for degree-4 graph with bicolored-forest constraint.
Uses MRV backtracking with incremental union-find and undo journal."""
import time
import random
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']
_ADJ = [[] for _ in range(_N)]
for _a, _b in _EDGES:
    _ADJ[_a].append(_b)
    _ADJ[_b].append(_a)


def _bfs_2color():
    """Return proper 2-coloring if bipartite, else None."""
    color = [-1] * _N
    for start in range(_N):
        if color[start] != -1:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            cu = color[u]
            for v in _ADJ[u]:
                if color[v] == -1:
                    color[v] = 1 - cu
                    q.append(v)
                elif color[v] == cu:
                    return None
    return color


def _is_valid_full(coloring):
    """Check proper coloring and bicolored-forest constraint for a complete coloring."""
    for a, b in _EDGES:
        if coloring[a] == coloring[b]:
            return False
    present = sorted(set(coloring))
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            ci, cj = present[i], present[j]
            parent = list(range(_N))
            rank = [0] * _N

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True

            for a, b in _EDGES:
                ca, cb = coloring[a], coloring[b]
                if ca in (ci, cj) and cb in (ci, cj):
                    if not union(a, b):
                        return False
    return True


def _backtrack_mrv(k, deadline):
    """MRV backtracking with incremental union-find and undo journal."""
    color = [-1] * _N
    # Track which colors are used by each vertex's neighbors
    # For each color pair, track the union-find state
    # We'll use a simpler approach: maintain per-color-pair UF with undo

    # Precompute: for each vertex, list of edges incident to it
    # adj_edges[v] = list of (u, edge_index) for edges (v,u)
    
    # Incremental UF per color pair
    # Since k is small (3-6), we can maintain UF for each pair
    # But we need undo capability
    
    # Alternative: just do full check at each step but with MRV ordering
    # and forward checking
    
    # MRV: choose uncolored vertex with fewest available colors
    
    # State: color array
    # For each uncolored vertex, compute available colors
    
    def get_available_colors(v, color):
        """Return list of colors available for vertex v."""
        used = set()
        for nb in _ADJ[v]:
            if color[nb] != -1:
                used.add(color[nb])
        return [c for c in range(k) if c not in used]
    
    def check_bicolored_after_assign(v, c, color):
        """After assigning color c to vertex v, check all color pairs involving c."""
        # Get all colors in v's neighborhood
        neighbor_colors = set()
        for nb in _ADJ[v]:
            if color[nb] != -1:
                neighbor_colors.add(color[nb])
        
        # For each color d in neighbor_colors (d != c), check {c, d} pair
        for d in neighbor_colors:
            if d == c:
                continue
            ci, cj = min(c, d), max(c, d)
            
            # Check if adding v creates a cycle in {ci, cj} subgraph
            # v has color ci (or cj). Its neighbors with the other color
            # must not be connected in the {ci, cj} subgraph.
            
            # Find neighbors of v with color d
            neighbors_with_d = [nb for nb in _ADJ[v] if color[nb] == d]
            
            if len(neighbors_with_d) < 2:
                continue
            
            # Check if any two neighbors_with_d are connected in {ci, cj} subgraph
            # BFS from each neighbor, restricted to vertices with color ci or cj
            visited = set()
            for start in neighbors_with_d:
                if start in visited:
                    continue
                q = deque([start])
                visited.add(start)
                while q:
                    curr = q.popleft()
                    for nxt in _ADJ[curr]:
                        if color[nxt] in (ci, cj):
                            if nxt in neighbors_with_d and nxt != start:
                                return False  # Cycle detected
                            if nxt not in visited:
                                visited.add(nxt)
                                q.append(nxt)
        return True
    
    def mrv_backtrack(pos, color, deadline):
        if time.time() > deadline:
            return False
        
        if pos == _N:
            return True
        
        # MRV: find uncolored vertex with fewest available colors
        best_v = -1
        best_avail = None
        best_count = k + 1
        
        for v in range(_N):
            if color[v] != -1:
                continue
            avail = get_available_colors(v, color)
            if len(avail) < best_count:
                best_count = len(avail)
                best_v = v
                best_avail = avail
                if len(avail) == 0:
                    break  # Dead end
                if len(avail) == 1:
                    break  # Can't do better than 1
        
        if best_v == -1:
            return False  # No uncolored vertices but pos != _N shouldn't happen
        
        if best_count == 0:
            return False  # Dead end
        
        # Try each available color for best_v
        for c in best_avail:
            color[best_v] = c
            
            if check_bicolored_after_assign(best_v, c, color):
                if mrv_backtrack(pos + 1, color, deadline):
                    return True
            
            color[best_v] = -1
        
        return False
    
    if mrv_backtrack(0, color, deadline):
        return color[:]
    return None


def _greedy_color(k, order=None):
    """Greedy coloring with k colors."""
    if order is None:
        order = sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v))
    coloring = [-1] * _N
    for v in order:
        used = set()
        for nb in _ADJ[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        for c in range(k):
            if c not in used:
                coloring[v] = c
                break
        if coloring[v] == -1:
            return None
    return coloring


def construct():
    deadline = time.time() + 80.0

    # 1) Check bipartite (2 colors)
    c2 = _bfs_2color()
    if c2 is not None and _is_valid_full(c2):
        return [int(x) for x in c2]

    # 2) Try k=3,4,5,... with greedy first, then MRV backtracking
    best = None
    for k in range(3, 12):
        if time.time() > deadline - 5.0:
            break
        
        # Try greedy with multiple orderings
        rng = random.Random(42 + k)
        found = None
        
        # Standard ordering
        order1 = sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v))
        c = _greedy_color(k, order1)
        if c is not None and _is_valid_full(c):
            found = c
        
        if found is None:
            for attempt in range(5):
                if time.time() > deadline - 3.0:
                    break
                order = list(range(_N))
                rng.shuffle(order)
                c = _greedy_color(k, order)
                if c is not None and _is_valid_full(c):
                    found = c
                    break
        
        if found is None:
            # Try MRV backtracking
            remaining = deadline - time.time() - 2.0
            if remaining > 1.0:
                found = _backtrack_mrv(k, deadline - 2.0)
        
        if found is not None:
            best = found
            break

    # 3) Fallback: greedy with more colors
    if best is None:
        for k in range(5, 15):
            if time.time() > deadline - 2.0:
                break
            c = _greedy_color(k)
            if c is not None and _is_valid_full(c):
                best = c
                break

    # 4) Last resort: unique colors
    if best is None:
        best = list(range(_N))

    return [int(x) for x in best]
