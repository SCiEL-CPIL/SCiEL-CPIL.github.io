"""Find minimal number of colors such that each pair of colors induces a forest.
Uses backtracking with incremental bicolored cycle detection via union-find per color pair."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _check_coloring(coloring, edges):
    """Check if coloring is valid: adjacent differ, and no bicolored cycle."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
            pair_edges[pair].append((u, v))
    
    for pair, elist in pair_edges.items():
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def _find_coloring_with_k_colors(k, edges, n, deadline):
    """Try to find a valid coloring with k colors. Returns coloring or None."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices by degree (higher first)
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos_to_vertex = order
    vertex_to_pos = {v: i for i, v in enumerate(order)}
    
    coloring = [-1] * n
    # For each pair of colors, maintain a UnionFind
    # pair_ufs: dict from (c1, c2) to UnionFind
    pair_ufs = {}
    # Track which vertices have been assigned
    assigned = [False] * n
    
    # For each color, track which vertices have that color
    color_vertices = [set() for _ in range(k)]
    
    def get_uf(c1, c2):
        pair = (min(c1, c2), max(c1, c2))
        if pair not in pair_ufs:
            pair_ufs[pair] = UnionFind()
        return pair_ufs[pair]
    
    def assign(v, c):
        """Assign color c to vertex v. Returns False if invalid."""
        # Check adjacency constraint
        for nb in adj[v]:
            if coloring[nb] == c:
                return False
        
        coloring[v] = c
        assigned[v] = True
        color_vertices[c].add(v)
        
        # Check bicolored cycles for each neighbor with a different color
        for nb in adj[v]:
            if coloring[nb] == -1:
                continue
            d = coloring[nb]
            if d == c:
                continue
            uf = get_uf(c, d)
            if not uf.union(v, nb):
                # Cycle detected - undo
                coloring[v] = -1
                assigned[v] = False
                color_vertices[c].remove(v)
                return False
        return True
    
    def unassign(v, c):
        """Undo assignment of color c to vertex v."""
        coloring[v] = -1
        assigned[v] = False
        color_vertices[c].discard(v)
        # Note: we don't remove from union-find (it's monotonic), 
        # but this is fine because we're backtracking - the UF state is only
        # valid for the current path. We need to be careful here.
        # Actually, union-find is not easily reversible. We need a different approach.
        # Let's use a different strategy: don't maintain incremental UF, instead
        # check for cycles by BFS when needed.
        pass
    
    # The incremental union-find approach has issues with backtracking because
    # union-find is not easily reversible. Let's use a different approach:
    # At each step, after assigning a color, check if any bicolored cycle exists
    # by doing BFS in the subgraph of the affected color pairs.
    
    # Better approach: use a proper backtracking with cycle detection via BFS
    # only on the affected pairs.
    
    def check_bicolored_cycle_for_vertex(v, c):
        """Check if assigning color c to v creates a bicolored cycle.
        For each neighbor with a different color, check if there's a path from v
        to that neighbor in the subgraph of colors {c, neighbor_color} that doesn't
        use the direct edge."""
        for nb in adj[v]:
            if coloring[nb] == -1:
                continue
            d = coloring[nb]
            if d == c:
                return False  # Same color adjacent
            
            # BFS from v to nb in subgraph of colors {c, d}, excluding direct edge v-nb
            visited = set()
            queue = [v]
            visited.add(v)
            found = False
            while queue:
                cur = queue.pop(0)
                for w in adj[cur]:
                    if w in visited:
                        continue
                    if coloring[w] in (c, d):
                        # Skip the direct edge v-nb
                        if (cur == v and w == nb) or (cur == nb and w == v):
                            continue
                        if w == nb:
                            found = True
                            break
                        visited.add(w)
                        queue.append(w)
                if found:
                    break
            if found:
                return False  # Cycle found
        return True
    
    # Backtracking with BFS-based cycle check
    # This is slower but correct with backtracking
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        v = pos_to_vertex[pos]
        
        # Try colors in order
        for c in range(k):
            # Check if color c is valid for v
            valid = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            coloring[v] = c
            if check_bicolored_cycle_for_vertex(v, c):
                if backtrack(pos + 1):
                    return True
            coloring[v] = -1
        return False
    
    if backtrack(0):
        return coloring[:]
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, 5s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Try k = 2, 3, 4, ... until we find a valid coloring
    for k in range(2, 10):
        if time.time() > deadline - 15:
            break
        result = _find_coloring_with_k_colors(k, edges, n, deadline)
        if result is not None:
            # Verify
            if _check_coloring(result, edges):
                return [int(c) for c in result]
    
    # Fallback: use greedy with more colors
    # Simple approach: assign colors greedily, ensuring no bicolored cycles
    # With enough colors (e.g., 8+), this should be easy
    
    # Greedy with verification
    coloring = [0] * n
    for attempt in range(100):
        coloring = [0] * n
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        valid = True
        for v in order:
            # Find a color that doesn't conflict
            used_colors = set()
            for nb in adj[v]:
                if coloring[nb] != 0 or nb != v:
                    used_colors.add(coloring[nb])
            
            # Try colors 0, 1, 2, ...
            for c in range(50):
                if c not in used_colors:
                    # Check bicolored cycle
                    coloring[v] = c
                    ok = True
                    for nb in adj[v]:
                        if coloring[nb] == -1 or nb == v:
                            continue
                        d = coloring[nb]
                        if d == c:
                            ok = False
                            break
                        # Check for cycle in {c, d} subgraph
                        # BFS from v to nb excluding direct edge
                        visited = set()
                        queue = [v]
                        visited.add(v)
                        found_cycle = False
                        while queue:
                            cur = queue.pop(0)
                            for w in adj[cur]:
                                if w in visited:
                                    continue
                                if coloring[w] in (c, d) and coloring[w] != -1:
                                    if (cur == v and w == nb) or (cur == nb and w == v):
                                        continue
                                    if w == nb:
                                        found_cycle = True
                                        break
                                    visited.add(w)
                                    queue.append(w)
                            if found_cycle:
                                break
                        if found_cycle:
                            ok = False
                            break
                    if ok:
                        break
                    else:
                        coloring[v] = 0
                        continue
                else:
                    continue
            else:
                valid = False
                break
        
        if valid and _check_coloring(coloring, edges):
            return [int(c) for c in coloring]
    
    # Ultimate fallback: 50 distinct colors
    return list(range(n))
# EVOLVE-BLOCK-END
