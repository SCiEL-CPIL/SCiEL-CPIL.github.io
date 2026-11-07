"""CSP-based approach with MRV heuristic, forward checking, and incremental union-find.
Solves the bicolored-forest constraint via backtracking with aggressive pruning."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency
adj = [[] for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)


def _check_coloring(coloring, edges):
    """Validate: adjacent differ, and each color-pair subgraph is a forest."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    # Union-find per color pair
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


class UF:
    """Lightweight union-find for a fixed set of n vertices."""
    __slots__ = ('parent', 'rank')
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        r = self.rank
        if r[rx] < r[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if r[rx] == r[ry]:
            r[rx] += 1
        return True
    def snapshot(self):
        return (self.parent[:], self.rank[:])
    def restore(self, snap):
        self.parent = snap[0]
        self.rank = snap[1]


def _solve_csp(k, deadline):
    """CSP backtracking with MRV, forward checking, and incremental per-pair UF."""
    n = N
    coloring = [-1] * n
    domains = [set(range(k)) for _ in range(n)]
    
    # Per-pair union-find: key -> UF
    pair_uf = {}
    # For each pair, track which edges have been added (to know what to undo)
    pair_edges_added = {}  # key -> list of (u, v)
    
    # Track how many times each color is used (for symmetry breaking)
    color_counts = [0] * k
    
    # Precompute: for each vertex, its neighbors
    # MRV: pick uncolored vertex with smallest domain
    
    def get_uf(c1, c2):
        key = (min(c1, c2), max(c1, c2))
        if key not in pair_uf:
            pair_uf[key] = UF(n)
            pair_edges_added[key] = []
        return pair_uf[key], key
    
    def try_assign(v, c):
        """Try to assign color c to vertex v. Returns (ok, undo_info)."""
        # Check adjacency
        for nb in adj[v]:
            if coloring[nb] == c:
                return False, None
        
        # Collect affected pairs and snapshot
        affected = []  # list of (key, uf, snap, edges_before)
        for nb in adj[v]:
            cn = coloring[nb]
            if cn != -1 and cn != c:
                key = (min(cn, c), max(cn, c))
                if key not in [a[0] for a in affected]:
                    if key in pair_uf:
                        snap = pair_uf[key].snapshot()
                        edges_before = len(pair_edges_added[key])
                        affected.append((key, pair_uf[key], snap, edges_before))
                    else:
                        affected.append((key, None, None, 0))
        
        # Try unions
        valid = True
        for nb in adj[v]:
            cn = coloring[nb]
            if cn == -1 or cn == c:
                continue
            key = (min(cn, c), max(cn, c))
            uf, _ = get_uf(cn, c)
            if not uf.union(v, nb):
                valid = False
                break
            pair_edges_added[key].append((v, nb))
        
        if not valid:
            # Rollback the unions we just did
            for key, uf_obj, snap, edges_before in affected:
                if uf_obj is not None:
                    uf_obj.restore(snap)
                    del pair_edges_added[key][edges_before:]
                else:
                    if key in pair_uf:
                        del pair_uf[key]
                        del pair_edges_added[key]
            return False, None
        
        return True, (affected, color_counts)
    
    def undo_assign(v, c, undo_info):
        """Undo assignment."""
        affected, _ = undo_info
        for key, uf_obj, snap, edges_before in affected:
            if uf_obj is not None:
                uf_obj.restore(snap)
                del pair_edges_added[key][edges_before:]
            else:
                if key in pair_uf and not pair_edges_added.get(key):
                    del pair_uf[key]
                    del pair_edges_added[key]
                elif key in pair_edges_added:
                    del pair_edges_added[key][edges_before:]
        coloring[v] = -1
        color_counts[c] -= 1
    
    def forward_check(v, c):
        """After assigning v=c, check if any uncolored neighbor has empty domain."""
        for nb in adj[v]:
            if coloring[nb] == -1:
                # Remove c from nb's domain (it can't have same color as neighbor)
                # Actually, we check: can nb still be colored?
                # nb's domain must have at least one color not in {c} and not equal to any colored neighbor
                # Simplified: just check that nb has at least one available color
                avail = 0
                for cc in range(k):
                    if cc == c:
                        continue
                    # Check if cc conflicts with any other colored neighbor of nb
                    conflict = False
                    for nb2 in adj[nb]:
                        if coloring[nb2] == cc:
                            conflict = True
                            break
                    if not conflict:
                        avail += 1
                        break
                if avail == 0:
                    return False
        return True
    
    # Symmetry breaking: vertex 0 gets color 0
    coloring[0] = 0
    color_counts[0] = 1
    domains[0] = {0}
    # Remove 0 from neighbors of 0
    for nb in adj[0]:
        if 0 in domains[nb]:
            domains[nb].discard(0)
    
    nodes_visited = [0]
    
    def backtrack():
        if time.time() > deadline:
            return None
        nodes_visited[0] += 1
        if nodes_visited[0] % 100000 == 0:
            pass  # could log
        
        # MRV: find uncolored vertex with smallest domain
        min_size = k + 1
        best_v = -1
        best_domain = None
        for v in range(n):
            if coloring[v] == -1:
                # Compute effective domain
                eff = []
                for c in range(k):
                    # c must not be used by any colored neighbor
                    conflict = False
                    for nb in adj[v]:
                        if coloring[nb] == c:
                            conflict = True
                            break
                    if not conflict:
                        eff.append(c)
                if len(eff) < min_size:
                    min_size = len(eff)
                    best_v = v
                    best_domain = eff
                    if min_size == 0:
                        return None
                    if min_size == 1:
                        break
        
        if best_v == -1:
            # All colored
            return coloring[:]
        
        # Try each color in best_domain
        # Order: try colors that are already used first (symmetry)
        used_colors = [c for c in range(k) if color_counts[c] > 0]
        new_colors = [c for c in best_domain if color_counts[c] == 0]
        # Only try the first new color (symmetry breaking)
        ordered = [c for c in best_domain if c in used_colors]
        if new_colors:
            ordered.append(new_colors[0])
        
        for c in ordered:
            if c not in best_domain:
                continue
            # Forward check
            if not forward_check(best_v, c):
                continue
            
            ok, undo_info = try_assign(best_v, c)
            if ok:
                coloring[best_v] = c
                color_counts[c] += 1
                
                result = backtrack()
                if result is not None:
                    return result
                
                undo_assign(best_v, c, undo_info)
            
            if time.time() > deadline:
                return None
        
        return None
    
    result = backtrack()
    if result is not None:
        # Verify
        if _check_coloring(result, EDGES):
            return result
    return None


def _local_reduce(colors, deadline):
    """Try to reduce color count by recoloring vertices to lower colors."""
    n = len(colors)
    current = list(colors)
    current_k = max(current) + 1
    
    for target in range(current_k - 1, 0, -1):
        if time.time() > deadline:
            break
        verts = [i for i in range(n) if current[i] == target]
        if not verts:
            current_k = target
            continue
        
        all_ok = True
        for v in verts:
            found = False
            for c in range(target):
                # Check adjacency
                if any(current[nb] == c for nb in adj[v]):
                    continue
                trial = list(current)
                trial[v] = c
                if _check_coloring(trial, EDGES):
                    current = trial
                    found = True
                    break
            if not found:
                all_ok = False
                break
        
        if all_ok:
            current_k = target
    
    return current, current_k


def construct():
    start = time.time()
    deadline = start + 80  # Safety margin before 90s limit
    
    # Try k from 3 upward
    for k in range(3, 15):
        if time.time() > deadline - 5:
            break
        
        result = _solve_csp(k, deadline)
        if result is not None:
            # Try to reduce
            reduced, rk = _local_reduce(result, deadline)
            if _check_coloring(reduced, EDGES):
                result = reduced
            # Renumber
            distinct = sorted(set(result))
            mapping = {c: i for i, c in enumerate(distinct)}
            final = [int(mapping[c]) for c in result]
            if _check_coloring(final, EDGES):
                return final
            return [int(c) for c in result]
    
    # Fallback: greedy coloring with many colors
    coloring = [-1] * N
    for v in range(N):
        used = set(coloring[nb] for nb in adj[v] if coloring[nb] != -1)
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
    return [int(c) for c in coloring]
