"""Solver for few-color graph coloring with no bicolored cycles."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

class RollbackUF:
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.history = []

    def _add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        self._add(x)
        # Path compression is tricky with rollback, so we use path halving or just simple find
        # For small N, simple find is fine.
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        return root

    def union(self, a, b):
        self._add(a)
        self._add(b)
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False # Cycle
        # Union by rank
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        # Record state for rollback
        # We need to restore parent[rb] and rank[ra]
        self.history.append((rb, self.parent[rb], ra, self.rank[ra]))
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def rollback(self, count):
        for _ in range(count):
            rb, old_parent_rb, ra, old_rank_ra = self.history.pop()
            self.parent[rb] = old_parent_rb
            self.rank[ra] = old_rank_ra

def _check_bicolored_cycles_full(coloring, adj):
    """Full check: verify no bicolored cycles using union-find per color pair."""
    n = P['n']
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    
    pair_edges = defaultdict(list)
    for u, v in P['edges']:
        cu, cv = coloring[u], coloring[v]
        if cu < cv:
            pair_edges[(cu, cv)].append((u, v))
        elif cv < cu:
            pair_edges[(cv, cu)].append((u, v))
    
    for (c1, c2), edges in pair_edges.items():
        uf = RollbackUF()
        for u, v in edges:
            if not uf.union(u, v):
                return False
    return True

def _backtrack_coloring(adj, num_colors, deadline, order=None):
    n = P['n']
    coloring = [-1] * n
    
    if order is None:
        # Order by degree descending, then by index
        order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    
    # UFs for each color pair
    ufs = {}
    def get_uf(c1, c2):
        if c1 > c2: c1, c2 = c2, c1
        if (c1, c2) not in ufs:
            ufs[(c1, c2)] = RollbackUF()
        return ufs[(c1, c2)]

    def dfs(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return coloring[:]
        
        v = order[idx]
        used_by_neighbors = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used_by_neighbors.add(coloring[nb])
        
        for c in range(num_colors):
            if c in used_by_neighbors:
                continue
            
            coloring[v] = c
            # Check and update UFs
            valid = True
            changes = [] # List of (uf_key, num_unions)
            
            # We need to check all neighbors
            # Group neighbors by color to handle multiple unions efficiently?
            # Actually, just iterate neighbors.
            neighbor_colors = defaultdict(list)
            for nb in adj[v]:
                if coloring[nb] != -1:
                    neighbor_colors[coloring[nb]].append(nb)
            
            for cn, nbs in neighbor_colors.items():
                key = (c, cn) if c < cn else (cn, c)
                uf = get_uf(key)
                unions_for_this_pair = 0
                for nb in nbs:
                    if uf.union(v, nb):
                        unions_for_this_pair += 1
                    else:
                        # Cycle found
                        # Rollback unions for this pair done so far
                        uf.rollback(unions_for_this_pair)
                        # Rollback previous pairs?
                        # We need to rollback all changes made in this step
                        valid = False
                        break
                if valid:
                    if unions_for_this_pair > 0:
                        changes.append((key, unions_for_this_pair))
                else:
                    break
            
            if valid:
                result = dfs(idx + 1)
                if result is not None:
                    return result
            
            # Rollback all changes
            for key, count in changes:
                ufs[key].rollback(count)
            coloring[v] = -1
        
        return None

    return dfs(0)

def construct():
    deadline = time.time() + 85
    adj = _build_adj()
    
    for nc in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        result = _backtrack_coloring(adj, nc, deadline)
        if result is not None:
            if _check_bicolored_cycles_full(result, adj):
                return [int(x) for x in result]
    
    return [int(i) for i in range(P['n'])]
