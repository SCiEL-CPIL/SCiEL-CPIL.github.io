"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']

adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def solve_csp(k, deadline):
    """Backtracking CSP solver for k-coloring with bicolored-forest constraint."""
    coloring = [-1] * n
    # For each pair (c1, c2), maintain union-find to detect cycles
    # We'll use a dict mapping (c1,c2) -> parent array
    # But that's expensive. Instead, track edges per pair and check connectivity.
    
    # Alternative: maintain for each color pair a set of edges, and check if adding
    # an edge creates a cycle using DFS/BFS on the induced subgraph.
    
    # Better: use union-find per color pair, but only for pairs that are active.
    
    # For efficiency, let's use a different approach:
    # When assigning color c to vertex v, for each neighbor u with color d != c:
    #   The edge (u,v) goes into pair (c,d). Check if it creates a cycle.
    # To check cycle: in the subgraph induced by colors {c,d}, is there already a path from u to v?
    
    # We can maintain adjacency lists per color pair for the induced subgraph.
    
    # pair_adj[(c1,c2)] = list of edges in that pair's subgraph
    # To check cycle: when adding edge (u,v) to pair (c1,c2), check if u and v are already connected in that subgraph.
    
    # Use union-find per pair, but only create when needed.
    
    sys.setrecursionlimit(10000)
    
    # State: coloring, and for each pair (a,b), a union-find structure
    # We'll use a dict: uf[(a,b)] = parent list
    # But copying is expensive. Let's use a different approach.
    
    # Alternative: for each color pair, maintain a set of edges and a DSU.
    # When backtracking, we need to undo. Use a stack of operations.
    
    # Let's use a simpler but potentially slower approach:
    # For each assignment, check if it creates a cycle by doing BFS in the induced subgraph.
    
    def get_pair_edges(c1, c2, coloring):
        """Get all edges where both endpoints have color c1 or c2."""
        es = []
        for u, v in edges:
            cu, cv = coloring[u], coloring[v]
            if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                es.append((u, v))
        return es
    
    def has_cycle_in_pair(c1, c2, coloring):
        """Check if the subgraph induced by colors c1 and c2 has a cycle."""
        # Build adjacency for vertices with color c1 or c2
        in_pair = [False] * n
        for v in range(n):
            if coloring[v] == c1 or coloring[v] == c2:
                in_pair[v] = True
        
        # Build adjacency list
        adj_pair = [[] for _ in range(n)]
        for u, v in edges:
            if in_pair[u] and in_pair[v]:
                adj_pair[u].append(v)
                adj_pair[v].append(u)
        
        # BFS/DFS to detect cycle
        visited = [False] * n
        def dfs(u, parent):
            visited[u] = True
            for w in adj_pair[u]:
                if not visited[w]:
                    if dfs(w, u):
                        return True
                elif w != parent:
                    return True
            return False
        
        for v in range(n):
            if in_pair[v] and not visited[v]:
                if dfs(v, -1):
                    return True
        return False
    
    # This is too slow for backtracking. Let me use incremental approach.
    
    # Better approach: maintain for each color pair a DSU with rollback.
    
    class DSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
        
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return False  # cycle detected
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            return True
        
        def clone(self):
            d = DSU.__new__(DSU)
            d.parent = self.parent[:]
            d.rank = self.rank[:]
            return d
    
    # For backtracking, we need to save/restore DSU states.
    # This is expensive. Let's use a different strategy.
    
    # Strategy: Use a greedy + local search approach instead of full backtracking.
    # Or use backtracking with limited depth and good heuristics.
    
    # Let me try a different approach: 
    # 1. Start with a proper k-coloring (greedy or backtracking)
    # 2. Check bicolored forest constraint
    # 3. If violated, fix by recoloring
    
    # Actually, let's try a systematic backtracking with the constraint checked incrementally.
    # Key insight: when we assign color c to vertex v, we only need to check pairs (c, d) 
    # where d is a color already used by a neighbor of v.
    
    # For each such pair (c, d), we need to verify that adding edge (v, u) where u has color d
    # doesn't create a cycle in the subgraph induced by {c, d}.
    
    # To do this efficiently, maintain for each active pair (c1, c2) a DSU.
    # When assigning color c to v:
    #   For each neighbor u with color d != c:
    #     Check if u and v are already connected in DSU[(c,d)]
    #     If yes -> cycle -> invalid
    #     If no -> union u and v in DSU[(c,d)]
    #   Save the changes for backtracking
    
    # Let's implement this with explicit undo stack.
    
    # pairs: dict from frozenset({c1,c2}) to DSU
    # undo_stack: list of (pair_key, x, y, was_connected, old_parent, old_rank) for each union
    
    pairs = {}  # frozenset -> DSU
    undo_log = []  # for backtracking
    
    def get_dsu(c1, c2):
        key = frozenset((c1, c2))
        if key not in pairs:
            pairs[key] = DSU(n)
        return pairs[key], key
    
    def assign_color(v, c):
        """Assign color c to vertex v. Returns list of undo operations or None if invalid."""
        ops = []
        for u in adj[v]:
            d = coloring[u]
            if d == -1 or d == c:
                continue
            dsu, key = get_dsu(c, d)
            # Check if adding edge (u,v) creates a cycle
            if not dsu.union(u, v):
                # Cycle detected - undo all previous ops in this call
                for op in ops:
                    undo_dsu(op)
                return None
            ops.append((key, u, v))
        
        coloring[v] = c
        return ops
    
    def undo_dsu(op):
        key, u, v = op
        dsu = pairs[key]
        # We need to undo the union. This is tricky with path compression.
        # Simplification: don't use path compression, just track parent changes.
        # Actually, let's recompute: just find the parent of the root that was changed.
        # This is getting complicated. Let's use a simpler approach.
        pass
    
    # The undo is complex. Let me use a different strategy:
    # Instead of incremental DSU with undo, use a simpler check.
    
    # New approach: Backtracking with full validation at each step.
    # For a 50-vertex graph with k=3 or 4, this might be feasible with good pruning.
    
    # Let's try: backtracking with MRV and forward checking, 
    # checking bicolored constraint only at leaves or periodically.
    
    # Actually, the bicolored constraint is global, so we can't easily do forward checking.
    
    # Let me try a completely different approach:
    # 1. Find a proper k-coloring using standard backtracking
    # 2. Then check if it satisfies bicolored-forest
    # 3. If not, try to fix by recoloring vertices
    
    # For step 1, standard graph coloring backtracking is fast for k>=3 on 4-regular graphs.
    
    def proper_coloring_backtrack(k, order):
        """Find a proper k-coloring using backtracking."""
        coloring = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(k):
                valid = True
                for u in adj[v]:
                    if coloring[u] == c:
                        valid = False
                        break
                if valid:
                    coloring[v] = c
                    if backtrack(idx + 1):
                        return True
                    coloring[v] = -1
            return False
        
        if backtrack(0):
            return coloring
        return None
    
    # Try to find proper colorings and check bicolored constraint
    # Use different vertex orderings for diversity
    
    deadline_check = lambda: time.time() > deadline
    
    # First, try a simple approach: find proper colorings with k colors,
    # then check bicolored forest constraint.
    
    import random
    
    for k_try in range(3, min(k + 1, 6)):
        if deadline_check():
            break
        
        # Try multiple random orderings
        for attempt in range(200):
            if deadline_check():
                break
            
            # Create ordering: try different strategies
            if attempt % 3 == 0:
                order = list(range(n))
                random.shuffle(order)
            elif attempt % 3 == 1:
                # Degree-based ordering (all same degree here, so just reverse)
                order = list(range(n))
            else:
                # Random
                order = list(range(n))
                random.shuffle(order)
            
            coloring = proper_coloring_backtrack(k_try, order)
            if coloring is None:
                continue
            
            # Check bicolored forest constraint
            valid = True
            for c1 in range(k_try):
                for c2 in range(c1 + 1, k_try):
                    if has_cycle_in_pair(c1, c2, coloring):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Count actual colors used
                colors_used = set(coloring)
                actual_k = len(colors_used)
                # Remap to 0..actual_k-1
                remap = {c: i for i, c in enumerate(sorted(colors_used))}
                new_coloring = [remap[c] for c in coloring]
                return new_coloring, actual_k
    
    return None, None


def construct():
    deadline = 80.0
    start = time.time()
    
    # Try k=3, 4, 5 in order
    for k in range(3, 6):
        if time.time() - start > deadline:
            break
        
        result, k_used = solve_csp(k, deadline)
        if result is not None:
            # Verify
            coloring = result
            # Check proper
            for u, v in P['edges']:
                if coloring[u] == coloring[v]:
                    break
            else:
                # Check bicolored forest
                max_c = max(coloring) + 1
                valid = True
                for c1 in range(max_c):
                    for c2 in range(c1 + 1, max_c):
                        # Check cycle
                        in_pair = [False] * n
                        for v in range(n):
                            if coloring[v] == c1 or coloring[v] == c2:
                                in_pair[v] = True
                        adj_pair = [[] for _ in range(n)]
                        for u, v in P['edges']:
                            if in_pair[u] and in_pair[v]:
                                adj_pair[u].append(v)
                                adj_pair[v].append(u)
                        visited = [False] * n
                        def dfs_cycle(u, parent):
                            visited[u] = True
                            for w in adj_pair[u]:
                                if not visited[w]:
                                    if dfs_cycle(w, u):
                                        return True
                                elif w != parent:
                                    return True
                            return False
                        for v in range(n):
                            if in_pair[v] and not visited[v]:
                                if dfs_cycle(v, -1):
                                    valid = False
                                    break
                        if not valid:
                            break
                if valid:
                    return [int(c) for c in coloring]
    
    # Fallback: return a valid coloring (might use more colors)
    # Use greedy with many colors to ensure validity
    coloring = [-1] * n
    for v in range(n):
        used = set()
        for u in adj[v]:
            if coloring[u] != -1:
                used.add(coloring[u])
        c = 0
        while c in used:
            c += 1
        coloring[v] = c
    
    # This might not satisfy bicolored constraint. 
    # Use enough colors that no bicolored cycle can form.
    # If we use n colors (each vertex unique), no bicolored cycle is possible.
    # But we want to minimize. Let's try a smarter fallback.
    
    # Actually, with 50 vertices and 4-regular, let's just use a proper coloring
    # with enough colors to guarantee no bicolored cycles.
    # If each color class is an independent set and small enough...
    
    # Safest fallback: assign each vertex a unique color
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
