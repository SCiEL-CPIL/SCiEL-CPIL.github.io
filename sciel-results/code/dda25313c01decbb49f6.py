import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = [set() for _ in range(N)]
for u, v in edges:
    adj[u].add(v)
    adj[v].add(u)

# Precompute edge list by color-pair for quick DSU checks
# We'll use a fast incremental approach

def _check_bicolored_cycles(colors):
    """Return set of color-pairs that have a bicolored cycle."""
    color_pairs = set()
    for u, v in edges:
        a, b = colors[u], colors[v]
        if a != b:
            color_pairs.add((min(a, b), max(a, b)))
    bad_pairs = set()
    for a, b in color_pairs:
        parent = list(range(N))
        def find(x):
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
        for u, v in edges:
            cu, cv = colors[u], colors[v]
            if (cu == a and cv == b) or (cu == b and cv == a):
                if not union(u, v):
                    bad_pairs.add((a, b))
                    break
    return bad_pairs

def _is_valid(colors):
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    return len(_check_bicolored_cycles(colors)) == 0

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Try 4-coloring with backtracking + forward checking + incremental DSU
    # Colors: 0, 1, 2, 3
    
    # Vertex ordering: use a simple order but prioritize high-degree / constrained vertices
    # All vertices have degree 4, so use a heuristic: process in order that maximizes constraint propagation
    # For circulant C(50,{1,7}), use modular ordering for symmetry breaking
    
    # Symmetry breaking: fix first few vertices to reduce equivalent solutions
    # v[0]=0, v[1]=1, v[7]=2 (since 0,1,7 are all adjacent)
    colors = [-1] * N
    colors[0] = 0
    colors[1] = 1
    colors[7] = 2
    
    # DSU state per color pair
    # We'll maintain DSU incrementally
    # For each color pair (a,b), track the DSU of vertices with those colors
    
    # For incremental DSU, we need to be able to undo. Use a stack-based approach.
    # Actually, let's use a simpler approach: at each step, check if adding this color
    # creates a bicolored cycle using the current partial assignment.
    
    # Order of assignment: use a greedy + backtracking hybrid
    # Process vertices in an order that creates constraints early
    
    # For C(50,{1,7}), the natural order is 0,1,2,...,49
    # But let's use a smarter order: interleave to catch cycles early
    # Actually, for backtracking, the order matters. Let's use:
    # 0, 1, 7, 2, 8, 14, 3, 9, 15, ... (spiral out)
    
    # Simpler: just use 0,1,2,...,49 with symmetry breaking
    order = list(range(N))
    # Remove already-assigned vertices from order
    assigned_set = {0, 1, 7}
    order = [v for v in order if v not in assigned_set]
    
    # For each color pair, maintain a DSU
    # We'll use a dictionary: pair -> list of parent arrays (for undo)
    # Actually, for efficiency, let's just re-check DSU at each step for the affected pairs
    
    # Better approach: maintain DSU incrementally with undo stack
    # Each DSU operation (union) is pushed onto a stack for that pair
    
    # Let me use a different strategy: 
    # At each vertex assignment, only the color pairs involving the new color and 
    # the colors of already-assigned neighbors can be affected.
    
    # For simplicity and correctness, I'll do:
    # 1. Assign a color to a vertex
    # 2. For each color pair (c_new, c_neighbor) where neighbor is already assigned,
    #    check if adding this edge creates a cycle in that pair's subgraph
    
    # DSU with path compression is hard to undo. Use union-by-rank with explicit stack.
    
    # Let me use a simpler but effective approach:
    # Backtracking with full DSU check only on the affected pairs.
    
    # Actually, the key insight from the cell finding: "Backtracking with forward checking 
    # using incremental DSU per color-pair prunes the search space dramatically"
    
    # Let me implement a clean backtracking:
    
    class DSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.stack = []  # for undo
        
        def find(self, x):
            # No path compression for undo-ability; use union by rank
            while self.parent[x] != x:
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.stack.append((ry, rx, self.rank[rx]))
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
        
        def undo(self):
            if self.stack:
                ry, rx, old_rank = self.stack.pop()
                self.parent[ry] = ry
                self.rank[rx] = old_rank
    
    # DSU for each color pair
    # Max colors = 4, so pairs: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
    pair_keys = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    pair_index = {p: i for i, p in enumerate(pair_keys)}
    dsus = [DSU(N) for _ in pair_keys]
    
    # Track which vertices have which color
    # For each pair, track edges that have been added
    
    def get_pair_key(a, b):
        if a > b:
            a, b = b, a
        return pair_index.get((a, b), -1)
    
    def try_assign(v, c):
        """Try to assign color c to vertex v. Returns list of DSU states that need undo."""
        undos = []
        ok = True
        for nb in adj[v]:
            if colors[nb] != -1:
                if colors[nb] == c:
                    return None  # adjacent same color
                pk = get_pair_key(c, colors[nb])
                if pk >= 0:
                    if not dsus[pk].union(v, nb):
                        return None  # would create bicolored cycle
                    undos.append((pk,))
        return undos
    
    def undo_assign(undos):
        for item in reversed(undos):
            pk = item[0]
            dsus[pk].undo()
    
    # Backtracking
    best_result = None
    nodes_explored = 0
    
    def backtrack(idx):
        nonlocal nodes_explored, best_result
        if time.time() > deadline:
            return False
        if idx == len(order):
            # All assigned, verify
            result = [int(colors[v]) for v in range(N)]
            if _is_valid(result):
                best_result = result
                return True
            return False
        
        v = order[idx]
        # Try colors 0,1,2,3
        for c in range(4):
            # Check if color c is available (no neighbor has color c)
            conflict = False
            for nb in adj[v]:
                if colors[nb] == c:
                    conflict = True
                    break
            if conflict:
                continue
            
            undos = try_assign(v, c)
            if undos is None:
                continue
            
            colors[v] = c
            nodes_explored += 1
            if backtrack(idx + 1):
                return True
            undo_assign(undos)
            colors[v] = -1
        
        return False
    
    # Run backtracking
    if backtrack(0):
        return best_result
    
    # If backtracking didn't finish in time, try a heuristic approach
    # Try a structured 4-coloring for C(50,{1,7})
    # Pattern: use (v mod 5) // 2 or similar
    # Actually, for C(50, {1,7}), a 4-coloring can be:
    # color(v) = (v // 5) % 4  -- no, this doesn't work well
    # 
    # Let's try: color(v) = (v % 5) % 4 -- no
    #
    # Better: exploit the circulant structure
    # C(50, {1,7}) has period 50. 
    # Try: color(v) = (v * 7) % 4 -- since 7*1=7≡3 mod 4, 7*7=49≡1 mod 4
    # Adjacent: v and v+1: colors differ by 3 mod 4 = -1, so different. Good.
    # v and v+7: colors differ by 49 mod 4 = 1, so different. Good.
    # But does it create bicolored cycles?
    
    colors_struct = [(v * 7) % 4 for v in range(N)]
    if _is_valid(colors_struct):
        return [int(c) for c in colors_struct]
    
    # Try another pattern: (v % 5) mapped to 4 colors
    # 50 vertices, 5 per "ring" if we think of it as 5x10
    # color(v) = (v // 10) % 4
    colors_struct2 = [(v // 10) % 4 for v in range(N)]
    if _is_valid(colors_struct2):
        return [int(c) for c in colors_struct2]
    
    # Try: (v % 5) * 2 % 4
    colors_struct3 = [(v % 5) * 2 % 4 for v in range(N)]
    if _is_valid(colors_struct3):
        return [int(c) for c in colors_struct3]
    
    # Try: (v // 5 + v % 5) % 4
    colors_struct4 = [(v // 5 + v % 5) % 4 for v in range(N)]
    if _is_valid(colors_struct4):
        return [int(c) for c in colors_struct4]
    
    # Fallback: return a valid 50-coloring (always valid)
    return [int(i) for i in range(N)]
