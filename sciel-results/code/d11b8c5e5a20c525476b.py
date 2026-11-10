"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    edges = P['edges']
    n = P['n']
    deadline = time.time() + 80.0
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge list for faster access
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    def check_bicolored(coloring, max_colors):
        """Check if any pair of colors induces a cycle. Returns True if valid (all forests)."""
        # Group vertices by color
        color_to_verts = defaultdict(list)
        for i, c in enumerate(coloring):
            color_to_verts[c].append(i)
        
        colors = list(color_to_verts.keys())
        if len(colors) < 2:
            return True
        
        for idx in range(len(colors)):
            c1 = colors[idx]
            for c2 in colors[idx+1:]:
                verts = set(color_to_verts[c1]) | set(color_to_verts[c2])
                if len(verts) <= 1:
                    continue
                # Count edges within this subset
                edge_count = 0
                for u in verts:
                    for v in adj[u]:
                        if v in verts and u < v:
                            edge_count += 1
                # A forest on k vertices has at most k-1 edges
                if edge_count >= len(verts):
                    return False
        return True
    
    def is_valid(coloring, max_colors):
        """Full validity check."""
        # Proper coloring
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        # Bicolored forest check
        return check_bicolored(coloring, max_colors)
    
    # Strategy: Try 3-colorings first (best score = 16.67)
    # Use systematic search with pruning
    
    best_result = None
    best_colors = 50
    
    # First: try simple periodic patterns with 3 colors
    for period in range(3, 12):
        for offset in range(period):
            coloring = [((i + offset) % period) % 3 for i in range(n)]
            if is_valid(coloring, 3):
                return [int(c) for c in coloring]
    
    # Try i % 3 with small perturbations
    base = [i % 3 for i in range(n)]
    if is_valid(base, 3):
        return [int(c) for c in base]
    
    # Systematic backtracking for 3-coloring
    # Order vertices by degree (highest first) for better pruning
    deg_order = sorted(range(n), key=lambda x: -len(adj[x]))
    pos_in_order = [0] * n
    for i, v in enumerate(deg_order):
        pos_in_order[v] = i
    
    def backtrack_3colors():
        """Backtracking search for valid 3-coloring."""
        nonlocal deadline
        coloring = [-1] * n
        assigned = [False] * n
        
        def count_edges_in_pair(c1, c2):
            """Count edges between vertices colored c1 or c2."""
            count = 0
            for u, v in edges:
                cu = coloring[u]
                cv = coloring[v]
                if cu == -1 or cv == -1:
                    continue
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1) or (cu == c1 and cv == c1) or (cu == c2 and cv == c2):
                    # Both endpoints have colors in {c1, c2}
                    count += 1
            return count
        
        def count_verts_in_pair(c1, c2):
            count = 0
            for i in range(n):
                if coloring[i] == c1 or coloring[i] == c2:
                    count += 1
            return count
        
        def can_assign(v, c):
            """Check if assigning color c to vertex v is consistent."""
            # Check proper coloring
            for u in adj[v]:
                if assigned[u] and coloring[u] == c:
                    return False
            # Check bicolored constraints incrementally
            # For each other color d, check if {c, d} would form a cycle
            for d in range(3):
                if d == c:
                    continue
                # Tentatively assign
                coloring[v] = c
                assigned[v] = True
                # Count edges and vertices in {c, d}
                e_count = 0
                v_count = 0
                for i in range(n):
                    if coloring[i] == c or coloring[i] == d:
                        v_count += 1
                for u, w in edges:
                    cu = coloring[u]
                    cw = coloring[w]
                    if (cu == c or cu == d) and (cw == c or cw == d):
                        e_count += 1
                if e_count >= v_count:
                    coloring[v] = -1
                    assigned[v] = False
                    return False
                coloring[v] = -1
                assigned[v] = False
            return True
        
        def solve(idx):
            nonlocal deadline
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            v = deg_order[idx]
            for c in range(3):
                if can_assign(v, c):
                    coloring[v] = c
                    assigned[v] = True
                    if solve(idx + 1):
                        return True
                    coloring[v] = -1
                    assigned[v] = False
            return False
        
        return solve(0)
    
    if time.time() < deadline:
        if backtrack_3colors():
            # Extract result
            pass  # coloring is set inside; need to restructure
    
    # Since backtracking with nonlocal is tricky, use a class-based approach
    class Solver:
        def __init__(self):
            self.coloring = [-1] * n
            self.assigned = [False] * n
            self.found = False
        
        def can_assign(self, v, c):
            for u in self.adj[v]:
                if self.assigned[u] and self.coloring[u] == c:
                    return False
            # Check bicolored constraints
            for d in range(3):
                if d == c:
                    continue
                self.coloring[v] = c
                self.assigned[v] = True
                v_count = sum(1 for i in range(n) if self.coloring[i] in (c, d))
                e_count = sum(1 for u, w in self.edges if self.coloring[u] in (c, d) and self.coloring[w] in (c, d))
                self.coloring[v] = -1
                self.assigned[v] = False
                if e_count >= v_count:
                    return False
            return True
        
        def solve(self, idx):
            if time.time() > self.deadline:
                return False
            if idx == n:
                self.found = True
                return True
            v = self.deg_order[idx]
            for c in range(3):
                if self.can_assign(v, c):
                    self.coloring[v] = c
                    self.assigned[v] = True
                    if self.solve(idx + 1):
                        return True
                    self.coloring[v] = -1
                    self.assigned[v] = False
            return False
    
    if time.time() < deadline:
        solver = Solver()
        solver.adj = adj
        solver.edges = edges
        solver.deadline = deadline
        solver.deg_order = deg_order
        if solver.solve(0):
            return [int(c) for c in solver.coloring]
    
    # Try 4-coloring
    class Solver4:
        def __init__(self):
            self.coloring = [-1] * n
            self.assigned = [False] * n
            self.found = False
        
        def can_assign(self, v, c):
            for u in self.adj[v]:
                if self.assigned[u] and self.coloring[u] == c:
                    return False
            for d in range(4):
                if d == c:
                    continue
                self.coloring[v] = c
                self.assigned[v] = True
                v_count = sum(1 for i in range(n) if self.coloring[i] in (c, d))
                e_count = sum(1 for u, w in self.edges if self.coloring[u] in (c, d) and self.coloring[w] in (c, d))
                self.coloring[v] = -1
                self.assigned[v] = False
                if e_count >= v_count:
                    return False
            return True
        
        def solve(self, idx):
            if time.time() > self.deadline:
                return False
            if idx == n:
                self.found = True
                return True
            v = self.deg_order[idx]
            for c in range(4):
                if self.can_assign(v, c):
                    self.coloring[v] = c
                    self.assigned[v] = True
                    if self.solve(idx + 1):
                        return True
                    self.coloring[v] = -1
                    self.assigned[v] = False
            return False
    
    if time.time() < deadline:
        solver4 = Solver4()
        solver4.adj = adj
        solver4.edges = edges
        solver4.deadline = deadline
        solver4.deg_order = deg_order
        if solver4.solve(0):
            return [int(c) for c in solver4.coloring]
    
    # Fallback: use i%5
    coloring = [i % 5 for i in range(n)]
    if is_valid(coloring, 5):
        return [int(c) for c in coloring]
    
    # Last resort: unique colors
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
