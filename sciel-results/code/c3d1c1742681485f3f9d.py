"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    n = P['n']
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def construct():
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency()
    
    # Order vertices by degree (all degree 4 here) then by index for determinism
    # Use a good vertex ordering: sort by degree descending, then by index
    order = list(range(n))
    
    # We'll do a backtracking search minimizing number of colors
    # Union-find per color pair to detect bicolored cycles
    
    class DSU:
        def __init__(self):
            self.parent = {}
            self.rank = {}
        
        def find(self, x):
            if x not in self.parent:
                self.parent[x] = x
                self.rank[x] = 0
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False  # cycle detected
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
    
    # For each color pair (c1, c2), we maintain a DSU over the vertices
    # When we assign color c to vertex v, for each neighbor u with color d != c:
    #   We add edge (v, u) to the DSU for pair (min(c,d), max(c,d))
    #   If union fails, we have a bicolored cycle -> invalid
    
    # State: color[v] = assigned color or -1
    # dsu[c1][c2] = DSU object for color pair (c1, c2) where c1 < c2
    
    best_solution = None
    best_num_colors = n + 1
    
    deadline = time.time() + 55  # safety margin
    
    # Try increasing number of colors
    for max_colors in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Backtracking search with max_colors colors
        color = [-1] * n
        # dsu_map[(c1, c2)] where c1 < c2
        dsu_map = {}
        
        def get_dsu(c1, c2):
            key = (min(c1, c2), max(c1, c2))
            if key not in dsu_map:
                dsu_map[key] = DSU()
            return dsu_map[key]
        
        # Track which vertices have which color for quick neighbor checking
        # For each color c, list of vertices with that color
        color_to_verts = defaultdict(list)
        
        def backtrack(idx):
            nonlocal best_solution, best_num_colors
            
            if time.time() > deadline:
                return False
            
            if idx == n:
                # Found a valid solution
                num_colors = len(set(color))
                if num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_solution = color[:]
                return True
            
            v = order[idx]
            
            # Try each color from 0 to max_colors-1
            # Pruning: color must differ from all neighbors
            neighbor_colors = set()
            for u in adj[v]:
                if color[u] != -1:
                    neighbor_colors.add(color[u])
            
            for c in range(max_colors):
                if c in neighbor_colors:
                    continue
                
                # Check if assigning color c to v creates a bicolored cycle
                # For each neighbor u with color d != c:
                #   Add (v, u) to DSU for pair (min(c,d), max(c,d))
                #   If any union fails, this assignment is invalid
                new_dsu_updates = []
                valid = True
                for u in adj[v]:
                    d = color[u]
                    if d == -1 or d == c:
                        continue
                    key = (min(c, d), max(c, d))
                    dsu = get_dsu(key)
                    if not dsu.union(v, u):
                        valid = False
                        break
                    new_dsu_updates.append((key, v, u))
                
                if not valid:
                    continue
                
                # Apply assignment
                color[v] = c
                color_to_verts[c].append(v)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                color[v] = -1
                color_to_verts[c].remove(v)
                
                # Undo DSU updates - this is tricky. 
                # Instead, let's use a different approach: 
                # We'll save/restore the DSU state
                # For now, this is expensive. Let's try a different strategy.
                # Actually, undoing union-find is hard. Let me restructure.
                
                break  # This won't work well. Let me rethink.
        
        # The DSU undo problem makes pure backtracking hard.
        # Let me try a different approach: greedy with local search, or 
        # use a different cycle detection that's easier to undo.
        
        # Alternative: Instead of DSU, track the graph for each color pair
        # and check for cycles. But that's also expensive.
        
        # Let me try a completely different approach: 
        # Use a greedy algorithm with careful color ordering, then local search.
        break
    
    # Given the complexity, let me try a simpler approach:
    # 1. Start with a valid coloring (e.g., all different colors)
    # 2. Try to reduce colors via local search
    
    # Initial: each vertex gets its own color
    color = list(range(n))
    
    def is_valid(col):
        """Check if coloring is valid: proper + no bicolored cycles"""
        # Check proper coloring
        for u, v in edges:
            if col[u] == col[v]:
                return False
        
        # Check no bicolored cycles using DSU per color pair
        dsu = {}
        for u, v in edges:
            cu, cv = col[u], col[v]
            if cu == cv:
                continue
            key = (min(cu, cv), max(cu, cv))
            if key not in dsu:
                dsu[key] = DSU()
            if not dsu[key].union(u, v):
                return False
        return True
    
    # Local search: try to merge colors (reduce number of distinct colors)
    # Start with n colors, try to reduce
    
    current = color[:]
    current_valid = is_valid(current)
    assert current_valid, "Initial coloring should be valid"
    
    best = current[:]
    best_count = n
    
    start_time = time.time()
    
    # Greedy reduction: try to recolor vertex v to match an existing color
    # that doesn't violate constraints
    improved = True
    while improved and time.time() - start_time < 50:
        improved = False
        # Get current distinct colors
        distinct = sorted(set(current))
        
        for v in range(n):
            if time.time() - start_time > 50:
                break
            # Try to change color[v] to each existing color
            for c in distinct:
                if c == current[v]:
                    continue
                # Check if c is adjacent to any neighbor of v
                if any(current[u] == c for u in adj[v]):
                    continue
                
                # Try the change
                old_c = current[v]
                current[v] = c
                
                if is_valid(current):
                    # Check if this reduces the number of distinct colors
                    new_distinct = set(current)
                    if len(new_distinct) < len(set(best)):
                        best = current[:]
                        best_count = len(new_distinct)
                        improved = True
                    elif len(new_distinct) < len(set(current[:])):
                        # Even if not better than best, keep if it reduces
                        pass
                    # Keep the change if it's valid
                    if len(set(current)) < len(set(best)):
                        best = current[:]
                        best_count = len(set(current))
                else:
                    current[v] = old_c
        
        # Also try to recolor to a completely new lower color
        # This is harder, skip for now
    
    # Try more aggressive: for each vertex, try all colors 0..best_count-1
    for iteration in range(20):
        if time.time() - start_time > 55:
            break
        improved = False
        distinct = sorted(set(best))
        for v in range(n):
            if time.time() - start_time > 55:
                break
            for c in range(best_count):
                if c == best[v]:
                    continue
                if any(best[u] == c for u in adj[v]):
                    continue
                trial = best[:]
                trial[v] = c
                if is_valid(trial):
                    if len(set(trial)) < len(set(best)):
                        best = trial
                        best_count = len(set(best))
                        improved = True
                        break
        if not improved:
            break
    
    # Return best solution
    result = [int(x) for x in best]
    return result

# EVOLVE-BLOCK-END
