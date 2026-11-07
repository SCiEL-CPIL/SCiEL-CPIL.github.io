import time
import random
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def check_bicolored_forest(colors, a, b):
    """Check if subgraph induced by colors a and b is a forest using union-find."""
    verts = [u for u in range(N) if colors[u] in (a, b)]
    if len(verts) <= 1:
        return True
    parent = {v: v for v in verts}
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
        if colors[u] in (a, b) and colors[v] in (a, b):
            if not union(u, v):
                return False
    return True

def count_bicolored_cycles(colors):
    """Count total number of bicolored cycles (excess edges in each bicolored subgraph)."""
    color_set = set(colors)
    total_excess = 0
    proper_violations = 0
    for u, v in edges:
        if colors[u] == colors[v]:
            proper_violations += 1
    for a in color_set:
        for b in color_set:
            if a >= b:
                continue
            edge_count = 0
            vertex_count = 0
            for u in range(N):
                if colors[u] in (a, b):
                    vertex_count += 1
            for u, v in edges:
                if (colors[u] == a and colors[v] == b) or (colors[u] == b and colors[v] == a):
                    edge_count += 1
            if vertex_count > 0:
                max_forest_edges = vertex_count - 1
            else:
                max_forest_edges = 0
            excess = edge_count - max_forest_edges
            if excess > 0:
                total_excess += excess
    return proper_violations * 100 + total_excess

def is_valid(colors):
    return count_bicolored_cycles(colors) == 0

def construct():
    deadline = time.time() + 85.0
    start_time = time.time()
    
    # Variable ordering: vertices in order of decreasing degree, then by index
    # All vertices have degree 4 in this circulant graph, so use index order
    # But let's try a smarter ordering: start with vertices that have most constraints
    
    # For the circulant graph C(50,{1,7}), all vertices have degree 4
    # Use a fixed ordering that helps backtracking: group by residue classes
    
    # Variable order: try to assign in a pattern that spreads colors
    var_order = list(range(N))
    
    # Backtracking with forward checking
    def backtrack(k, k_max, colors, pair_components):
        """
        k: index in var_order
        k_max: max number of colors to use
        colors: current assignment
        pair_components: dict mapping (a,b) -> UnionFind structure for that color pair
        """
        if time.time() > deadline:
            return None
        
        if k == N:
            return colors[:]
        
        v = var_order[k]
        
        # Try each color from 0 to k_max-1
        for c in range(k_max):
            # Check proper coloring: no neighbor has color c
            proper = True
            for nb in adj[v]:
                if colors[nb] == c:
                    proper = False
                    break
            if not proper:
                continue
            
            # Check bicolored cycle constraint
            # For each color d != c, check if adding edge (v, nb) where nb has color d
            # would create a cycle in the (c,d) subgraph
            valid = True
            new_components = {}
            
            for d in range(k_max):
                if d == c:
                    continue
                # Check edges from v to neighbors with color d
                for nb in adj[v]:
                    if colors[nb] == d:
                        # This edge (v, nb) would be in the (c, d) subgraph
                        # Check if v and nb are already connected in (c, d) components
                        key = (min(c, d), max(c, d))
                        if key in pair_components:
                            uf = pair_components[key]
                            if uf.find(v) == uf.find(nb):
                                valid = False
                                break
                        # If not in pair_components, they're not connected yet
                        # (only edges between already-assigned vertices of colors c,d matter)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Assign color c to v
            colors[v] = c
            
            # Update pair_components: for each neighbor nb with color d != c,
            # union v and nb in the (c,d) component
            new_pair_components = {}
            for d in range(k_max):
                if d == c:
                    continue
                key = (min(c, d), max(c, d))
                if key in pair_components:
                    new_pair_components[key] = pair_components[key].copy()
                # Union v with all neighbors of color d
                for nb in adj[v]:
                    if colors[nb] == d:
                        if key not in new_pair_components:
                            new_pair_components[key] = pair_components.get(key, None)
                        if new_pair_components[key] is None:
                            # Create new UnionFind
                            new_pair_components[key] = UnionFind()
                        new_pair_components[key].union(v, nb)
            
            # Copy existing pair_components for pairs not involving c
            for key, uf in pair_components.items():
                if key not in new_pair_components:
                    new_pair_components[key] = uf
            
            # Recurse
            result = backtrack(k + 1, k_max, colors, new_pair_components)
            if result is not None:
                return result
            
            # Undo
            colors[v] = -1
        
        return None
    
    class UnionFind:
        def __init__(self):
            self.parent = {}
        
        def find(self, x):
            if x not in self.parent:
                self.parent[x] = x
                return x
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            self.parent[rx] = ry
            return True
        
        def copy(self):
            new_uf = UnionFind()
            new_uf.parent = dict(self.parent)
            return new_uf
    
    # Try increasing number of colors
    for k_max in range(3, 10):
        if time.time() > deadline - 5:
            break
        
        colors = [-1] * N
        pair_components = {}
        
        result = backtrack(0, k_max, colors, pair_components)
        if result is not None:
            return [int(c) for c in result]
    
    # Fallback: use local search
    # Simple greedy initialization
    colors = [0] * N
    for i in range(N):
        used = set()
        for v in adj[i]:
            if v < i:
                used.add(colors[v])
        for c in range(50):
            if c not in used:
                colors[i] = c
                break
    
    # Local search to minimize
    best_colors = colors[:]
    best_score = count_bicolored_cycles(best_colors)
    
    current = best_colors[:]
    current_score = best_score
    temperature = 5.0
    cooling = 0.9995
    
    while time.time() < deadline - 2.0:
        v = random.randint(0, N - 1)
        old_color = current[v]
        new_color = random.randint(0, 7)
        if new_color == old_color:
            continue
        
        conflict = False
        for nb in adj[v]:
            if current[nb] == new_color:
                conflict = True
                break
        if conflict:
            continue
        
        current[v] = new_color
        new_score = count_bicolored_cycles(current)
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < (1.0 / (1.0 + delta / temperature)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_colors = current[:]
                if best_score == 0:
                    return [int(c) for c in best_colors]
        else:
            current[v] = old_color
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    if best_score == 0:
        return [int(c) for c in best_colors]
    
    # Systematic local search
    current = best_colors[:]
    current_score = best_score
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for v in range(N):
            old_color = current[v]
            for new_color in range(8):
                if new_color == old_color:
                    continue
                conflict = False
                for nb in adj[v]:
                    if current[nb] == new_color:
                        conflict = True
                        break
                if conflict:
                    continue
                current[v] = new_color
                new_score = count_bicolored_cycles(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_colors = current[:]
                    if best_score == 0:
                        return [int(c) for c in best_colors]
                    break
            current[v] = old_color
    
    if best_score == 0:
        return [int(c) for c in best_colors]
    
    if best_colors is not None:
        return [int(c) for c in best_colors]
    
    return [int(c) for c in colors]
