import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']
adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# Precompute degree for ordering
degrees = [len(adj[i]) for i in range(n)]

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit
    
    # Trivial valid solution: all different colors
    fallback = list(range(n))
    
    # Try increasing number of colors
    for num_colors in range(2, n + 1):
        if time.time() > deadline - 5:
            break
        result = _backtrack(num_colors, start_time, deadline)
        if result is not None:
            return result
    
    return fallback


def _backtrack(num_colors, start_time, deadline):
    """Backtracking with MRV, forward checking, and union-find for cycle detection."""
    
    # Order vertices by degree descending (process high-degree first)
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    # color[v] = assigned color or -1
    color = [-1] * n
    
    # For each pair of colors (c1, c2) with c1 < c2, maintain union-find
    # to detect cycles in the bicolored subgraph
    # Key: (c1, c2) -> dict mapping vertex -> parent (union-find)
    # We only need to track edges between same-color-pair vertices
    
    # Union-find per color pair
    # uf[(c1,c2)] = {vertex: parent_vertex}
    uf = {}
    
    def uf_find(pair_key, x):
        d = uf[pair_key]
        root = x
        while d[root] != root:
            root = d[root]
        # Path compression
        cur = x
        while d[cur] != root:
            nxt = d[cur]
            d[cur] = root
            cur = nxt
        return root
    
    def uf_union(pair_key, x, y):
        d = uf[pair_key]
        rx = uf_find(pair_key, x)
        ry = uf_find(pair_key, y)
        if rx == ry:
            return False  # cycle detected
        d[rx] = ry
        return True
    
    def get_pair_key(c1, c2):
        if c1 > c2:
            c1, c2 = c2, c1
        return (c1, c2)
    
    # For each color, track which vertices have that color
    color_vertices = [set() for _ in range(num_colors)]
    
    def assign(v, c):
        """Assign color c to vertex v. Returns True if valid, False if conflict."""
        color[v] = c
        color_vertices[c].add(v)
        
        # Check all neighbors: they must not have color c
        # (This is checked by the caller via domain filtering)
        
        # For each neighbor with a different color, check bicolored cycle
        for nb in adj[v]:
            nc = color[nb]
            if nc != -1 and nc != c:
                pk = get_pair_key(c, nc)
                if pk not in uf:
                    uf[pk] = {}
                d = uf[pk]
                # Initialize both vertices in this UF if not present
                if v not in d:
                    d[v] = v
                if nb not in d:
                    d[nb] = nb
                if not uf_union(pk, v, nb):
                    # Cycle detected - undo
                    return False
        
        return True
    
    def unassign(v, c):
        """Undo assignment of color c to vertex v."""
        color[v] = -1
        color_vertices[c].discard(v)
        # Note: we don't undo UF state (we use snapshot/restore approach instead)
        # Actually for simplicity, we'll use a different approach: track UF state changes
    
    # Better approach: use recursion with explicit state saving
    # For 50 vertices, we can afford to be a bit less efficient
    
    # Let's use a cleaner recursive approach with state snapshots
    # Since num_colors is small, the UF states are manageable
    
    # Actually, let's use a simpler approach: just check cycle by DFS when needed
    # For small graphs this is fine
    
    def has_bicolored_cycle(c1, c2, color_arr):
        """Check if subgraph induced by colors c1 and c2 has a cycle."""
        if c1 == c2:
            return False
        # Build subgraph of vertices with color c1 or c2
        vertices = set()
        for v in range(n):
            if color_arr[v] in (c1, c2):
                vertices.add(v)
        
        if len(vertices) < 3:
            return False
        
        # BFS/DFS to detect cycle
        visited = set()
        parent = {}
        
        for start in vertices:
            if start in visited:
                continue
            # BFS
            queue = [start]
            visited.add(start)
            parent[start] = -1
            while queue:
                cur = queue.pop(0)
                for nb in adj[cur]:
                    if nb not in vertices:
                        continue
                    if nb == parent[cur]:
                        continue
                    if nb in visited:
                        return True  # cycle
                    visited.add(nb)
                    parent[nb] = cur
                    queue.append(nb)
        return False
    
    # This is O(E) per check, but with 50 vertices it's fast enough
    # Let's use the union-find approach but with proper rollback
    
    # I'll use a different strategy: maintain UF with rollback
    # Each UF operation is logged so we can undo
    
    # For simplicity and correctness, let's just use the DFS cycle check
    # It's O(n) per check and we have at most 50*49/2 = 1225 pairs max
    # But in practice we only check pairs that are actually used
    
    def check_valid_assignment(color_arr, v, c):
        """Check if assigning color c to vertex v is valid given current assignment."""
        # Check 1: no neighbor has color c
        for nb in adj[v]:
            if color_arr[nb] == c:
                return False
        
        # Check 2: for each neighbor with a different assigned color, 
        # the bicolored subgraph must remain a forest
        for nb in adj[v]:
            nc = color_arr[nb]
            if nc != -1 and nc != c:
                # Temporarily assign and check
                color_arr[v] = c
                if has_bicolored_cycle(c, nc, color_arr):
                    color_arr[v] = -1
                    return False
                color_arr[v] = -1
        
        return True
    
    # Recursive backtracking with MRV
    def backtrack(idx, used_colors_count):
        if time.time() > deadline - 2:
            return None
        
        if idx == n:
            return color[:]
        
        v = order[idx]
        
        # Try colors in order of least constraining value
        # First, try colors already used, then new colors
        colors_to_try = []
        for c in range(num_colors):
            colors_to_try.append(c)
        
        # Sort by: prefer colors that are already used (to minimize distinct colors)
        # Actually, we want to use as few colors as possible, so prefer already-used colors
        used = set()
        for i in range(n):
            if color[i] != -1:
                used.add(color[i])
        
        colors_to_try.sort(key=lambda c: (c not in used, c))
        
        for c in colors_to_try:
            if not check_valid_assignment(color, v, c):
                continue
            
            color[v] = c
            result = backtrack(idx + 1, used_colors_count + (1 if c not in used else 0))
            if result is not None:
                return result
            color[v] = -1
        
        return None
    
    result = backtrack(0, 0)
    return result


# EVOLVE-BLOCK-END
