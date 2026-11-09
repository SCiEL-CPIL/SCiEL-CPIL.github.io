"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _has_bicolored_cycle(adj, colors, k):
    """
    Returns True if any pair of colors induces a cycle.
    Uses union-find with early exit.
    """
    n = P['n']
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            nodes = set()
            for i in range(n):
                if colors[i] == c1 or colors[i] == c2:
                    nodes.add(i)
            if len(nodes) < 3:
                continue
            parent = {i: i for i in nodes}
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
            for u in nodes:
                for v in adj[u]:
                    if v in nodes and u < v:
                        if not union(u, v):
                            return True
    return False

def _solve_dlx(adj, n, k, deadline):
    """
    Solve the bicolored-forest coloring problem using Dancing Links (Algorithm X).
    
    Columns:
      - n columns for vertex assignments (col v = vertex v must be assigned)
      - k*n columns for color usage (col n + v*k + c = vertex v has color c)
      - For each edge (u,v) and color pair (c1,c2) with c1<c2:
        col 2*n + edge_idx * (k*(k-1)//2) + pair_idx
        This column is hit by row (v, c) if u has color c' != c such that
        {c, c'} == {c1, c2}... 
    
    Actually, let me use a cleaner formulation:
    
    Rows: (vertex v, color c) for all v in 0..n-1, c in 0..k-1
    Columns:
      1. Vertex columns: col_v (v=0..n-1) — each vertex must be assigned exactly one color
      2. Edge-color-pair columns: for each edge e=(u,v) and each unordered color pair {c1,c2},
         a column that must be hit at most once (at most one assignment to u or v uses that pair)
         Actually, we need: for edge (u,v), if u gets color c1 and v gets color c2 (c1!=c2),
         then the pair {c1,c2} is "used" by this edge. We need to ensure that for any pair {c1,c2},
         the edges using that pair form a forest.
    
    Hmm, this is getting complex. Let me use a different approach:
    
    The key insight: for each color pair {c1,c2}, the subgraph induced by vertices colored c1 or c2
    must be a forest. This means: for each such pair, the number of edges in the induced subgraph
    must be <= (number of vertices in the induced subgraph) - 1 (for each connected component).
    
    A simpler sufficient condition that's also necessary for forests: no cycle.
    
    Let me use a direct backtracking with efficient cycle detection using DSU with rollback.
    """
    pass  # Will be replaced by the actual solver below

def _backtrack_dlx(adj, n, k, deadline):
    """
    Backtracking solver with DSU-with-rollback for bicolored cycle detection.
    
    For each color pair (c1, c2), maintain a DSU over vertices colored c1 or c2.
    When assigning vertex v color c:
      For each neighbor u with color c2 != c:
        - Add v and u to DSU for pair (min(c,c2), max(c,c2))
        - Union v and u in that DSU
        - If union fails (cycle detected), this assignment is invalid
    
    We use a journal to undo DSU operations.
    """
    edges = P['edges']
    num_pairs = k * (k - 1) // 2
    
    # DSU state for each color pair
    # For pair index pi corresponding to (c1, c2):
    #   dsu_parent[pi] = dict mapping vertex -> parent
    #   dsu_rank[pi] = dict mapping vertex -> rank
    #   dsu_journal[pi] = list of (child, parent_before, rank_parent_before)
    
    dsu_parent = [{} for _ in range(num_pairs)]
    dsu_rank = [{} for _ in range(num_pairs)]
    dsu_journal = [[] for _ in range(num_pairs)]
    
    colors = [-1] * n
    solution = [None]
    
    def pair_idx(c1, c2):
        if c1 > c2:
            c1, c2 = c2, c1
        return c1 * (k - 1) - c1 * (c1 + 1) // 2 + c2 - c1 - 1
        # Actually let me use a simpler formula
        # For k colors, pairs are (0,1),(0,2),...,(0,k-1),(1,2),...,(k-2,k-1)
        # Index of (c1,c2) with c1<c2: c1*(2*k-c1-3)//2 + c2 - c1 - 1
        # Let me just use a dict
    
    # Use a dict for pair indices
    pair_map = {}
    pi = 0
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            pair_map[(c1, c2)] = pi
            pi += 1
    
    def get_pair_idx(c1, c2):
        if c1 > c2:
            c1, c2 = c2, c1
        return pair_map[(c1, c2)]
    
    def dsu_find(parent_dict, x):
        while parent_dict[x] != x:
            parent_dict[x] = parent_dict[parent_dict[x]]
            x = parent_dict[x]
        return x
    
    def dsu_union(parent_dict, rank_dict, journal, x, y):
        """Returns True if union was performed, False if cycle detected."""
        rx = dsu_find(parent_dict, x)
        ry = dsu_find(parent_dict, y)
        if rx == ry:
            return False  # Cycle!
        if rank_dict[rx] < rank_dict[ry]:
            rx, ry = ry, rx
        journal.append((ry, rx, rank_dict[rx]))
        parent_dict[ry] = rx
        if rank_dict[rx] == rank_dict[ry]:
            rank_dict[rx] += 1
        return True
    
    def dsu_undo(journal, parent_dict, rank_dict, count):
        for _ in range(count):
            child, parent, rank_before = journal.pop()
            parent_dict[child] = child
            rank_dict[parent] = rank_before
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == n:
            solution[0] = colors[:]
            return True
        
        # MRV: find uncolored vertex with fewest valid colors
        # For efficiency, just use vertex idx for now, but try to be smart
        
        v = idx
        used_by_neighbors = set()
        for u in adj[v]:
            if colors[u] != -1:
                used_by_neighbors.add(colors[u])
        
        valid_colors = [c for c in range(k) if c not in used_by_neighbors]
        
        # Try colors in order
        for c in valid_colors:
            # Check bicolored cycle constraint
            # For each neighbor u with color c2 != c:
            #   In DSU for pair (c, c2), check if adding edge (v,u) creates a cycle
            
            valid = True
            unions_done = []  # (pair_idx, count) for undo
            
            for u in adj[v]:
                c2 = colors[u]
                if c2 == -1 or c2 == c:
                    continue
                
                pidx = get_pair_idx(c, c2)
                pd = dsu_parent[pidx]
                pr = dsu_rank[pidx]
                pj = dsu_journal[pidx]
                
                # Ensure v and u are in the DSU
                if v not in pd:
                    pd[v] = v
                    pr[v] = 0
                if u not in pd:
                    pd[u] = u
                    pr[u] = 0
                
                if not dsu_union(pd, pr, pj, v, u):
                    valid = False
                    break
                else:
                    unions_done.append(pidx)
            
            if not valid:
                # Undo any unions we did
                for pidx in unions_done:
                    dsu_undo(dsu_journal[pidx], dsu_parent[pidx], dsu_rank[pidx], 1)
                continue
            
            # Assign color
            colors[v] = c
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            colors[v] = -1
            for pidx in unions_done:
                dsu_undo(dsu_journal[pidx], dsu_parent[pidx], dsu_rank[pidx], 1)
        
        return False
    
    if backtrack(0):
        return solution[0]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85
    adj = _build_adjacency()
    n = P['n']
    edges = P['edges']
    
    # Try k from 3 upward
    for k in range(3, 7):
        if time.time() > deadline - 5:
            break
        
        # Use backtracking with DSU rollback
        result = _backtrack_dlx(adj, n, k, deadline - 3)
        
        if result is not None:
            # Verify
            if all(result[u] != result[v] for u, v in edges):
                if not _has_bicolored_cycle(adj, result, k):
                    return [int(c) for c in result]
        
        # If backtracking failed, try random local search
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            
            colors = [random.randint(0, k - 1) for _ in range(n)]
            
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Check validity
                proper = all(colors[u] != colors[v] for u, v in edges)
                if proper and not _has_bicolored_cycle(adj, colors, k):
                    return [int(c) for c in colors]
                
                # Pick a random vertex and try to improve
                v = random.randint(0, n - 1)
                old_color = colors[v]
                
                # Try to find a better color
                best_new = old_color
                for c in range(k):
                    if c == old_color:
                        continue
                    colors[v] = c
                    proper = all(colors[u] != colors[v] for u, v in edges)
                    if proper and not _has_bicolored_cycle(adj, colors, k):
                        return [int(c) for c in colors]
                    colors[v] = old_color
                
                # Random change
                new_color = random.randint(0, k - 1)
                while new_color == old_color:
                    new_color = random.randint(0, k - 1)
                colors[v] = new_color
    
    # Fallback: use many colors
    return [i % 50 for i in range(n)]
# EVOLVE-BLOCK-END
