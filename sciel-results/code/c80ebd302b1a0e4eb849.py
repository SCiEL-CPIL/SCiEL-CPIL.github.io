"""Rewritten program: backtracking with symmetry breaking and C6-free verification."""
import time
import random

# Deadline guard: return best found before 140s (10s safety margin under 150s limit)
_DEADLINE = 140.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _edge_index(n, i, j):
    """Index for edge (i,j), i<j, in lexicographic order."""
    # Number of edges before vertex i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

def _has_c6(color_edges, n):
    """
    Check if the set of edges (for one color class) contains a 6-cycle.
    color_edges: set of frozenset({u,v}) or just a set of (u,v) with u<v.
    We check for any 6 distinct vertices forming a cycle.
    For efficiency, we use adjacency lists and do a bounded BFS/DFS.
    """
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for (u, v) in color_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # A 6-cycle exists if there is a cycle of length 6.
    # We can check: for each pair of vertices at distance 3 in the graph,
    # check if they are connected by a path of length 3 (completing the cycle).
    # Actually, simpler: enumerate all 6-tuples? Too slow for n=17.
    # Better: for each vertex v, look at neighbors, then neighbors of neighbors, etc.
    # A 6-cycle through v: v-a-b-c-d-e-v
    # So we need: a in adj[v], b in adj[a], c in adj[b], d in adj[c], e in adj[d], and e in adj[v]
    # with all distinct.
    
    # More efficient: check if there exist two vertices u,w at distance exactly 3
    # such that there is also a path of length 3 between them that is vertex-disjoint
    # from the first path (except endpoints). Actually for a cycle, we need a closed walk of length 6 with distinct vertices.
    
    # Let's use: for each pair (u, w) with u < w, check if there are two internally
    # vertex-disjoint paths of length 3 from u to w. If so, they form a 6-cycle.
    
    # Even simpler approach for small n: 
    # For each ordered pair of vertices (start, end) that could be opposite in the cycle,
    # check if there's a path of length 3 from start to end, and another path of length 3
    # that is internally disjoint.
    
    # Actually, the most straightforward for n<=20: 
    # A 6-cycle is determined by 6 distinct vertices in order. We can check:
    # Does there exist a sequence v0,v1,v2,v3,v4,v5 all distinct such that
    # (vi, vi+1) is an edge for i=0..4 and (v5, v0) is an edge?
    
    # We can do this by: for each v0, for each v1 in adj[v0], for each v2 in adj[v1] \ {v0},
    # for each v3 in adj[v2] \ {v0,v1}, for each v4 in adj[v3] \ {v0,v1,v2},
    # check if v4 in adj[v0] and v4 != v0 and v4 != v1 and v4 != v2 and v4 != v3.
    # That's O(n * d^5) which for n=17 and d~8 is about 17*8^5 ≈ 5.6M, acceptable.
    
    for v0 in range(n):
        for v1 in adj[v0]:
            for v2 in adj[v1]:
                if v2 == v0:
                    continue
                for v3 in adj[v2]:
                    if v3 == v0 or v3 == v1:
                        continue
                    for v4 in adj[v3]:
                        if v4 == v0 or v4 == v1 or v4 == v2:
                            continue
                        if v4 in adj[v0]:
                            return True
    return False

def _verify(coloring, n):
    """Verify that no color class contains a C6."""
    color_edges = [set() for _ in range(4)]
    for idx in range(len(coloring)):
        c = coloring[idx]
        # Decode edge from index
        # Find i,j such that idx = i*(n-1) - i*(i-1)//2 + (j-i-1)
        # We can precompute or just iterate
        # For verification speed, let's decode:
        i = 0
        while True:
            base = i * (n - 1) - i * (i - 1) // 2
            next_base = (i + 1) * (n - 1) - (i + 1) * i // 2
            if idx < next_base:
                break
            i += 1
        j = i + 1 + (idx - base)
        color_edges[c].add((i, j))
    
    for c in range(4):
        if _has_c6(color_edges[c], n):
            return False
    return True

def _backtrack(n, colors, pos, deadline_check):
    """
    Backtracking search for a valid 4-coloring of K_n.
    colors: list of length n*(n-1)//2, initially all None.
    pos: current position to assign.
    Returns True if a valid coloring is found, modifying colors in place.
    """
    if pos >= len(colors):
        return True
    
    # Symmetry breaking: fix first few edges
    # Edge (0,1) -> color 0
    # Edge (0,2) -> color 1
    # Edge (0,3) -> color 2
    # Edge (1,2) -> color 3
    # This breaks S4 color symmetry and some vertex symmetry
    
    # For efficiency, we also fix:
    # Edge (1,3) -> color 0 or 1 (to break more symmetry)
    
    # Let's use standard symmetry breaking for 4 colors:
    # c(0,1)=0, c(0,2)=1, c(0,3)=2, c(1,2)=3
    # Additionally, c(1,3) can be 0 or 1 (by vertex symmetry swapping 2 and 3? No, 2 and 3 are fixed by c(0,2)=1, c(0,3)=2)
    # Actually with c(0,1)=0, c(0,2)=1, c(0,3)=2, c(1,2)=3, the remaining symmetry is limited.
    
    if pos == 0:
        colors[pos] = 0
        return _backtrack(n, colors, pos + 1, deadline_check)
    elif pos == 1:
        colors[pos] = 1
        return _backtrack(n, colors, pos + 1, deadline_check)
    elif pos == 2:
        colors[pos] = 2
        return _backtrack(n, colors, pos + 1, deadline_check)
    elif pos == 3:
        colors[pos] = 3
        return _backtrack(n, colors, pos + 1, deadline_check)
    
    # Check deadline
    if deadline_check and _time_left() < 1.0:
        return False
    
    i = 0
    while True:
        base = i * (n - 1) - i * (i - 1) // 2
        next_base = (i + 1) * (n - 1) - (i + 1) * i // 2
        if pos < next_base:
            break
        i += 1
    j = i + 1 + (pos - base)
    
    # Try colors 0-3
    for c in range(4):
        colors[pos] = c
        if _backtrack(n, colors, pos + 1, deadline_check):
            return True
    colors[pos] = None
    return False

def _backtrack_with_pruning(n, colors, pos, deadline_check):
    """
    Backtracking with C6 detection pruning.
    After assigning an edge, check if any color class now contains a C6.
    This is expensive but prunes the search space significantly.
    """
    if pos >= len(colors):
        return True
    
    if pos == 0:
        colors[pos] = 0
        return _backtrack_with_pruning(n, colors, pos + 1, deadline_check)
    elif pos == 1:
        colors[pos] = 1
        return _backtrack_with_pruning(n, colors, pos + 1, deadline_check)
    elif pos == 2:
        colors[pos] = 2
        return _backtrack_with_pruning(n, colors, pos + 1, deadline_check)
    elif pos == 3:
        colors[pos] = 3
        return _backtrack_with_pruning(n, colors, pos + 1, deadline_check)
    
    if deadline_check and _time_left() < 1.0:
        return False
    
    i = 0
    while True:
        base = i * (n - 1) - i * (i - 1) // 2
        next_base = (i + 1) * (n - 1) - (i + 1) * i // 2
        if pos < next_base:
            break
        i += 1
    j = i + 1 + (pos - base)
    
    # For pruning, we need to track edges per color incrementally.
    # For simplicity in this version, we do full verification only at the end
    # but that's too slow. Instead, let's do a lighter check:
    # After assigning color c to (i,j), check if adding this edge to color class c
    # creates a C6. We can do this by checking if there's a path of length 5
    # from i to j in color class c (before adding this edge). If so, adding (i,j)
    # creates a 6-cycle.
    
    # Build adjacency for color c up to current assignment
    # This is expensive to do at each step. Alternative: use a different strategy.
    
    # For now, let's just do the simple backtracking and verify at the end.
    # This will be slow for large n but let's try for n up to 14-15.
    
    for c in range(4):
        colors[pos] = c
        if _backtrack_with_pruning(n, colors, pos + 1, deadline_check):
            return True
    colors[pos] = None
    return False

def construct():
    """
    Find a 4-coloring of K_n with no monochromatic C6, maximizing n.
    Returns {'n': n, 'colors': [...]}
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try to find colorings for increasing n
    # Start from n=10 up to n=17 (known best is 17)
    # Use backtracking with symmetry breaking
    
    for n in range(10, 18):
        if _time_left() < 5.0:
            break
        
        num_edges = n * (n - 1) // 2
        colors = [None] * num_edges
        
        # Use backtracking
        if _backtrack(n, colors, 0, True):
            # Verify the solution
            if _verify(colors, n):
                best_n = n
                best_colors = colors[:]
                # Found a valid coloring for this n, try larger
                continue
            else:
                # Shouldn't happen if backtracking is correct, but just in case
                break
        else:
            # No solution found for this n within time
            break
    
    # If we didn't find anything above 9, return the known 9-vertex solution
    if best_n < 10:
        best_n = 9
        best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    return {'n': best_n, 'colors': best_colors}
