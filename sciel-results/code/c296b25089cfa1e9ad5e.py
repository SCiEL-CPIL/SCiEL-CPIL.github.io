import time
import sys
from functools import lru_cache
from itertools import combinations

def construct():
    """
    Find the maximum n such that K_n has a 4-edge-coloring with no monochromatic C6 in any color.
    Uses a backtracking search with incremental C6 detection.
    
    Strategy:
    1. Try to build colorings for n = 18, 17, 16, ...
    2. For each n, use backtracking to assign colors to edges.
    3. Maintain a data structure to quickly detect if adding an edge creates a C6.
    4. To speed up, use symmetry breaking and heuristics.
    
    Since n=17 is known to be possible, we first try n=18 (which would be new).
    If n=18 fails within time limit, we fall back to known constructions or continue searching.
    """
    
    # Known: n=17 is achievable. Let's see if we can find n=18 or n=19.
    # For practical purposes within 120s, we'll implement a smart backtracker.
    
    # A C6 in color c is a cycle of length 6 where all edges are color c.
    # We need to avoid monochromatic C6 in each color class.
    
    # Approach: 
    # - Build the graph edge by edge in lexicographic order.
    # - For each new edge (u, v) assigned color c, check if this creates a C6 in color c.
    # - To check efficiently, we can look for paths of length 5 between u and v in color c 
    #   that don't use the edge (u,v) itself, forming a C6 when combined with (u,v).
    # - Actually, adding edge (u,v) with color c creates a C6 iff there exists a path 
    #   of length 5 from u to v in the current subgraph using only color c edges,
    #   and that path doesn't use edge (u,v) (which it can't since we're adding it now).
    #   Wait, that's not quite right. A C6 containing edge (u,v) would be a path of length 5
    #   from u to v. But we need the entire cycle to be color c.
    
    # Better approach: maintain for each color c, the set of edges, and when adding a new edge,
    # check if it completes a C6. To do this efficiently, we can precompute or maintain
    # "forbidden pairs" - pairs of vertices that cannot be connected by color c because
    # it would create a C6.
    
    # For each color c and each pair (u,v), we can maintain the number of length-5 paths
    # from u to v using only color c edges. If this count > 0, then adding edge (u,v) in color c
    # would create a C6.
    
    # But maintaining this dynamically is complex. Let's use a simpler but effective approach:
    # When considering assigning color c to edge (u,v), check all possible C6s that include (u,v).
    # A C6 containing (u,v) has the form: u - a - b - d - e - v - u (or similar orderings).
    # The other 5 vertices a,b,d,e must form a path from u to v of length 5 in color c.
    
    # For n up to 18-20, the number of such paths to check might be manageable with pruning.
    
    # Let's implement a backtracker with the following:
    # - Process edges in lexicographic order.
    # - For each edge, try colors 0-3.
    # - For each color assignment, check if it creates a C6.
    # - Use symmetry breaking: fix first few edges to reduce search space.
    
    def creates_c6(n, color, u, v, colors_matrix, edge_order):
        """
        Check if adding edge (u,v) with given color creates a monochromatic C6.
        colors_matrix[i][j] = color of edge (i,j), or -1 if not set.
        We need to find a path u->p1->p2->p3->p4->v where all edges have the same color.
        """
        # A C6 containing edge (u,v) in color c means there exists a path of length 5
        # from u to v using only color c edges in the existing graph.
        # Let's do a BFS/DFS to find if there's a path of length exactly 5 from u to v
        # using only color c edges, not using edge (u,v).
        
        # Since we're adding (u,v), we check if there's already a path of length 5 from u to v in color c.
        # If yes, adding (u,v) completes the C6.
        
        # Path of length 5: u -> a -> b -> c -> d -> v
        # All edges (u,a), (a,b), (b,c), (c,d), (d,v) must have color = color.
        
        # We can enumerate: for each a adjacent to u in color c,
        #   for each b adjacent to a in color c (b != u),
        #     for each c adjacent to b in color c (c != a),
        #       for each d adjacent to c in color c (d != b),
        #         if (d,v) exists in color c and d != u and v != a and v != b and v != c and d != v:
        #           found a C6.
        
        # But we also need to ensure the path doesn't revisit vertices (simple path).
        # For a C6, all 6 vertices must be distinct.
        
        # This is O(n^4) per edge check, which is too slow.
        
        # Alternative: maintain for each color c, a 2D array path_count_c[i][j] = number of
        # length-k paths from i to j in color c, for k=1,2,3,4,5.
        # When adding edge (u,v) in color c, check if path_count_c[u][v] > 0 for length 5.
        # Then update the path counts.
        
        # Let's use this approach.
        return None  # We'll handle this differently
        
    # Let's use a different strategy: precompute all C6s and use a constraint-based approach.
    # For n=18, there are C(18,6) = 18564 C6s, each with 6 edges. For each C6, at least one
    # edge must NOT be in each color class. This is a covering problem.
    
    # Actually, let's just implement a straightforward backtracker with good heuristics
    # and symmetry breaking, and hope it finds a solution for n=17 quickly (as a sanity check)
    # and then try n=18.
    
    start_time = time.time()
    time_limit = 110  # Leave some margin
    
    def solve(n):
        """Try to find a valid 4-coloring of K_n with no monochromatic C6."""
        total_edges = n * (n - 1) // 2
        colors = [-1] * total_edges  # -1 means unassigned
        
        # Build adjacency lists for each color to check C6 formation
        # adj[color][i] = set of vertices j such that edge (i,j) has this color
        adj = [set() for _ in range(n)] * 4  # This won't work, let's fix
        adj = [[set() for _ in range(n)] for _ in range(4)]
        
        # For efficient C6 checking, maintain path counts
        # path_count[c][k][i][j] = number of simple paths of length k from i to j in color c
        # For k=5, we need this to check C6 formation.
        # But updating these when adding edges is complex.
        
        # Simpler: when adding edge (u,v) in color c, check if there's a path of length 5
        # from u to v in color c by doing a limited DFS.
        
        def has_path_length_5(c, u, v):
            """Check if there's a simple path of length 5 from u to v using only color c edges."""
            # Path: u -> a -> b -> d -> e -> v
            # We need all 6 vertices distinct.
            for a in adj[c][u]:
                if a == v:
                    continue
                for b in adj[c][a]:
                    if b in (u, v):
                        continue
                    for d in adj[c][b]:
                        if d in (u, v, a):
                            continue
                        for e in adj[c][d]:
                            if e in (u, v, a, b):
                                continue
                            if e == v:
                                return True
            return False
        
        def get_edge_idx(u, v):
            if u > v:
                u, v = v, u
            return u * n - u * (u + 1) // 2 + (v - u - 1)
        
        def set_edge(u, v, c):
            idx = get_edge_idx(u, v)
            colors[idx] = c
            adj[c][u].add(v)
            adj[c][v].add(u)
            return idx
        
        def unset_edge(idx, u, v, c):
            colors[idx] = -1
            adj[c][u].discard(v)
            adj[c][v].discard(u)
        
        # Symmetry breaking: fix the first vertex's edges
        # Edge (0,1) = 0, (0,2) = 0, (0,3) = 1, etc. to break color symmetries
        
        # Let's use a simpler symmetry breaking: 
        # Fix (0,1)=0, (0,2)=0, (0,3)=1, (0,4)=2, (0,5)=3 if n>5
        # This breaks the color permutation symmetry.
        
        fixed_edges = []
        if n >= 2:
            fixed_edges.append((0, 1, 0))
        if n >= 3:
            fixed_edges.append((0, 2, 0))
        if n >= 4:
            fixed_edges.append((0, 3, 1))
        if n >= 5:
            fixed_edges.append((0, 4, 2))
        if n >= 6:
            fixed_edges.append((0, 5, 3))
        
        # Apply fixed edges
        for u, v, c in fixed_edges:
            set_edge(u, v, c)
        
        # Order of edges to assign
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in [(a, b) for a, b, _ in fixed_edges] and \
                   (j, i) not in [(a, b) for a, b, _ in fixed_edges]:
                    edge_list.append((i, j))
        
        def backtrack(idx):
            if time.time() - start_time > time_limit:
                return None
            if idx == len(edge_list):
                return colors[:]
            
            u, v = edge_list[idx]
            for c in range(4):
                # Try assigning color c to edge (u, v)
                set_edge(u, v, c)
                
                # Check if this creates a C6 in color c
                if has_path_length_5(c, u, v):
                    unset_edge(get_edge_idx(u, v), u, v, c)
                    continue
                
                result = backtrack(idx + 1)
                if result is not None:
                    unset_edge(get_edge_idx(u, v), u, v, c)
                    return result
                
                unset_edge(get_edge_idx(u, v), u, v, c)
            
            return None
        
        result = backtrack(0)
        return result
    
    # Try n = 18 first (would be new)
    for n in [18, 17]:
        if time.time() - start_time > time_limit:
            break
        result = solve(n)
        if result is not None:
            return {'n': n, 'colors': result}
    
    # Fallback: return a known valid coloring for n=9 (from the seed)
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
