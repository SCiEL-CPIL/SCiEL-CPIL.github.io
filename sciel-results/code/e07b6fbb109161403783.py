import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Start with the known K11 construction
    n_base = 11
    colors_base = [min(j - i, n_base - j + i) - 1 for i in range(n_base) for j in range(i + 1, n_base)]

    # We'll try to extend this to larger n using a greedy + local search approach
    # The key constraint: no monochromatic C4 (no two vertices with two common neighbours in same colour)

    def build_coloring(n, colors_list):
        """Rebuild the edge index from colors_list (length n*(n-1)/2)"""
        return colors_list

    def get_edge_idx(i, j, n):
        """Get the index in the colors list for edge (i,j) where i<j"""
        # Lexicographic order: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1)
        # Index = i*n - i*(i+1)//2 + (j - i - 1)
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def count_c4_violations(n, colors):
        """Count number of monochromatic C4s. A C4 is two vertices u,v with two common neighbours in same colour."""
        violations = 0
        # For each pair of vertices (u,v), count common neighbours per colour
        # If any colour has >= 2 common neighbours, that's a violation
        for u in range(n):
            for v in range(u + 1, n):
                common = [0] * 5
                for w in range(n):
                    if w == u or w == v:
                        continue
                    cuw = colors[get_edge_idx(min(u, w), max(u, w), n)]
                    cvw = colors[get_edge_idx(min(v, w), max(v, w), n)]
                    if cuw == cvw:
                        common[cuw] += 1
                for c in range(5):
                    if common[c] >= 2:
                        violations += 1
        return violations

    def is_valid(n, colors):
        """Check if coloring has no monochromatic C4"""
        return count_c4_violations(n, colors) == 0

    # Fast violation count using adjacency lists
    def fast_violation_count(n, colors):
        """Faster: for each colour, build adjacency and check for C4s"""
        # For each colour c, build adjacency list
        # A C4 exists iff there are two vertices with >= 2 common neighbours in colour c
        adj = [defaultdict(set) for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_edge_idx(i, j, n)]
                adj[c][i].add(j)
                adj[c][j].add(i)
        
        violations = 0
        for c in range(5):
            # For each pair of vertices, check common neighbours
            for u in range(n):
                for v in range(u + 1, n):
                    if v in adj[c][u]:
                        # They're adjacent in colour c, check common neighbours
                        common = adj[c][u] & adj[c][v]
                        if len(common) >= 2:
                            violations += 1
                    else:
                        # Not adjacent, but could still have common neighbours forming C4
                        # Actually C4 requires the 4 vertices to form a cycle, so u and v don't need to be adjacent
                        # Wait - a C4 is a 4-cycle. In a C4, opposite vertices are NOT adjacent.
                        # So we need: u and v are NOT adjacent in colour c, but have >= 2 common neighbours in colour c
                        common = adj[c][u] & adj[c][v]
                        if len(common) >= 2:
                            violations += 1
        return violations

    # Even faster: for each colour, for each pair of non-adjacent vertices, check common neighbours
    def fast_violation_count_v2(n, colors):
        """For each colour, check if any two non-adjacent vertices share 2+ neighbours"""
        adj = [set() for _ in range(n)]  # Will be per-colour, but let's do it differently
        # Build per-colour adjacency
        adj_c = [[set() for _ in range(n)] for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_edge_idx(i, j, n)]
                adj_c[c][i].add(j)
                adj_c[c][j].add(i)
        
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj_c[c][u]:
                        # Count common neighbours
                        common = 0
                        for w in range(n):
                            if w != u and w != v and w in adj_c[c][u] and w in adj_c[c][v]:
                                common += 1
                                if common >= 2:
                                    break
                        if common >= 2:
                            violations += 1
        return violations

    # Greedy extension: start with n_base, try to add vertices one by one
    def greedy_extend(n_start, colors_start, target_n, time_limit):
        """Try to extend the coloring from n_start to target_n"""
        n = n_start
        colors = colors_start[:]
        
        while n < target_n:
            if time.time() > time_limit:
                break
            
            # Try to add vertex n with each possible colour assignment
            best_assignment = None
            best_violations = float('inf')
            
            # For each possible assignment of colours to edges (n, 0..n-1)
            # There are 5^n possibilities - too many. Use greedy: assign one edge at a time
            new_colors = [0] * n  # colours for edges (n, 0), (n, 1), ..., (n, n-1)
            
            # Greedy: for each edge (n, j), pick the colour that minimizes new violations
            for j in range(n):
                best_c = 0
                best_score = float('inf')
                for c in range(5):
                    new_colors[j] = c
                    # Check how many C4s this creates
                    # A C4 involving vertex n: n, j, x, y where x,y are common neighbours of j in colour c,
                    # and edges (n,x) and (n,y) are also colour c
                    # Actually: C4 is n-j-x-y-n or similar
                    # Let's think: if edge (n,j) is colour c, then for any pair (x,y) where both (j,x) and (j,y) are colour c,
                    # if both (n,x) and (n,y) are also colour c, we get a C4 n-j-x-y-n
                    # So we need to avoid: for colour c, if (n,x) and (n,y) are both c, then (j,x) and (j,y) should not both be c
                    # Simpler: just count violations after assignment
                    
                    score = 0
                    # Check all pairs (x, y) with x < y, x != j, y != j
                    # If (n,x)=c and (n,y)=c and (j,x)=c and (j,y)=c, that's a C4
                    for x in range(n):
                        if x == j:
                            continue
                        if new_colors[x] != c:
                            continue
                        for y in range(x + 1, n):
                            if y == j:
                                continue
                            if new_colors[y] != c:
                                continue
                            # Check if (j,x) and (j,y) are both colour c
                            cx = colors[get_edge_idx(min(j, x), max(j, x), n)]
                            cy = colors[get_edge_idx(min(j, y), max(j, y), n)]
                            if cx == c and cy == c:
                                score += 1
                    if score < best_score:
                        best_score = score
                        best_c = c
                new_colors[j] = best_c
            
            # Build the full color list
            full_colors = colors + new_colors
            
            # Count total violations
            v = fast_violation_count_v2(n + 1, full_colors)
            if v == 0:
                colors = full_colors
                n += 1
            else:
                # Try random assignments
                found = False
                for attempt in range(1000):
                    if time.time() > time_limit:
                        break
                    rand_colors = [random.randint(0, 4) for _ in range(n)]
                    full_colors = colors + rand_colors
                    v = fast_violation_count_v2(n + 1, full_colors)
                    if v == 0:
                        colors = full_colors
                        n += 1
                        found = True
                        break
                if not found:
                    break
        
        return n, colors

    # Local search: try to fix violations by recoloring edges
    def local_search(n, colors, max_iter, time_limit):
        """Try to reduce violations to zero by recoloring edges"""
        colors = colors[:]
        for it in range(max_iter):
            if time.time() > time_limit:
                break
            v = fast_violation_count_v2(n, colors)
            if v == 0:
                return n, colors, True
            
            # Find a violated edge and recolor it
            # Pick a random edge and try all 5 colours
            best_improvement = 0
            best_edge = None
            best_color = None
            
            # Sample some edges
            num_edges = n * (n - 1) // 2
            for _ in range(min(50, num_edges)):
                ei = random.randint(0, num_edges - 1)
                old_c = colors[ei]
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    colors[ei] = new_c
                    new_v = fast_violation_count_v2(n, colors)
                    improvement = v - new_v
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_edge = ei
                        best_color = new_c
                    colors[ei] = old_c
            
            if best_edge is not None:
                colors[best_edge] = best_color
            else:
                # No improvement found, try random move (simulated annealing style)
                ei = random.randint(0, num_edges - 1)
                colors[ei] = random.randint(0, 4)
        
        final_v = fast_violation_count_v2(n, colors)
        return n, colors, final_v == 0

    # Start with base
    best_n = n_base
    best_colors = colors_base[:]
    
    # Try to extend
    n_ext, colors_ext = greedy_extend(n_base, colors_base, 29, deadline - 5.0)
    if n_ext > best_n:
        best_n = n_ext
        best_colors = colors_ext
    
    # Try local search on the extended coloring
    if n_ext > 1:
        n_ls, colors_ls, valid = local_search(n_ext, colors_ext, 5000, deadline - 2.0)
        if valid and n_ls >= best_n:
            best_n = n_ls
            best_colors = colors_ls
    
    # Also try starting from scratch with different seeds for local search
    if time.time() < deadline - 10.0:
        for target in [12, 13, 14, 15]:
            if time.time() > deadline - 5.0:
                break
            # Random initial coloring
            num_edges = target * (target - 1) // 2
            rand_colors = [random.randint(0, 4) for _ in range(num_edges)]
            n_ls, colors_ls, valid = local_search(target, rand_colors, 3000, deadline - 5.0)
            if valid and n_ls > best_n:
                best_n = n_ls
                best_colors = colors_ls
    
    # Final validation
    if best_n > 1:
        final_v = fast_violation_count_v2(best_n, best_colors)
        if final_v != 0:
            # Fall back to base
            best_n = n_base
            best_colors = colors_base
    
    return {'n': best_n, 'colors': best_colors}
