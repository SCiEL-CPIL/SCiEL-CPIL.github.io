"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _has_c6_in_colour(n, color, edge_color):
    """Check if colour class 'color' in K_n contains a 6-cycle.
    edge_color is a flat array of length n*(n-1)//2 with colours in lex order.
    We build adjacency for the subgraph of edges of 'color' and do BFS/DFS for cycles of length 6.
    For efficiency, we check via matrix multiplication or direct enumeration.
    Since n <= 20, we can use a bitset-based approach.
    """
    # Build adjacency bitsets for the color class
    adj = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if edge_color[idx] == color:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            idx += 1
    
    # Check for 6-cycles: a, b, c, d, e, f with a<b<c<d<e<f to avoid duplicates?
    # Actually, we just need to detect ANY 6-cycle.
    # A 6-cycle exists if there are vertices v0..v5 such that v0-v1-v2-v3-v4-v5-v0 are all edges.
    # To avoid checking all 6! permutations, we can fix v0 as the smallest vertex in the cycle.
    # But simpler: for each pair of non-adjacent vertices (u, w) at distance 3 in the cycle...
    # Alternative: enumerate all 6-tuples of distinct vertices? That's O(n^6) = 20^6 = 64M, too slow.
    
    # Better approach: For each edge (u, v), count paths of length 4 from u to v that don't repeat vertices.
    # If any such path exists and (u,v) is an edge, we have a 6-cycle? No, path of length 4 from u to v plus edge (u,v) gives a 5-cycle... wait.
    # A 6-cycle: v0-v1-v2-v3-v4-v5-v0. The distance from v0 to v3 is 3 (two ways).
    # So: for each triple of vertices (a, b, c) where a-b-c is a path of length 2, and we need to find a path of length 3 from a to c that avoids b and doesn't use a or c in the middle.
    # Actually, let's use the standard approach: 
    # For each pair (u, v) that is NOT an edge, count the number of common neighbors... no.
    
    # Let's use: a 6-cycle exists iff there exist vertices a,b,c,d,e,f distinct such that 
    # a~b, b~c, c~d, d~e, e~f, f~a.
    # We can iterate over all pairs of edges (a,b) and (c,d) that are disjoint, then check if there's a path b-...-c of length 2 and d-...-a of length 2.
    # This is getting complex. Let's just do a direct BFS for each starting vertex to find cycles of length 6.
    
    # For n up to 20, we can try: for each vertex v, do BFS up to depth 5, tracking visited sets.
    # But that's expensive.
    
    # Simplest correct approach for n<=20: 
    # Count the number of closed walks of length 6 that are cycles.
    # Actually, let's just enumerate all 6-combinations and check all 6!/2 = 360 orderings? No, that's too much.
    
    # Let's use: for each edge (u,v), the number of paths of length 4 from u to v in the subgraph, excluding u and v in the interior.
    # If count > 0, we have a 6-cycle.
    # Path of length 4 from u to v: u-x1-x2-x3-v where x1,x2,x3 are distinct from each other and from u,v.
    # We can compute this by:
    # 1. For each vertex x1 neighbor of u (x1 != v):
    # 2. For each vertex x2 neighbor of x1 (x2 != u, x2 != v):
    # 3. For each vertex x3 neighbor of x2 (x3 != x1, x3 != v):
    # 4. Check if x3 is neighbor of v.
    
    # This is O(deg(u) * deg(x1) * deg(x2)) which for dense graphs is O(n^3) per edge, total O(n^5) = 20^5 = 3.2M. Acceptable.
    
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue  # not an edge
            # Count paths of length 4 from i to j
            count = 0
            # x1 must be neighbor of i, x1 != j
            for x1 in range(n):
                if x1 == i or x1 == j:
                    continue
                if not (adj[i] & (1 << x1)):
                    continue
                # x2 must be neighbor of x1, x2 != i, x2 != j
                for x2 in range(n):
                    if x2 == i or x2 == j or x2 == x1:
                        continue
                    if not (adj[x1] & (1 << x2)):
                        continue
                    # x3 must be neighbor of x2, x3 != x1, x3 != j
                    for x3 in range(n):
                        if x3 == i or x3 == j or x3 == x1 or x3 == x2:
                            continue
                        if not (adj[x2] & (1 << x3)):
                            continue
                        if adj[x3] & (1 << j):
                            count += 1
                            break  # found one, no need to count more
            if count > 0:
                return True
    return False


def _check_no_c6(n, edge_color):
    """Check that no color class contains a 6-cycle."""
    for c in range(4):
        if _has_c6_in_colour(n, c, edge_color):
            return False
    return True


def _count_c6_in_colour(n, color, edge_color):
    """Count the number of 6-cycles in colour class 'color'."""
    adj = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if edge_color[idx] == color:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            idx += 1
    
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            for x1 in range(n):
                if x1 == i or x1 == j:
                    continue
                if not (adj[i] & (1 << x1)):
                    continue
                for x2 in range(n):
                    if x2 == i or x2 == j or x2 == x1:
                        continue
                    if not (adj[x1] & (1 << x2)):
                        continue
                    for x3 in range(n):
                        if x3 == i or x3 == j or x3 == x1 or x3 == x2:
                            continue
                        if not (adj[x2] & (1 << x3)):
                            continue
                        if adj[x3] & (1 << j):
                            total += 1
    return total


def _total_c6(n, edge_color):
    """Total number of 6-cycles across all 4 color classes."""
    total = 0
    for c in range(4):
        total += _count_c6_in_colour(n, c, edge_color)
    return total


def _k9_baseline():
    """Return the known K9 coloring."""
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return n, colors


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Start with K9 baseline
    n_best, colors_best = _k9_baseline()
    
    # Try to extend to n=10, 11, ..., 17 using simulated annealing
    # We'll start from n=9 and try to add vertices one by one
    
    # For each target n from 10 to 17:
    for target_n in range(10, 18):
        if time.time() > deadline:
            break
        
        # Initialize: take the current best coloring for target_n-1 vertices and add a new vertex
        # We'll use simulated annealing to find a valid coloring for target_n vertices
        
        # Start from the previous best coloring extended with random colors for the new vertex
        n_current = target_n - 1
        # Get the coloring for n_current from colors_best if n_current == n_best, else use a default
        if n_current == n_best:
            base_colors = colors_best[:]
        else:
            # Use a random initialization
            base_colors = [random.randint(0, 3) for _ in range(n_current * (n_current - 1) // 2)]
        
        # Add the new vertex (vertex n_current)
        # The new edges are (n_current, j) for j in 0..n_current-1
        # In lex order, these come after all edges of K_{n_current}
        new_edges_count = n_current
        initial_colors = base_colors + [random.randint(0, 3) for _ in range(new_edges_count)]
        
        # Simulated annealing
        current_colors = initial_colors[:]
        current_obj = _total_c6(target_n, current_colors)
        
        best_colors = current_colors[:]
        best_obj = current_obj
        
        # SA parameters
        T = 100.0
        T_min = 0.01
        cooling = 0.995
        max_steps = 50000
        
        step = 0
        while step < max_steps:
            if time.time() > deadline:
                break
            
            # Pick a random edge to flip
            edge_idx = random.randint(0, len(current_colors) - 1)
            old_color = current_colors[edge_idx]
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
            
            current_colors[edge_idx] = new_color
            new_obj = _total_c6(target_n, current_colors)
            
            delta = new_obj - current_obj
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T, T_min)):
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_colors = current_colors[:]
                    if best_obj == 0:
                        # Found a valid coloring!
                        n_best = target_n
                        colors_best = best_colors[:]
                        break
            else:
                current_colors[edge_idx] = old_color
            
            T *= cooling
            step += 1
        
        if best_obj == 0:
            n_best = target_n
            colors_best = best_colors[:]
            # Continue to try larger n
        else:
            # Could not find valid coloring for this n, try next n with different init
            # But we might still try again with more iterations
            pass
    
    # Final check: verify the best result
    if _check_no_c6(n_best, colors_best):
        return {'n': n_best, 'colors': colors_best}
    else:
        # Fallback to K9
        n, colors = _k9_baseline()
        return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
