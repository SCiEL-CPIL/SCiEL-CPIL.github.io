import numpy as np
import time

def construct():
    """
    Construct a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Uses a deterministic local search (Min-Conflicts heuristic) to find a valid coloring.
    """
    n_max = 29
    # We will try to find a valid coloring for n from n_max down to 1
    # We use a fixed seed for determinism
    np.random.seed(42)
    
    def count_conflicts(color, n):
        """
        Count the number of monochromatic C4s in the coloring of K_n.
        A C4 is defined by 4 distinct vertices a, b, c, d such that
        (a,b), (b,c), (c,d), (d,a) are all the same color.
        Alternatively, for each color c, check if the graph of edges with color c contains a C4.
        A C4 in a graph exists if there are two vertices with >= 2 common neighbors.
        """
        total_conflicts = 0
        for c in range(5):
            # Build adjacency matrix for color c
            # adj[u][v] = 1 if color[u][v] == c
            # We can check common neighbors efficiently
            # For each pair (u, v), count common neighbors in color c.
            # If count >= 2, we have a C4.
            # Actually, just counting pairs with >= 2 common neighbors is sufficient.
            # To be faster, we can use matrix multiplication or bitsets.
            # For n <= 29, O(n^3) is fine.
            
            # Let's use a simple loop
            # For each u, v (u < v), count common neighbors w != u, v
            # such that color[w][u] == c and color[w][v] == c.
            # If count >= 2, it's a conflict.
            # We count the number of such pairs (u, v).
            
            # Optimization: Precompute neighbor lists for each color?
            # Or just loop.
            for u in range(n):
                for v in range(u + 1, n):
                    common = 0
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if color[w][u] == c and color[w][v] == c:
                            common += 1
                            if common >= 2:
                                break
                    if common >= 2:
                        total_conflicts += 1
        return total_conflicts

    def min_conflicts_search(n, max_iters=10000):
        """
        Try to find a valid 5-coloring of K_n using Min-Conflicts.
        Returns the color array if successful, None otherwise.
        """
        # Initialize random coloring
        color = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                c = np.random.randint(0, 5)
                color[i][j] = c
                color[j][i] = c
        
        conflicts = count_conflicts(color, n)
        
        # If already valid, return
        if conflicts == 0:
            return color
        
        # Min-Conflicts
        # We need to find edges involved in conflicts.
        # This is expensive to recompute fully every step.
        # We'll just pick a random edge and try all 5 colors, picking the best.
        
        for _ in range(max_iters):
            # Pick a random edge
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            old_c = color[u][v]
            best_c = old_c
            best_conflicts = conflicts
            
            # Try all 5 colors
            for c in range(5):
                if c == old_c:
                    continue
                # Tentatively set color
                color[u][v] = c
                color[v][u] = c
                
                new_conflicts = count_conflicts(color, n)
                
                if new_conflicts < best_conflicts:
                    best_conflicts = new_conflicts
                    best_c = c
                else:
                    # Revert
                    color[u][v] = old_c
                    color[v][u] = old_c
                    continue
            # If we found a better color, it's already set in the loop above (if we didn't revert)
            # Wait, the logic above reverts if not better. If better, it stays set.
            # But we need to update the global 'conflicts' variable.
            
            if best_c != old_c:
                color[u][v] = best_c
                color[v][u] = best_c
                conflicts = best_conflicts
            else:
                # Revert to old if no change was made
                color[u][v] = old_c
                color[v][u] = old_c
            
            if conflicts == 0:
                return color
                
        return None

    # Try to find a valid coloring for decreasing n
    for n in range(n_max, 0, -1):
        # Use a different seed for each n to explore different landscapes
        np.random.seed(42 + n)
        result = min_conflicts_search(n, max_iters=5000)
        if result is not None:
            # Extract colors in lexicographic order
            colors_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors_list.append(int(result[i][j]))
            return {'n': n, 'colors': colors_list}
            
    # Fallback: return n=1
    return {'n': 1, 'colors': []}
