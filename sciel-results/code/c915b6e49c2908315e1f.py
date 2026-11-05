import time
import random
import math

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Uses Min-Conflicts heuristic which is robust and fast for this size.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds, leaving 5s margin

    def time_left():
        return deadline - time.time()

    def validate(colors, n):
        """Check if no color class contains a C4."""
        # For each color, build adjacency bitmasks
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        # Check for C4: any two vertices with >= 2 common neighbors in same color
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[c][i] & adj[c][j]
                    # If there are at least 2 common neighbors, a C4 exists
                    if common & (common - 1):
                        return False
        return True

    def count_conflicts(colors, n):
        """
        Count the number of pairs of vertices (i, j) that have >= 2 common neighbors in the same color.
        This is the objective function to minimize.
        """
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        conflicts = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[c][i] & adj[c][j]
                    if common & (common - 1):
                        conflicts += 1
        return conflicts

    def min_conflicts(n, max_time):
        """
        Run Min-Conflicts heuristic for n vertices.
        Returns a valid coloring if found, else None.
        """
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Precompute edge indices for fast access
        # index = i*n - i*(i+1)//2 + (j-i-1)
        def get_idx(i, j):
            if i > j:
                i, j = j, i
            return i * n - i * (i + 1) // 2 + (j - i - 1)
        
        # To make Min-Conflicts efficient, we need to identify vertices involved in conflicts.
        # A vertex v is involved in a conflict if there exists another vertex u such that
        # they have >= 2 common neighbors in the color of (u, v)?
        # Actually, the conflict is on the pair (u, v) if they share neighbors.
        # The "move" in Min-Conflicts is usually changing the color of an *edge*.
        # If we change the color of edge (u, v), we affect the common neighbor counts
        # for all pairs involving u or v.
        
        # Let's use a simpler variant: 
        # 1. Calculate current conflicts.
        # 2. If 0, return.
        # 3. Pick a random edge (u, v) that is part of a conflict.
        #    An edge (u, v) is "bad" if cn[color(u,v)][u][v] >= 2? No, the conflict is defined on the pair (u,v) 
        #    having common neighbors. The edge (u,v) itself doesn't "cause" the conflict directly in the same way,
        #    but changing its color changes the graph structure.
        #    
        #    Actually, a standard Min-Conflicts for graph coloring changes vertex colors.
        #    Here, we are coloring edges.
        #    Let's stick to: 
        #    - Maintain adj bitmasks.
        #    - Maintain conflict count.
        #    - Pick a random edge (u, v).
        #    - Try all 5 colors for (u, v).
        #    - Choose the color that minimizes the total conflict count (or reduces it).
        #    - If no improvement, accept random or best (plateau).
        
        # To be fast, we need incremental updates.
        # However, for n=26, num_edges = 325.
        # A full recalculation of conflicts takes O(5 * n^2 * word_size) which is very fast.
        # 5 * 26^2 = 3380 operations. 
        # Even doing this 10,000 times is ~33 million ops, which is < 1s in Python?
        # Python is slower. 33 million simple ops might take 1-2 seconds.
        # So we can afford ~10,000-50,000 restarts/steps in 115s.
        
        # Let's try a simple local search with full recalculation of conflicts for each move.
        # This is less bug-prone than incremental deltas.
        
        def get_conflicts_and_bad_edges(colors, n):
            adj = [[0] * n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            
            conflicts = 0
            bad_edges = []
            for c in range(5):
                for i in range(n):
                    for j in range(i + 1, n):
                        common = adj[c][i] & adj[c][j]
                        if common & (common - 1):
                            conflicts += 1
                            # The pair (i, j) is the source of conflict.
                            # Which edge to change? 
                            # Usually, we pick an edge incident to i or j.
                            # Let's just store the pair (i, j) as a "bad pair".
                            bad_edges.append((i, j))
            return conflicts, bad_edges, adj

        current_colors = colors[:]
        current_conflicts, _, _ = get_conflicts_and_bad_edges(current_colors, n)
        
        if current_conflicts == 0:
            return current_colors

        best_colors = current_colors[:]
        best_conflicts = current_conflicts

        t_end = time.time() + max_time
        
        while time.time() < t_end:
            if current_conflicts == 0:
                return current_colors
            
            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best_colors = current_colors[:]
            
            # Pick a random bad pair (i, j)
            # If no bad pairs, we are stuck (should not happen if conflicts > 0)
            # We need to pick an edge to change.
            # Changing an edge (u, v) affects conflicts involving u or v.
            # Let's pick a random vertex u that is part of a conflict, and a random neighbor v.
            # Or simpler: Pick a random edge (u, v).
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            ei = get_idx(u, v)
            old_c = current_colors[ei]
            
            # Try all 5 colors
            best_new_c = old_c
            min_conflicts_for_edge = current_conflicts
            
            # We need to estimate the change in conflicts.
            # Full recalculation for each of 5 colors is 5 * O(n^2).
            # For n=26, this is fast enough.
            
            for new_c in range(5):
                if new_c == old_c:
                    continue
                
                # Temporarily change
                current_colors[ei] = new_c
                
                # Recalculate conflicts
                c_conf, _, _ = get_conflicts_and_bad_edges(current_colors, n)
                
                if c_conf < min_conflicts_for_edge:
                    min_conflicts_for_edge = c_conf
                    best_new_c = new_c
                
            # Revert to best
            current_colors[ei] = best_new_c
            current_conflicts = min_conflicts_for_edge
            
            # If no improvement was found (plateau), do a random move to escape
            if best_new_c == old_c:
                # Randomly pick a new color different from old
                new_c = random.randint(0, 4)
                if new_c == old_c:
                    new_c = (new_c + 1) % 5
                current_colors[ei] = new_c
                current_conflicts, _, _ = get_conflicts_and_bad_edges(current_colors, n)

        # End of time for this n
        if best_conflicts == 0:
            return best_colors
        return None

    # Try n from 28 down to 26
    # We allocate more time to larger n? Or just try each?
    # Since 26 is the known target, we try 28, 27, 26.
    # If we find a valid one, we return it.
    
    # Time allocation: 
    # 28: 10s
    # 27: 10s
    # 26: 90s (most likely to succeed, and we want to be sure)
    
    time_alloc = {
        28: 10,
        27: 10,
        26: 90,
        25: 5,
        24: 5
    }
    
    for n in [28, 27, 26, 25, 24]:
        if time_left() < 5:
            break
        t_alloc = time_alloc.get(n, 5)
        if time_left() < t_alloc + 1:
            t_alloc = max(1, int(time_left() - 1))
            
        result = min_conflicts(n, t_alloc)
        if result is not None:
            # Double check validation
            if validate(result, n):
                return {'n': n, 'colors': result}
            else:
                # Should not happen, but if it does, ignore
                continue
                
    # Fallback to a small valid n
    # n=1 is always valid
    return {'n': 1, 'colors': [0]}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
