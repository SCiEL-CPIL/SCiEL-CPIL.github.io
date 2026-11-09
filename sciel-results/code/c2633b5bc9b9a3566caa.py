import time
import random
import sys

def _is_valid_coloring(n, colors):
    """
    Verify if the coloring is valid.
    colors is a list of length n*(n-1)/2.
    """
    if n < 2:
        return True
    # Build adjacency sets for each color
    color_adj = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            color_adj[c][i].add(j)
            color_adj[c][j].add(i)
            idx += 1
    # Check for C4: for each color, no two vertices should have 2+ common neighbors
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = color_adj[c][u] & color_adj[c][v]
                if len(common) >= 2:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 seconds margin
    
    best_n = 1
    best_colors = []
    
    # Helper to flatten cm to list
    def flatten(cm, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(cm[i][j])
        return colors

    # Helper to build conflict matrix for current graph
    # conflict[c][u][v] = 1 if u and v have a common neighbor in color c
    def build_conflicts(cm, n):
        adj = [[set() for _ in range(n)] for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = cm[i][j]
                adj[c][i].add(j)
                adj[c][j].add(i)
        
        conflicts = [[[0] * n for _ in range(n)] for _ in range(5)]
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[c][u] & adj[c][v]:
                        conflicts[c][u][v] = 1
                        conflicts[c][v][u] = 1
        return conflicts, adj

    # Try to extend coloring from size n to n+1
    def try_extend(cm, n, conflicts, adj, max_attempts=20):
        new_n = n + 1
        # new_cm is a copy of cm extended
        new_cm = [row[:] + [-1] for row in cm]
        new_cm.append([-1] * new_n)
        
        # We need to assign colors to edges (new_n-1, i) for i in 0..n-1
        # Let new_v = new_n - 1
        new_v = new_n - 1
        
        # assigned[i] = color of edge (new_v, i)
        assigned = [-1] * n
        
        # Precompute potential valid colors for each i?
        # A color c is valid for i if for all j where assigned[j]==c, conflicts[c][i][j] == 0.
        # This depends on the assignment of other vertices.
        
        # Backtracking state
        # Order of vertices to assign: random shuffle
        order = list(range(n))
        
        for attempt in range(max_attempts):
            random.shuffle(order)
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    # Success
                    for i in range(n):
                        c = assigned[i]
                        new_cm[new_v][i] = c
                        new_cm[i][new_v] = c
                    return True
                
                u = order[idx]
                # Try colors 0-4
                # Heuristic: try colors that are less constrained?
                # For now, just try 0-4 with random tie breaking or fixed order.
                colors_to_try = [0, 1, 2, 3, 4]
                random.shuffle(colors_to_try)
                
                for c in colors_to_try:
                    valid = True
                    # Check against all previously assigned vertices j in order[0..idx-1]
                    # such that assigned[j] == c
                    # Actually, we only need to check against j where assigned[j] == c.
                    # We can iterate through all j < idx in the order, or maintain a list of assigned vertices per color.
                    
                    # Optimization: Maintain assigned_vertices[c] = list of indices assigned color c so far.
                    # But since we are in backtracking, we need to add/remove.
                    # Simpler: just check all j in 0..n-1 where assigned[j] == c.
                    # Since n is small (<30), this is fast.
                    
                    for j in range(n):
                        if assigned[j] == c:
                            if conflicts[c][u][j] == 1:
                                valid = False
                                break
                    
                    if valid:
                        assigned[u] = c
                        if backtrack(idx + 1):
                            return True
                        assigned[u] = -1
                
                return False
            
            if backtrack(0):
                return new_cm, True
        
        return None, False

    # Main loop
    # Start with n=1
    cm = [[0]]
    n = 1
    conflicts, adj = build_conflicts(cm, n)
    
    # We can jump start to a known good n if we want, but let's try to build.
    # To ensure we reach high n, we might need to restart if stuck.
    
    stuck_count = 0
    while n < 29:
        if time.time() > deadline:
            break
            
        success = False
        # Try to extend
        # Use a moderate number of attempts for each extension step
        attempts_for_step = 50 if n < 20 else 100
        new_cm, success = try_extend(cm, n, conflicts, adj, attempts_for_step)
        
        if success:
            n += 1
            cm = new_cm
            # Update conflicts and adj
            conflicts, adj = build_conflicts(cm, n)
            stuck_count = 0
            # Update best
            if n > best_n:
                best_n = n
                best_colors = flatten(cm, n)
        else:
            stuck_count += 1
            if stuck_count > 20:
                # If stuck for a while, maybe restart from a smaller base?
                # Or just break and return what we have.
                # Given the time limit, breaking is safer.
                # However, we can try to restart the whole process from n=1 with a different seed?
                # That might be too slow.
                # Let's just break.
                break

    # If we didn't reach a high n, ensure we return a valid small one
    if best_n < 2:
        best_n = 2
        best_colors = [0]
    elif best_n < n:
        # This shouldn't happen as we update best_n on success
        pass

    return {'n': best_n, 'colors': best_colors}
