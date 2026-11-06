import random
import sys
import time

N = 16
sys.setrecursionlimit(10000)

def construct():
    # Generate all edges in lexicographic order
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    
    num_edges = len(edges)
    best_count = 0
    best_assignment = [0] * num_edges
    
    # To improve performance, we can pre-compute the paths for each edge
    # paths_for_edge[e_idx] = list of (w, x) such that u-w-x-v is a path
    # where (u,v) = edges[e_idx]
    paths_for_edge = [[] for _ in range(num_edges)]
    for idx, (u, v) in enumerate(edges):
        for w in range(N):
            if w == u or w == v:
                continue
            for x in range(N):
                if x == u or x == v or x == w:
                    continue
                paths_for_edge[idx].append((w, x))

    def get_valid_colors(idx, colors):
        u, v = edges[idx]
        # Intersection of constraints
        # Start with all possible colors 0..119
        # But we only care about colors that are either already used or new.
        # Actually, we can just compute the intersection of forbidden sets' complements?
        # Constraint: If path has 3 distinct colors, c must be in that set.
        # So valid colors = Intersection of (Path Color Sets) for all "rainbow" paths.
        # If no rainbow paths, any color is valid.
        
        valid = None
        for w, x in paths_for_edge[idx]:
            c1 = colors[u][w]
            c2 = colors[w][x]
            c3 = colors[x][v]
            
            # If any edge is uncolored, this path doesn't constrain us yet
            # (Assuming we only color sequentially, uncolored means -1)
            if c1 == -1 or c2 == -1 or c3 == -1:
                continue
                
            s = {c1, c2, c3}
            if len(s) == 3:
                if valid is None:
                    valid = s.copy()
                else:
                    valid.intersection_update(s)
                    if not valid:
                        return []
        
        if valid is None:
            # No constraints, any color 0..119 is valid
            return list(range(120))
        else:
            return list(valid)

    def dfs(idx, colors, used_colors):
        nonlocal best_count, best_assignment
        
        if idx == num_edges:
            if len(used_colors) > best_count:
                best_count = len(used_colors)
                # Copy current assignment
                res = []
                for i in range(N):
                    for j in range(i + 1, N):
                        res.append(colors[i][j])
                best_assignment = res
            return True

        # MRV Heuristic: Pick the next edge with the fewest valid colors
        # To save time, we just pick the current idx, but we can shuffle the edge order initially.
        # For a true CSP, we'd pick min(len(valid)). 
        # Let's stick to the given order for speed, but use the "New Color First" value ordering.
        
        u, v = edges[idx]
        valid_colors = get_valid_colors(idx, colors)
        
        if not valid_colors:
            return False

        # Value Ordering:
        # 1. New colors (not in used_colors)
        # 2. Existing colors
        new_colors = [c for c in valid_colors if c not in used_colors]
        old_colors = [c for c in valid_colors if c in used_colors]
        
        # Shuffle to break symmetry and explore different branches
        random.shuffle(new_colors)
        random.shuffle(old_colors)
        
        # Try new colors first to maximize the count
        # We limit the number of new colors to try to avoid deep backtracking on "greedy" paths that fail later?
        # No, just try all.
        
        for c in new_colors + old_colors:
            colors[u][v] = c
            colors[v][u] = c
            if c not in used_colors:
                used_colors.add(c)
            
            if dfs(idx + 1, colors, used_colors):
                return True
            
            colors[u][v] = -1
            colors[v][u] = -1
            if c not in used_colors:
                # Only remove if it wasn't added in this step? 
                # Wait, if c was in used_colors before, we shouldn't remove it.
                # We added it only if c not in used_colors.
                # So if c is not in used_colors now, it means it was added here.
                pass # The logic above adds it. We need to remove it if we added it.
            
            # Correct removal logic:
            # We check if c was in used_colors BEFORE assignment.
            # Simpler: track if we added it.
            # Let's refactor the loop to be safe.
            
        return False

    # Refactored DFS to handle used_colors correctly
    def dfs_safe(idx, colors, used_colors):
        nonlocal best_count, best_assignment
        
        if idx == num_edges:
            if len(used_colors) > best_count:
                best_count = len(used_colors)
                res = []
                for i in range(N):
                    for j in range(i + 1, N):
                        res.append(colors[i][j])
                best_assignment = res
            return True

        u, v = edges[idx]
        valid_colors = get_valid_colors(idx, colors)
        
        if not valid_colors:
            return False

        new_colors = [c for c in valid_colors if c not in used_colors]
        old_colors = [c for c in valid_colors if c in used_colors]
        
        random.shuffle(new_colors)
        random.shuffle(old_colors)
        
        candidates = new_colors + old_colors
        
        for c in candidates:
            is_new = c not in used_colors
            colors[u][v] = c
            colors[v][u] = c
            if is_new:
                used_colors.add(c)
            
            if dfs_safe(idx + 1, colors, used_colors):
                return True
            
            colors[u][v] = -1
            colors[v][u] = -1
            if is_new:
                used_colors.remove(c)
                
        return False

    # Run multiple restarts with different random seeds/orders if needed
    # But a single deep search with good heuristics might suffice.
    # To be safe against getting stuck in a bad local optimum (though this is exact search, not local),
    # we just run it once. If it fails, we return the best found (which might be 0 if it failed immediately? No, best_count updates on success).
    # If dfs_safe returns False, it means no solution was found with this specific randomization?
    # No, dfs_safe explores ALL possibilities. If it returns False, no valid coloring exists?
    # But the problem states a solution exists.
    # The only reason for False is if we hit a dead end and backtracked all the way.
    # Since the graph is small, it should find a solution.
    
    colors = [[-1 for _ in range(N)] for _ in range(N)]
    used_colors = set()
    
    # Time limit check? The recursion is depth 120.
    # If it's too slow, we might need to limit branching.
    # But 120 edges, branching factor <= 120. 120^120 is huge.
    # However, constraints are tight.
    # Let's add a simple time check in the loop?
    # Or just run it. Python recursion is slow.
    # Optimization: Use iterative deepening or just accept the risk.
    # Given the seed score 1.0, and this is a strict improvement in search strategy, it should work.
    
    # To prevent timeout, we can add a node counter and stop if it exceeds a limit,
    # returning the best found so far.
    
    nodes_visited = 0
    MAX_NODES = 1000000 # 1 million nodes
    
    def dfs_limited(idx, colors, used_colors):
        nonlocal best_count, best_assignment, nodes_visited
        
        if nodes_visited > MAX_NODES:
            return False
        
        nodes_visited += 1
        
        if idx == num_edges:
            if len(used_colors) > best_count:
                best_count = len(used_colors)
                res = []
                for i in range(N):
                    for j in range(i + 1, N):
                        res.append(colors[i][j])
                best_assignment = res
            return True

        u, v = edges[idx]
        valid_colors = get_valid_colors(idx, colors)
        
        if not valid_colors:
            return False

        new_colors = [c for c in valid_colors if c not in used_colors]
        old_colors = [c for c in valid_colors if c in used_colors]
        
        random.shuffle(new_colors)
        random.shuffle(old_colors)
        
        candidates = new_colors + old_colors
        
        for c in candidates:
            is_new = c not in used_colors
            colors[u][v] = c
            colors[v][u] = c
            if is_new:
                used_colors.add(c)
            
            if dfs_limited(idx + 1, colors, used_colors):
                return True
            
            colors[u][v] = -1
            colors[v][u] = -1
            if is_new:
                used_colors.remove(c)
                
        return False

    # Try a few times with different random seeds to ensure we find a good one
    for attempt in range(5):
        random.seed(attempt)
        colors = [[-1 for _ in range(N)] for _ in range(N)]
        used_colors = set()
        nodes_visited = 0
        dfs_limited(0, colors, used_colors)
        
        if best_count > 0:
            # If we found a good solution, we might break early if it's high
            if best_count >= 100:
                break

    if best_assignment is None or all(x == 0 for x in best_assignment):
         # Fallback if something went wrong
         return [0] * num_edges

    return best_assignment
