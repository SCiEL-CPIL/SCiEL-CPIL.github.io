import time
import random
import copy
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin
    
    # We will try to maximize n.
    # Strategy: 
    # 1. Build a graph greedily with randomization and backtracking.
    # 2. Use a "kick" mechanism to escape local optima.
    # 3. Keep track of the best valid graph found.
    
    best_n = 0
    best_graph = []
    
    def is_valid(g, n):
        """Check if graph g of size n has no monochromatic triangles."""
        for i in range(n):
            for j in range(i + 1, n):
                c = g[i][j]
                for k in range(j + 1, n):
                    if g[i][k] == c and g[j][k] == c:
                        return False
        return True

    def try_add_vertex(g, n, max_backtrack_depth=100):
        """
        Attempts to add a vertex to graph g (size n).
        Uses backtracking. Returns True if successful.
        Modifies g in place if successful.
        """
        # We want to assign colors for edges (i, n) for i in 0..n-1.
        # We process i from 0 to n-1.
        # assignments[i] is the color of edge (i, n).
        
        assignments = [-1] * n
        
        def backtrack(i, depth):
            if i == n:
                return True
            
            # Heuristic: Sort colors by how "safe" they are.
            # A color c is safe for edge (i, n) if for all k < i, 
            # it's not the case that g[k][i] == c and assignments[k] == c.
            # Actually, we just need to check if assigning c creates a triangle 
            # with any previously assigned edge (k, n) where k < i.
            # Triangle (k, i, n): edges (k,i), (i,n), (k,n).
            # Colors: g[k][i], assignments[i], assignments[k].
            # Conflict if g[k][i] == assignments[i] == assignments[k].
            
            # Let's count how many k < i would cause a conflict if we pick color c.
            # If g[k][i] == c and assignments[k] == c, then conflict.
            
            color_scores = []
            for c in range(4):
                conflicts = 0
                for k in range(i):
                    if g[k][i] == c and assignments[k] == c:
                        conflicts += 1
                color_scores.append((conflicts, c))
            
            # Sort by conflicts (ascending), then random tie-breaker
            random.shuffle(color_scores)
            color_scores.sort(key=lambda x: x[0])
            
            for _, c in color_scores:
                assignments[i] = c
                # Check validity against all k < i
                valid = True
                for k in range(i):
                    if g[k][i] == c and assignments[k] == c:
                        valid = False
                        break
                
                if valid:
                    if backtrack(i + 1, depth + 1):
                        return True
                
            assignments[i] = -1
            return False

        # Start backtracking
        # To avoid deep recursion timeouts, we limit depth or rely on the global deadline.
        # The backtracking here is limited to the current vertex addition.
        if backtrack(0, 0):
            # Success
            for i in range(n):
                g[i].append(assignments[i])
            new_row = [0] * (n + 1)
            for i in range(n):
                new_row[i] = assignments[i]
            g.append(new_row)
            return True
        return False

    def kick(g, n, num_flips):
        """
        Randomly flip num_flips edges in g.
        Returns True if the graph remains valid.
        """
        old_g = copy.deepcopy(g)
        
        for _ in range(num_flips):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                g[i][j] = random.randint(0, 3)
                g[j][i] = g[i][j]
        
        if is_valid(g, n):
            return True
        else:
            # Restore
            g[:] = old_g
            return False

    # Main Loop
    # Start with a small n and grow
    current_n = 1
    current_graph = [[0]]
    
    # We can try to seed with a known good small graph to start faster?
    # Or just start from scratch. Starting from scratch with greedy is usually fine.
    
    attempts = 0
    max_attempts = 50000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Try to add a vertex
        if try_add_vertex(current_graph, current_n):
            current_n += 1
            if current_n > best_n:
                best_n = current_n
                best_graph = copy.deepcopy(current_graph)
        
        else:
            # Failed to add vertex. 
            # Try a kick to escape local optimum.
            # The number of flips should be small to keep the graph mostly intact.
            # As n grows, the graph becomes more constrained, so fewer flips might be needed?
            # Or maybe more? Let's try a fixed small number, e.g., 5-10% of edges.
            num_edges = current_n * (current_n - 1) // 2
            num_flips = max(1, num_edges // 20) # 5% of edges
            
            # Try a few kicks
            kick_success = False
            for _ in range(10):
                if kick(current_graph, current_n, num_flips):
                    kick_success = True
                    break
            
            if not kick_success:
                # If we can't even kick validly, the graph is very tight.
                # Maybe we should restart with a smaller n?
                # Or just keep trying kicks?
                # Let's try a more aggressive kick or restart.
                # For now, just continue. The loop will try again.
                pass

    # Final check and return
    # Ensure best_graph is valid
    if best_n > 0 and not is_valid(best_graph, best_n):
        # Fallback to a known valid small graph if something went wrong
        # Construct n=1
        best_n = 1
        best_graph = [[0]]
        
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            colors.append(best_graph[i][j])
            
    return {'n': best_n, 'colors': colors}
