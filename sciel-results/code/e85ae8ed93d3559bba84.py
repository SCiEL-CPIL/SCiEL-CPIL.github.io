import time
import random

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4s using a high-performance
    bitmask-based backtracking solver with MRV heuristic.
    """
    
    def popcount(x):
        return bin(x).count('1')
        
    def solve(N, time_limit=100):
        start_time = time.time()
        # neighbors[c] is a list of N integers (bitmasks)
        # neighbors[c][u] has bit v set if (u,v) is color c
        neighbors = [ [0] * N for _ in range(5) ]
        
        # Edges to be assigned. We process them in a specific order.
        # For bitmask efficiency, we don't need a list of edges if we iterate 
        # through vertices or use a precomputed list of (u,v) pairs.
        edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        
        # Assignment array for final output
        # We store color for (u,v) in a 2D array or flat list
        color_matrix = [[-1]*N for _ in range(N)]
        
        def is_valid(u, v, c):
            """Check if adding (u,v) with color c creates a C4."""
            # Common neighbors of u and v in color c
            common = neighbors[c][u] & neighbors[c][v]
            # If there are 2 or more common neighbors, a C4 is formed
            # (u, common1, v, common2, u)
            return popcount(common) < 2

        def backtrack(unassigned_idx, unassigned_edges):
            if time.time() - start_time > time_limit:
                return False
            
            if unassigned_idx == len(unassigned_edges):
                return True
                
            # MRV Heuristic: Find the edge with the fewest valid colors
            # To make this fast, we can just pick the current edge if we have a good static order,
            # or do a light MRV.
            # For N <= 15, E <= 105. Checking all remaining is O(E^2) in worst case per step?
            # No, we can just iterate the remaining edges to find the one with min options.
            
            best_idx = -1
            min_opts = 5
            best_valid = None
            
            # Optimization: If N is small, full MRV is fast.
            # If N is large, we might sample.
            # Let's do full MRV for N <= 16.
            for i in range(unassigned_idx, len(unassigned_edges)):
                u, v = unassigned_edges[i]
                valid_colors = []
                for c in range(5):
                    if is_valid(u, v, c):
                        valid_colors.append(c)
                        
                if len(valid_colors) < min_opts:
                    min_opts = len(valid_colors)
                    best_idx = i
                    best_valid = valid_colors
                    if min_opts == 0:
                        return False
                    if min_opts == 1:
                        break
                        
            if best_idx == -1:
                return False
                
            # Swap best edge to current position
            unassigned_edges[unassigned_idx], unassigned_edges[best_idx] = \
                unassigned_edges[best_idx], unassigned_edges[unassigned_idx]
            
            u, v = unassigned_edges[unassigned_idx]
            
            # Shuffle valid colors for randomization
            random.shuffle(best_valid)
            
            for c in best_valid:
                # Assign
                neighbors[c][u] |= (1 << v)
                neighbors[c][v] |= (1 << u)
                color_matrix[u][v] = c
                color_matrix[v][u] = c
                
                if backtrack(unassigned_idx + 1, unassigned_edges):
                    return True
                
                # Undo
                neighbors[c][u] &= ~(1 << v)
                neighbors[c][v] &= ~(1 << u)
                color_matrix[u][v] = -1
                color_matrix[v][u] = -1
                
            return False

        # Try to solve for N
        # Use a specific edge order: distance-based or lexicographic?
        # Lexicographic is often good for structure.
        base_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        
        max_restarts = 200
        for _ in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Reset state
            neighbors = [ [0] * N for _ in range(5) ]
            color_matrix = [[-1]*N for _ in range(N)]
            
            # Shuffle edges to break symmetry
            current_edges = base_edges[:]
            random.shuffle(current_edges)
            
            if backtrack(0, current_edges):
                # Success
                final_colors = []
                for i in range(N):
                    for j in range(i+1, N):
                        final_colors.append(color_matrix[i][j])
                return {'n': N, 'colors': final_colors}
                
        return None

    total_time_limit = 110
    start_total = time.time()
    
    # Try larger N first
    # We know R_5(C4) > 26, so 26 is possible but hard.
    # Bitmask solver is fast, so we can try 20, 21, etc.
    # However, search space grows exponentially.
    # Let's try 15, 16, 17...
    
    for N in range(17, 10, -1):
        time_remaining = total_time_limit - (time.time() - start_total)
        if time_remaining < 2:
            break
        
        # Allocate time based on difficulty
        # Smaller N gets less time, Larger N gets more?
        # Actually, if 17 is easy, we stop. If 17 is hard, we might fail.
        # Let's give 10s to 17, 10s to 16, etc.
        alloc_time = min(10, time_remaining)
        
        res = solve(N, time_limit=alloc_time)
        if res is not None:
            return res
            
    # Fallback: Hardcoded solution for n=9 (Paley-based or known)
    # n=9, 36 edges.
    # A valid 5-coloring for n=9 is easy to find.
    # Let's just return a simple valid one if search fails.
    # We can construct one using the fact that K_9 can be decomposed into 3 Hamiltonian cycles?
    # No, C4-free.
    # Let's just use a random greedy for n=9 as last resort, it will almost certainly succeed.
    
    def greedy_fallback(N):
        neighbors = [ [0] * N for _ in range(5) ]
        color_matrix = [[-1]*N for _ in range(N)]
        edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(edges)
        for u, v in edges:
            for c in range(5):
                if popcount(neighbors[c][u] & neighbors[c][v]) < 2:
                    neighbors[c][u] |= (1 << v)
                    neighbors[c][v] |= (1 << u)
                    color_matrix[u][v] = c
                    color_matrix[v][u] = c
                    break
        # Verify
        for i in range(N):
            for j in range(i+1, N):
                if color_matrix[i][j] == -1:
                    return None
        final_colors = []
        for i in range(N):
            for j in range(i+1, N):
                final_colors.append(color_matrix[i][j])
        return {'n': N, 'colors': final_colors}

    res = greedy_fallback(12)
    if res:
        return res
        
    # Absolute fallback: n=1
    return {'n': 1, 'colors': []}

