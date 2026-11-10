import random
import time

def construct():
    n = 16
    # Target number of colors. We know 17 works. Try 14 first, then 15.
    # The bound is 13 <= f(16) <= 15.
    
    # Precompute all K4s (sets of 4 vertices)
    k4s = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    k4s.append((a, b, c, d))
    
    # Helper to get edges of a K4
    def get_edges(k4):
        a, b, c, d = k4
        return [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        
    # Map edge to index in the flat list returned by construct
    # The return format is lexicographic order of pairs (i,j)
    # Index for (i,j) is sum_{m=0}^{i-1} (15-m) + (j-i-1) ? 
    # Actually, simpler: just use a dictionary or a 2D array and flatten at the end.
    # But we need to return a list of 120 integers.
    
    # Let's use a 2D array for internal representation, easier to access.
    # colors[i][j] for i < j.
    
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    # Try k = 14 first. If it fails within time, try k = 15? 
    # Or just stick to one k per run? 
    # The evaluator runs the function once. 
    # If I try 14 and fail, I should return a valid 15-colouring or 17-colouring.
    # Since 17 is known valid, I can always fall back to 17.
    # But I want to minimize.
    
    # Strategy:
    # 1. Try to find a 14-colouring.
    # 2. If not found in time/iterations, try 15-colouring.
    # 3. If not found, return 17-colouring.
    
    def try_find_coloring(k, time_budget):
        # Initialize random coloring
        colors = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                colors[i][j] = random.randint(0, k-1)
        
        # Precompute edge lists for each K4 for speed?
        # k4_edges = [get_edges(k4) for k4 in k4s]
        
        # Function to check if a K4 is valid
        def is_valid_k4(k4):
            edges = get_edges(k4)
            c_set = set()
            for u, v in edges:
                c_set.add(colors[u][v])
            return len(c_set) >= 5
        
        # Main loop
        while time.time() - start_time < time_budget:
            # Find a violating K4
            # Randomly sample K4s to find a violation quickly
            # If no violation in a full pass, we are done
            violated_k4 = None
            # Check all K4s? 1820 is small.
            # But checking set size is fast.
            # Let's just iterate.
            for k4 in k4s:
                if not is_valid_k4(k4):
                    violated_k4 = k4
                    break
            
            if violated_k4 is None:
                return colors
            
            # Pick a random edge in the violated K4
            edges = get_edges(violated_k4)
            u, v = random.choice(edges)
            
            # Current colors in this K4
            c_set = set(colors[a][b] for a, b in edges)
            
            # Pick a new color not in c_set if possible
            available_colors = [c for c in range(k) if c not in c_set]
            if available_colors:
                new_color = random.choice(available_colors)
            else:
                # All colors are present? Then len(c_set) >= k.
                # If k < 5, this can't happen for a violation (len < 5).
                # If k >= 5 and it's a violation, len(c_set) < 5.
                # So available_colors is never empty if k >= 5.
                new_color = random.randint(0, k-1)
            
            # Update
            colors[u][v] = new_color
            colors[v][u] = new_color # Symmetry, though we only read i<j
            
        return None # Timeout

    # Try k=14
    res14 = try_find_coloring(14, 50)
    if res14 is not None:
        # Flatten
        return [res14[i][j] for i in range(n) for j in range(i+1, n)]
        
    # Try k=15
    res15 = try_find_coloring(15, 50)
    if res15 is not None:
        return [res15[i][j] for i in range(n) for j in range(i+1, n)]
        
    # Fallback to k=17 (known valid)
    # The seed program uses (i+j)%17
    return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
