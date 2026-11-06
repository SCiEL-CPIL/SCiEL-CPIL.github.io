import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate the base 32-vertex coloring from the seed
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
        
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    def get_base_color(i, j):
        # i, j are 0..31
        if i // 16 == j // 16:
            return log[(i % 16) ^ (j % 16)] % 3
        else:
            return 3
            
    # Initialize n=32
    n_current = 32
    colors = {}
    for i in range(n_current):
        for j in range(i + 1, n_current):
            colors[(i, j)] = get_base_color(i, j)
            
    best_n = n_current
    best_colors = {k: v for k, v in colors.items()}
    
    def get_color(i, j):
        if i > j: i, j = j, i
        return colors.get((i, j))
        
    def set_color(i, j, c):
        if i > j: i, j = j, i
        colors[(i, j)] = c
        
    def count_mono_triangles(n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[(i, j)]
                for k in range(j + 1, n):
                    if colors[(i, k)] == c_ij and colors[(j, k)] == c_ij:
                        count += 1
        return count
        
    def local_search(n, max_iterations=2000):
        # Returns True if valid (0 triangles), False otherwise
        # Uses a simple hill-climbing approach
        current_penalty = count_mono_triangles(n)
        if current_penalty == 0:
            return True
            
        # Precompute neighbors for faster checks? 
        # For n~50, simple check is O(n^3) ~ 125k ops. 
        # 2000 iterations * 125k = 250M ops. Might be slow in Python.
        # Let's optimize penalty update or use a smaller step.
        
        # Optimized Local Search:
        # Pick a random triangle that is monochromatic, flip one of its edges.
        # Or pick a random edge, check if flipping reduces penalty.
        
        # Let's try: Pick random edge, flip, check delta.
        # Delta calculation is expensive if done from scratch.
        # Let's just do a few full checks.
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                return False
            if current_penalty == 0:
                return True
                
            # Find a monochromatic triangle
            found = False
            for _ in range(100): # Try 100 random triangles
                i = random.randint(0, n - 1)
                j = random.randint(i + 1, n - 1)
                k = random.randint(j + 1, n - 1)
                c_ij = colors[(i, j)]
                c_ik = colors[(i, k)]
                c_jk = colors[(j, k)]
                if c_ij == c_ik == c_jk:
                    # Flip one of the edges
                    edge_to_flip = random.choice([(i, j), (i, k), (j, k)])
                    u, v = edge_to_flip
                    old_c = colors[(u, v)]
                    new_c = random.choice([c for c in range(4) if c != old_c])
                    colors[(u, v)] = new_c
                    new_penalty = count_mono_triangles(n)
                    if new_penalty < current_penalty:
                        current_penalty = new_penalty
                        found = True
                    else:
                        colors[(u, v)] = old_c
                    break
            
            if not found:
                # If no mono triangle found in random sample, maybe it's 0?
                # But we know current_penalty > 0.
                # So we just accept a random flip to escape local minimum
                i = random.randint(0, n - 1)
                j = random.randint(i + 1, n - 1)
                old_c = colors[(i, j)]
                new_c = random.choice([c for c in range(4) if c != old_c])
                colors[(i, j)] = new_c
                new_penalty = count_mono_triangles(n)
                if new_penalty < current_penalty:
                    current_penalty = new_penalty
                else:
                    # Revert if worse, or keep if same?
                    # Let's keep if same to explore
                    pass
                    
        return current_penalty == 0

    # Try to extend
    while time.time() < deadline:
        next_n = n_current + 1
        if next_n > 62:
            break
            
        # Add new vertex
        for i in range(next_n):
            if i == next_n - 1: continue
            colors[(i, next_n - 1)] = random.randint(0, 3)
            
        # Local search
        success = local_search(next_n, max_iterations=500)
        
        if success:
            n_current = next_n
            best_n = n_current
            best_colors = {k: v for k, v in colors.items()}
        else:
            # Revert
            for i in range(next_n):
                if i == next_n - 1: continue
                colors.pop((i, next_n - 1), None)
            
            # If we fail to extend, we might try a different random seed for the new vertex
            # But to save time, we might just stop or try a few more times
            # Let's try 3 restarts for this n
            found = False
            for _ in range(3):
                for i in range(next_n):
                    if i == next_n - 1: continue
                    colors[(i, next_n - 1)] = random.randint(0, 3)
                if local_search(next_n, max_iterations=500):
                    n_current = next_n
                    best_n = n_current
                    best_colors = {k: v for k, v in colors.items()}
                    found = True
                    break
                else:
                    for i in range(next_n):
                        if i == next_n - 1: continue
                        colors.pop((i, next_n - 1), None)
            
            if not found:
                # Try to improve current best_n with more search?
                # Or break if we can't extend
                # Let's try to extend again with different init
                # If we fail 3 times, we assume stuck for this n
                # But maybe we can jump to n+2? No, must be contiguous.
                # So we break the extension loop
                break

    # Format output
    final_colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            final_colors.append(best_colors[(i, j)])
            
    return {'n': best_n, 'colors': final_colors}
