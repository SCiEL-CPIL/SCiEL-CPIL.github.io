import random
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for safety
    
    # 1. Generate the seed n=32 coloring (Greenwood-Gleason / Clebsch based)
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
        
    # Build the n=32 adjacency matrix
    # Vertices 0..31
    # Color 3 for edges between the two halves (0-15, 16-31)
    # Colors 0,1,2 for edges within halves based on GF(16) cubic residues
    # Actually, the seed code in the prompt does:
    # log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
    # This creates a 32-vertex graph.
    
    max_n = 50
    color = [[0] * max_n for _ in range(max_n)]
    
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                c = log[(i % 16) ^ (j % 16)] % 3
            else:
                c = 3
            color[i][j] = c
            color[j][i] = c
            
    n = 32
    best_n = 32
    best_colors = None
    
    # Helper to extract colors list for current n
    def get_colors():
        res = []
        for i in range(n):
            for j in range(i + 1, n):
                res.append(color[i][j])
        return res
    
    best_colors = get_colors()
    
    # Local search to add a vertex
    def try_add_vertex(current_n, seed):
        random.seed(seed)
        
        # Precompute counts for fast delta calculation?
        # We need counts of c[w] for each u and color.
        # But c changes.
        
        # Initialize random colors for edges (current_n, 0..current_n-1)
        c = [random.randint(0, 3) for _ in range(current_n)]
        
        def count_triangles():
            t = 0
            for u in range(current_n):
                cu = c[u]
                for w in range(u + 1, current_n):
                    if cu == c[w] and cu == color[u][w]:
                        t += 1
            return t
            
        t = count_triangles()
        
        # Hill Climbing
        for step in range(2000):
            if t == 0:
                return c
                
            # Pick random u
            u = random.randint(0, current_n - 1)
            old_c = c[u]
            
            # Find best new color
            best_new_c = old_c
            best_delta = 0
            
            for new_c in range(4):
                if new_c == old_c:
                    continue
                    
                # Calculate delta
                # Triangles removed: w such that color[u][w] == old_c and c[w] == old_c
                # Triangles added: w such that color[u][w] == new_c and c[w] == new_c
                
                # Note: w can be any vertex 0..current_n-1 except u.
                # In the triangle (current_n, u, w), the condition is:
                # c[u] == c[w] == color[u][w].
                
                rem = 0
                add = 0
                for w in range(current_n):
                    if w == u:
                        continue
                    if color[u][w] == old_c and c[w] == old_c:
                        rem += 1
                    if color[u][w] == new_c and c[w] == new_c:
                        add += 1
                        
                delta = add - rem
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
                    
            if best_new_c == old_c:
                # No improvement found for this u. 
                # Try a random kick to escape local minimum?
                # Or just break and restart?
                # Let's try a random flip if stuck for a while.
                # For now, just break out of inner loop and try another u?
                # Actually, if no single flip improves, we are at a local minimum.
                # We can try a random flip with small probability (Simulated Annealing style)
                # Or just restart the local search.
                # Given the small size, restarting is cheap.
                break
                
            c[u] = best_new_c
            t += best_delta
            
        if t == 0:
            return c
        return None

    # Main loop: try to extend to 50
    # We will try multiple seeds for the local search at each step
    # If we get stuck, we might try to restart the whole construction?
    # No, the base 32 is fixed. We just need to find a path.
    # If we get stuck at N, we return N.
    
    attempts = 0
    while n < max_n and time.time() - start_time < time_limit:
        success = False
        # Try a few random seeds for the local search
        for s in range(50):
            new_c = try_add_vertex(n, attempts * 100 + s)
            if new_c is not None:
                # Commit
                for u in range(n):
                    color[n][u] = new_c[u]
                    color[u][n] = new_c[u]
                n += 1
                best_n = n
                best_colors = get_colors()
                success = True
                break
                
        if not success:
            # If we failed to add a vertex, we might be stuck.
            # We can try to increase the time limit for local search?
            # Or just break.
            # To maximize n, we should probably break here as we can't add more.
            # However, maybe a different random seed for the *initial* state of the local search
            # would have worked? We already tried 50.
            # It's unlikely we can jump over this barrier without changing the existing graph.
            break

    return {'n': best_n, 'colors': best_colors}
