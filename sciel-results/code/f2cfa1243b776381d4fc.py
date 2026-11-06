import time
import random

def construct():
    deadline = time.time() + 115
    
    def count_mono_triangles(colors, n):
        # colors is a list of length n*(n-1)//2
        # We need a way to access color(i, j) quickly
        # Let's use a 2D list or a dictionary
        # For speed, a 2D list of size n x n is better
        # But we store in lexicographic order for output.
        # Let's convert to a 2D list for internal processing
        adj = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                for k in range(j + 1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        count += 1
        return count

    def local_search(n, time_limit, seed):
        rng = random.Random(seed)
        
        # Initialize random colors
        num_edges = n * (n - 1) // 2
        colors = [rng.randint(0, 3) for _ in range(num_edges)]
        
        # Convert to adjacency matrix for fast access
        adj = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        # Precompute current cost
        cost = count_mono_triangles(colors, n)
        
        start_time = time.time()
        T = 1.0 # Temperature for SA
        T_min = 0.01
        
        while time.time() < start_time + time_limit:
            # Pick random edge
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = adj[i][j]
            
            # Try other colors
            new_c = rng.randint(0, 3)
            while new_c == old_c:
                new_c = rng.randint(0, 3)
            
            # Calculate delta cost
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                c_ik = adj[i][k]
                c_jk = adj[j][k]
                if c_ik == c_jk:
                    if c_ik == old_c:
                        delta -= 1
                    if c_ik == new_c:
                        delta += 1
            
            # Acceptance criterion
            if delta <= 0:
                # Accept
                adj[i][j] = new_c
                adj[j][i] = new_c
                # Update colors list
                # Find index of (i, j)
                # idx = i*(n-1) - i*(i+1)//2 + j - i - 1
                idx = i * (n - 1) - i * (i + 1) // 2 + j - i - 1
                colors[idx] = new_c
                cost += delta
            else:
                # Metropolis criterion
                if rng.random() < pow(2.71828, -delta / T):
                    adj[i][j] = new_c
                    adj[j][i] = new_c
                    idx = i * (n - 1) - i * (i + 1) // 2 + j - i - 1
                    colors[idx] = new_c
                    cost += delta
            
            # Cool down
            T *= 0.9999
            
            if cost == 0:
                return colors
        
        return None

    best_n = 0
    best_colors = []
    
    # Try to find a valid coloring for n=50, 49, ...
    for n in range(50, 31, -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline - 2:
                break
            
            time_left = min(20, deadline - time.time() - 2)
            if time_left < 1:
                break
            
            result = local_search(n, time_left, seed)
            if result is not None:
                # Verify
                if count_mono_triangles(result, n) == 0:
                    return {'n': n, 'colors': result}
    
    # If no valid coloring found, return the best partial result?
    # The task requires a valid coloring.
    # If we can't find a valid one, we should return a valid one for a smaller n.
    # We know n=32 is valid.
    # Let's ensure we return a valid n=32 if we fail.
    
    # Re-implement the K32 construction as fallback
    def _mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)
        
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
                
    return {'n': 32, 'colors': colors}
