import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    def get_fallback():
        # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16)
        # doubled: two copies with colour 3 on all edges between them
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
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}

    def build_seed_coloring(n):
        """Build a seed coloring using doubled GF(16) for n <= 32, or extend for larger n."""
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
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i < 32 and j < 32:
                    # Use GF(16) doubled coloring for first 32 vertices
                    if i // 16 == j // 16:
                        colors.append(log[(i % 16) ^ (j % 16)] % 3)
                    else:
                        colors.append(3)
                else:
                    # For vertices >= 32, assign random colors
                    colors.append(random.randint(0, 3))
        return colors

    def count_mono_triangles(adj, n):
        """Count total monochromatic triangles using bitsets."""
        total = 0
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[c][i] & (1 << j):
                        total += (adj[c][i] & adj[c][j]).bit_count()
        return total // 3

    def local_search(n, colors, adj, time_limit, max_restarts=5):
        """Local search with simulated annealing and restarts."""
        total_mono = count_mono_triangles(adj, n)
        if total_mono == 0:
            return colors[:]
        
        best_colors = colors[:]
        best_total = total_mono
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        for restart in range(max_restarts):
            # Reset temperature
            temp = 5.0 + random.random() * 5.0
            cooling = 0.9995
            current_total = total_mono
            
            # Optionally perturb if not first restart
            if restart > 0:
                # Perturb some random edges
                num_perturb = max(1, num_edges // 20)
                for _ in range(num_perturb):
                    idx = random.randint(0, num_edges - 1)
                    u, v = edges[idx]
                    old_c = colors[idx]
                    new_c = random.choice([c for c in range(4) if c != old_c])
                    colors[idx] = new_c
                    adj[old_c][u] ^= (1 << v)
                    adj[old_c][v] ^= (1 << u)
                    adj[new_c][u] ^= (1 << v)
                    adj[new_c][v] ^= (1 << u)
                    # Recalculate total
                    current_total = count_mono_triangles(adj, n)
            
            # Simulated annealing loop
            while time.time() < time_limit:
                idx = random.randint(0, num_edges - 1)
                u, v = edges[idx]
                c_old = colors[idx]
                
                # Find best new color
                best_delta = 0
                best_c_new = c_old
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    n_old = (adj[c_old][u] & adj[c_old][v]).bit_count()
                    n_new = (adj[c_new][u] & adj[c_new][v]).bit_count()
                    delta = n_new - n_old
                    if delta < best_delta:
                        best_delta = delta
                        best_c_new = c_new
                
                # Acceptance criterion
                accept = False
                if best_delta < 0:
                    accept = True
                elif temp > 0.01:
                    if random.random() < pow(2.71828, -best_delta / temp):
                        accept = True
                
                if accept:
                    colors[idx] = best_c_new
                    adj[c_old][u] ^= (1 << v)
                    adj[c_old][v] ^= (1 << u)
                    adj[best_c_new][u] ^= (1 << v)
                    adj[best_c_new][v] ^= (1 << u)
                    current_total += best_delta
                    temp *= cooling
                    
                    if current_total < best_total:
                        best_total = current_total
                        best_colors = colors[:]
                        if best_total == 0:
                            return best_colors
                else:
                    temp *= cooling
        
        return best_colors if best_total == 0 else None

    def solve_n(n, time_limit):
        """Try to solve for n vertices within time_limit."""
        # Build seed coloring
        colors = build_seed_coloring(n)
        
        # Build adjacency bitsets
        adj = [[0] * n for _ in range(4)]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for (u, v), c in zip(edges, colors):
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
        
        # Run local search
        result = local_search(n, colors, adj, start_time + time_limit)
        if result is not None:
            return {'n': n, 'colors': result}
        return None

    # Try to find solutions for n=50 down to 33
    # Allocate time wisely, giving more time to larger n
    time_per_n = 10.0
    for n in range(50, 32, -1):
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
        t_limit = min(time_per_n, remaining)
        res = solve_n(n, t_limit)
        if res:
            return res
            
    return get_fallback()
