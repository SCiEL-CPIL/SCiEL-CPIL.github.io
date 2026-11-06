import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def get_seed():
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
        return 32, colors

    best_n, best_colors = get_seed()

    def solve_local_search(n, time_limit):
        # Initialize random coloring
        colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        
        # Precompute offsets
        offsets = [0] * n
        cnt = 0
        for i in range(n):
            offsets[i] = cnt
            cnt += n - 1 - i

        def get_color(u, v):
            if u > v: u, v = v, u
            return colors[offsets[u] + (v - u - 1)]

        def set_color(u, v, c):
            if u > v: u, v = v, u
            colors[offsets[u] + (v - u - 1)] = c

        # Build adjacency bitmasks
        adj = [[0] * n for _ in range(4)]
        
        def rebuild_adj():
            for c in range(4):
                for u in range(n):
                    adj[c][u] = 0
            for u in range(n):
                for v in range(u + 1, n):
                    c = get_color(u, v)
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)

        def count_tris():
            total = 0
            for u in range(n):
                for v in range(u + 1, n):
                    c = get_color(u, v)
                    common = adj[c][u] & adj[c][v]
                    total += bin(common).count('1')
            return total // 3

        rebuild_adj()
        current_tris = count_tris()
        best_tris = current_tris
        best_colors_copy = colors[:]

        while time.time() < time_limit:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v: continue
            
            old_c = get_color(u, v)
            new_c = (old_c + 1) % 4
            
            broken = bin(adj[old_c][u] & adj[old_c][v]).count('1')
            
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            
            added = bin(adj[new_c][u] & adj[new_c][v]).count('1')
            
            delta = added - broken
            new_tris = current_tris + delta
            
            if new_tris < current_tris:
                current_tris = new_tris
                set_color(u, v, new_c)
                if current_tris < best_tris:
                    best_tris = current_tris
                    best_colors_copy = colors[:]
                    if current_tris == 0:
                        return True, best_colors_copy
            else:
                adj[old_c][u] |= (1 << v)
                adj[old_c][v] |= (1 << u)
                adj[new_c][u] &= ~(1 << v)
                adj[new_c][v] &= ~(1 << u)

        return best_tris == 0, best_colors_copy

    # Try n=50
    time_left = deadline - time.time()
    if time_left > 20:
        found, sol = solve_local_search(50, time.time() + time_left - 5)
        if found:
            return {'n': 50, 'colors': sol}
            
    # Try n=40 if time permits?
    time_left = deadline - time.time()
    if time_left > 20:
        found, sol = solve_local_search(40, time.time() + time_left - 5)
        if found:
            return {'n': 40, 'colors': sol}

    return {'n': best_n, 'colors': best_colors}
