import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def get_paley_colors(n):
        if n == 29:
            p = 29
            squares = {x*x % p for x in range(1, p)}
            colors = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i+1, n):
                    if (j-i) % p in squares:
                        colors[i, j] = 0
                        colors[j, i] = 0
                    else:
                        colors[i, j] = 1
                        colors[j, i] = 1
            return colors
        elif n == 17:
            p = 17
            squares = {x*x % p for x in range(1, p)}
            colors = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i+1, n):
                    if (j-i) % p in squares:
                        colors[i, j] = 0
                        colors[j, i] = 0
                    else:
                        colors[i, j] = 1
                        colors[j, i] = 1
            return colors
        return None

    def init_state(n, colors):
        adj = [None, None]
        adj[0] = [0] * n
        adj[1] = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                c = colors[i, j]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        return adj

    def is_bad_pair(u, v, c, adj, n):
        S = adj[c][u] & adj[c][v]
        if S.bit_count() < 4:
            return False
        bits = []
        s = S
        while s:
            lsb = s & -s
            idx = lsb.bit_length() - 1
            bits.append(idx)
            s ^= lsb
        L = len(bits)
        for i in range(L):
            x = bits[i]
            for j in range(i+1, L):
                y = bits[j]
                T = adj[c][x] & adj[c][y] & S
                if T.bit_count() >= 2:
                    return True
        return False

    def get_bad_count(n, adj):
        count = 0
        for c in [0, 1]:
            for u in range(n):
                for v in range(u+1, n):
                    if is_bad_pair(u, v, c, adj, n):
                        count += 1
        return count

    def local_search(n, init_colors, time_limit):
        colors = init_colors.copy()
        adj = init_state(n, colors)
        bad_count = get_bad_count(n, adj)
        
        if bad_count == 0:
            return colors
        
        best_colors = colors.copy()
        best_bad = bad_count
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if bad_count == 0:
                return colors
            
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            affected = set()
            for x in range(n):
                if x != i and x != j:
                    affected.add((min(i, x), max(i, x)))
                    affected.add((min(j, x), max(j, x)))
            affected.add((i, j))
            
            old_bad_sum = 0
            for u, v in affected:
                for c in [0, 1]:
                    if is_bad_pair(u, v, c, adj, n):
                        old_bad_sum += 1
                        
            old_c = colors[i, j]
            new_c = 1 - old_c
            
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            
            colors[i, j] = new_c
            colors[j, i] = new_c
            
            new_bad_sum = 0
            for u, v in affected:
                for c in [0, 1]:
                    if is_bad_pair(u, v, c, adj, n):
                        new_bad_sum += 1
                        
            delta = new_bad_sum - old_bad_sum
            bad_count += delta
            
            accept = False
            if delta < 0:
                accept = True
                best_colors = colors.copy()
                best_bad = bad_count
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                prob = np.exp(-delta / T)
                accept = random.random() < prob
                
            if not accept:
                # Revert
                adj[new_c][i] &= ~(1 << j)
                adj[new_c][j] &= ~(1 << i)
                adj[old_c][i] |= (1 << j)
                adj[old_c][j] |= (1 << i)
                colors[i, j] = old_c
                colors[j, i] = old_c
                bad_count -= delta
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        if get_bad_count(n, init_state(n, best_colors)) == 0:
            return best_colors
        return None

    # Try n=30
    if time.time() < deadline - 60:
        n = 30
        # Hybrid init: Paley 29 + 1 random
        paley29 = get_paley_colors(29)
        colors30 = np.zeros((30, 30), dtype=np.int8)
        colors30[:29, :29] = paley29
        for i in range(29):
            c = random.randint(0, 1)
            colors30[i, 29] = c
            colors30[29, i] = c
            
        result = local_search(n, colors30, 50)
        if result is not None:
            flat = [int(result[i, j]) for i in range(n) for j in range(i+1, n)]
            return {'n': n, 'colors': flat}

    # Try n=29
    if time.time() < deadline - 20:
        n = 29
        colors = get_paley_colors(n)
        result = local_search(n, colors, 20)
        if result is not None:
            flat = [int(result[i, j]) for i in range(n) for j in range(i+1, n)]
            return {'n': n, 'colors': flat}

    # Fallback n=17
    n = 17
    colors = get_paley_colors(n)
    flat = [int(colors[i, j]) for i in range(n) for j in range(i+1, n)]
    return {'n': n, 'colors': flat}
