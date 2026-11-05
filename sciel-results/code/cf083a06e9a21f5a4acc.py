import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 118.0  # Safety margin (150s kill, keep 32s buffer)

    # --- Helper: GF(16) multiplication for the Clebsch graph ---
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

    def build_seed_32():
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors

    def count_mono_triangles(adj, n):
        """Count total monochromatic triangles using bitmasks."""
        total = 0
        for k in range(4):
            for i in range(n):
                m = adj[k][i]
                while m:
                    lb = m & (-m)
                    j = lb.bit_length() - 1
                    common = adj[k][i] & adj[k][j]
                    total += common.bit_count()
                    m ^= lb
        return total // 6

    def greedy_init(n, seed_colors=None, seed=None):
        """
        Greedy initialization that tries to avoid triangles.
        If seed_colors provided, use them for edges among first m vertices.
        """
        if seed is not None:
            random.seed(seed)
        
        adj = [0] * (4 * n)  # flat: adj[k*n + i] = bitmask of neighbors of i in color k
        color = [[-1] * n for _ in range(n)]
        
        # If we have seed colors, load them
        m = 0
        if seed_colors is not None:
            m = int((2 * len(seed_colors)) ** 0.5 + 0.5)
            # Actually, len(seed_colors) = m*(m-1)//2, so m = (1 + sqrt(1+8*len))/2
            L = len(seed_colors)
            m = int((1 + (1 + 8 * L) ** 0.5) / 2)
            idx = 0
            for i in range(m):
                for j in range(i + 1, m):
                    k = seed_colors[idx]
                    color[i][j] = k
                    color[j][i] = k
                    adj[k * n + i] |= (1 << j)
                    adj[k * n + j] |= (1 << i)
                    idx += 1
        
        # Fill in remaining edges greedily
        for i in range(n):
            for j in range(i + 1, n):
                if color[i][j] != -1:
                    continue
                # Try each color, pick the one that creates fewest new triangles
                best_k = -1
                best_delta = float('inf')
                for k in range(4):
                    # Count triangles that would be created by adding edge (i,j) with color k
                    common = adj[k * n + i] & adj[k * n + j]
                    delta = common.bit_count()
                    if delta < best_delta or (delta == best_delta and random.random() < 0.5):
                        best_delta = delta
                        best_k = k
                
                color[i][j] = best_k
                color[j][i] = best_k
                adj[best_k * n + i] |= (1 << j)
                adj[best_k * n + j] |= (1 << i)
        
        return adj, color, count_mono_triangles(adj, n)

    def random_init(n, seed=None):
        """Random initialization."""
        if seed is not None:
            random.seed(seed)
        
        adj = [0] * (4 * n)
        color = [[-1] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                k = random.randint(0, 3)
                color[i][j] = k
                color[j][i] = k
                adj[k * n + i] |= (1 << j)
                adj[k * n + j] |= (1 << i)
        
        return adj, color, count_mono_triangles(adj, n)

    def run_sa(adj, color, n, time_limit, T_init=1.5, T_min=0.05, cooling_rate=0.9997, 
               max_steps_no_improve=15000, seed=None):
        """Simulated annealing to minimize monochromatic triangles."""
        if time.time() > time_limit:
            return None, None, float('inf')

        if seed is not None:
            random.seed(seed)

        current_score = count_mono_triangles(adj, n)
        if current_score == 0:
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(color[i][j])
            return adj, color, 0

        best_score = current_score
        best_adj = adj[:]
        best_color = [row[:] for row in color]

        T = T_init
        steps_since_improve = 0

        while time.time() < time_limit:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i

            old_k = color[i][j]

            # Pick new color
            new_k = random.randint(0, 3)
            if new_k == old_k:
                continue

            # Calculate delta in mono triangles
            common_old = adj[old_k * n + i] & adj[old_k * n + j]
            delta_remove = common_old.bit_count()

            common_new = adj[new_k * n + i] & adj[new_k * n + j]
            delta_add = common_new.bit_count()

            delta = delta_add - delta_remove

            # Accept or reject
            if delta <= 0 or random.random() < np.exp(-delta / T):
                # Apply change
                adj[old_k * n + i] ^= (1 << j)
                adj[old_k * n + j] ^= (1 << i)
                adj[new_k * n + i] |= (1 << j)
                adj[new_k * n + j] |= (1 << i)
                color[i][j] = new_k
                color[j][i] = new_k
                current_score += delta

                steps_since_improve += 1
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj[:]
                    best_color = [row[:] for row in color]
                    steps_since_improve = 0
                    if best_score == 0:
                        return best_adj, best_color, 0
            else:
                pass

            T *= cooling_rate
            if T < T_min:
                T = T_min

            if steps_since_improve > max_steps_no_improve:
                # Restart from best
                adj = best_adj[:]
                color = [row[:] for row in best_color]
                current_score = best_score
                T = T_init
                steps_since_improve = 0

        return best_adj, best_color, best_score

    # Plan:
    # 1. Try N=50 with multiple restarts using seed-based greedy initialization
    # 2. Try N=48, 46 as fallbacks
    
    seed_32 = build_seed_32()
    
    # Strategy 1: Direct SA on N=50 with seed-based greedy init
    time_per_attempt = 18.0
    num_attempts = 5  # 5 * 18 = 90s
    
    for i in range(num_attempts):
        if time.time() > DEADLINE - 25:
            break
        
        # Create seed-based greedy initialization for N=50
        N = 50
        m = 32
        adj, color, score = greedy_init(N, seed_colors=seed_32, seed=i * 1000 + 42)
        
        res_adj, res_color, res_score = run_sa(adj, color, N, time_per_attempt, seed=i * 2000 + 7)
        
        if res_score == 0:
            res = []
            for a in range(N):
                for b in range(a + 1, N):
                    res.append(res_color[a][b])
            return {'n': N, 'colors': res}
    
    # Strategy 2: Try N=48
    for i in range(3):
        if time.time() > DEADLINE - 15:
            break
        N = 48
        m = 32
        adj, color, score = greedy_init(N, seed_colors=seed_32, seed=i * 1000 + 42)
        
        res_adj, res_color, res_score = run_sa(adj, color, N, 12.0, seed=i * 2000 + 7)
        
        if res_score == 0:
            res = []
            for a in range(N):
                for b in range(a + 1, N):
                    res.append(res_color[a][b])
            return {'n': N, 'colors': res}
    
    # Strategy 3: Try N=46
    for i in range(2):
        if time.time() > DEADLINE - 8:
            break
        N = 46
        m = 32
        adj, color, score = greedy_init(N, seed_colors=seed_32, seed=i * 1000 + 42)
        
        res_adj, res_color, res_score = run_sa(adj, color, N, 8.0, seed=i * 2000 + 7)
        
        if res_score == 0:
            res = []
            for a in range(N):
                for b in range(a + 1, N):
                    res.append(res_color[a][b])
            return {'n': N, 'colors': res}
    
    # Fallback to Seed 32
    return {'n': 32, 'colors': seed_32}
