import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def get_fallback():
        # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16)
        # doubled: two copies with colour 3 on all edges between them
        # This yields n=32.
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
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return {'n': 32, 'colors': colors}

    def count_mono_triangles(adj, n):
        total = 0
        for k in range(4):
            for i in range(n):
                m = adj[k][i]
                while m:
                    lb = m & (-m)
                    j = lb.bit_length() - 1
                    if j > i:
                        common = adj[k][i] & adj[k][j]
                        total += common.bit_count()
                    m ^= lb
        return total // 3

    def greedy_init(n, rng):
        adj = [[0] * n for _ in range(4)]
        color = [[-1] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                best_k = -1
                best_delta = float('inf')
                for k in range(4):
                    common = adj[k][i] & adj[k][j]
                    delta = common.bit_count()
                    if delta < best_delta:
                        best_delta = delta
                        best_k = k
                    elif delta == best_delta and rng.random() < 0.5:
                        best_k = k
                
                color[i][j] = best_k
                color[j][i] = best_k
                adj[best_k][i] |= (1 << j)
                adj[best_k][j] |= (1 << i)
        
        return adj, color, count_mono_triangles(adj, n)

    def run_sa(adj, color, n, time_limit, rng, T_init=1.0, T_min=0.01, cooling_rate=0.999):
        current_score = count_mono_triangles(adj, n)
        if current_score == 0:
            return adj, color, 0
        
        best_score = current_score
        best_adj = [row[:] for row in adj]
        best_color = [row[:] for row in color]
        
        T = T_init
        # Pre-compute list of edges for faster iteration? No, random is better for SA.
        # But we can prioritize edges with high triangle counts.
        
        while time.time() < time_limit:
            # Pick a random edge
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_k = color[i][j]
            # Pick a new color different from old
            new_k = rng.randint(0, 3)
            while new_k == old_k:
                new_k = rng.randint(0, 3)
            
            # Calculate delta
            # Remove old
            common_old = adj[old_k][i] & adj[old_k][j]
            delta_remove = common_old.bit_count()
            
            # Add new
            common_new = adj[new_k][i] & adj[new_k][j]
            delta_add = common_new.bit_count()
            
            delta = delta_add - delta_remove
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                # Accept
                adj[old_k][i] ^= (1 << j)
                adj[old_k][j] ^= (1 << i)
                adj[new_k][i] |= (1 << j)
                adj[new_k][j] |= (1 << i)
                color[i][j] = new_k
                color[j][i] = new_k
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_adj = [row[:] for row in adj]
                    best_color = [row[:] for row in color]
                    if best_score == 0:
                        return best_adj, best_color, 0
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        return best_adj, best_color, best_score

    def try_solve_n(n, time_limit, rng):
        # Strategy: Multiple restarts with Greedy Init + SA
        best_result = None
        best_score = float('inf')
        
        # Determine number of restarts based on time
        # Each restart takes roughly 1-5 seconds depending on n and convergence
        # Let's just loop until time is up
        
        while time.time() < time_limit - 1:
            adj, color, score = greedy_init(n, rng)
            
            if score == 0:
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        colors.append(color[i][j])
                return {'n': n, 'colors': colors}
            
            # Run SA for a short burst
            # Estimate time per SA run
            # If n is large, SA is slower.
            sa_time = min(2.0, (time_limit - time.time()) * 0.8)
            
            adj_sa, color_sa, score_sa = run_sa(adj, color, n, time.time() + sa_time, rng)
            
            if score_sa < best_score:
                best_score = score_sa
                best_adj = adj_sa
                best_color = color_sa
                
                if score_sa == 0:
                    colors = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            colors.append(color_sa[i][j])
                    return {'n': n, 'colors': colors}
        
        if best_score == 0:
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(best_color[i][j])
            return {'n': n, 'colors': colors}
            
        return None

    # Try to find solutions for n=50 down to 33
    # We prioritize larger n.
    # If we find 50, we return immediately.
    # If not, we try 49, etc.
    
    rng = random.Random(12345)
    
    # Allocate time:
    # Try 50 for 30s
    # Try 49 for 20s
    # Try 48 for 15s
    # ...
    # Fallback to 32
    
    time_budgets = {
        50: 30,
        49: 20,
        48: 15,
        47: 10,
        46: 5,
        45: 5,
        44: 5,
        43: 5,
        42: 5,
        41: 5,
        40: 5,
        39: 5,
        38: 5,
        37: 5,
        36: 5,
        35: 5,
        34: 5,
        33: 5
    }
    
    for n in sorted(time_budgets.keys(), reverse=True):
        if time.time() > deadline - 2:
            break
        
        allocated = time_budgets[n]
        # Adjust if we have less time remaining
        remaining = deadline - time.time() - 1
        if remaining < 2:
            break
        
        t_limit = min(allocated, remaining)
        
        res = try_solve_n(n, t_limit, rng)
        if res:
            return res
            
    return get_fallback()

if __name__ == "__main__":
    # For local testing
    result = construct()
    print(result['n'])
