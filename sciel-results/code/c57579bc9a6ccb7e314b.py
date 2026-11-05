import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_seed_coloring():
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

    best_result = get_seed_coloring()

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

    def greedy_init(n, seed=None):
        """Greedy initialization that tries to avoid triangles."""
        if seed is not None:
            random.seed(seed)
        
        adj = [[0] * n for _ in range(4)]
        color = [[-1] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                # Try each color, pick the one that creates fewest new triangles
                best_k = -1
                best_delta = float('inf')
                for k in range(4):
                    # Count triangles that would be created by adding edge (i,j) with color k
                    common = adj[k][i] & adj[k][j]
                    delta = common.bit_count()
                    if delta < best_delta or (delta == best_delta and random.random() < 0.5):
                        best_delta = delta
                        best_k = k
                
                color[i][j] = best_k
                color[j][i] = best_k
                adj[best_k][i] |= (1 << j)
                adj[best_k][j] |= (1 << i)
        
        return adj, color, count_mono_triangles(adj, n)

    def random_init(n, seed=None):
        """Random initialization."""
        if seed is not None:
            random.seed(seed)
        
        adj = [[0] * n for _ in range(4)]
        color = [[-1] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                k = random.randint(0, 3)
                color[i][j] = k
                color[j][i] = k
                adj[k][i] |= (1 << j)
                adj[k][j] |= (1 << i)
        
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
        best_adj = [row[:] for row in adj]
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
            common_old = adj[old_k][i] & adj[old_k][j]
            delta_remove = common_old.bit_count()

            common_new = adj[new_k][i] & adj[new_k][j]
            delta_add = common_new.bit_count()

            delta = delta_add - delta_remove

            # Accept or reject
            if delta <= 0 or random.random() < np.exp(-delta / T):
                # Apply change
                adj[old_k][i] ^= (1 << j)
                adj[old_k][j] ^= (1 << i)
                adj[new_k][i] |= (1 << j)
                adj[new_k][j] |= (1 << i)
                color[i][j] = new_k
                color[j][i] = new_k
                current_score += delta

                steps_since_improve += 1
                if current_score < best_score:
                    best_score = current_score
                    best_adj = [row[:] for row in adj]
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
                adj = [row[:] for row in best_adj]
                color = [row[:] for row in best_color]
                current_score = best_score
                T = T_init
                steps_since_improve = 0

        return best_adj, best_color, best_score

    def try_extend_from_32(time_limit):
        """Try to extend the 32-vertex solution to larger n."""
        seed_result = get_seed_coloring()
        n0 = 32
        colors0 = seed_result['colors']
        
        # Build adj and color for n0
        adj = [[0] * n0 for _ in range(4)]
        color = [[-1] * n0 for _ in range(n0)]
        idx = 0
        for i in range(n0):
            for j in range(i + 1, n0):
                k = colors0[idx]
                color[i][j] = k
                color[j][i] = k
                adj[k][i] |= (1 << j)
                adj[k][j] |= (1 << i)
                idx += 1
        
        # Try to add vertices one at a time
        for target_n in range(n0 + 1, 51):
            if time.time() > time_limit:
                break
            
            # Add new vertex
            n_new = target_n
            adj_new = [[0] * n_new for _ in range(4)]
            color_new = [[-1] * n_new for _ in range(n_new)]
            
            # Copy existing
            for i in range(n0):
                for j in range(i + 1, n0):
                    k = color[i][j]
                    color_new[i][j] = k
                    color_new[j][i] = k
                    adj_new[k][i] |= (1 << j)
                    adj_new[k][j] |= (1 << i)
            
            # Assign colors for edges from new vertex
            new_v = n0
            for i in range(n0):
                # Try each color, pick one that creates fewest triangles
                best_k = -1
                best_delta = float('inf')
                for k in range(4):
                    # Count triangles with new_v, i, and some w
                    # Triangle (new_v, i, w) exists if color(new_v,w) == color(i,w) == color(new_v,i)
                    # For fixed color k of (new_v, i), count w where color(i,w) == k and color(new_v,w) == k
                    # But we haven't assigned color(new_v,w) yet for w > i
                    # So just count common neighbors of new_v and i with color k among existing vertices
                    # Actually, new_v has no edges yet except to i, so no triangles can be formed
                    # with new_v as one vertex unless we've already assigned some edges
                    # Let's just assign randomly for now and let SA fix it
                    pass
                
                # For now, assign randomly
                k = random.randint(0, 3)
                color_new[new_v][i] = k
                color_new[i][new_v] = k
                adj_new[k][new_v] |= (1 << i)
                adj_new[k][i] |= (1 << new_v)
            
            # Now run SA to fix any triangles
            adj_s, color_s, score = run_sa(adj_new, color_new, n_new, time.time() + 2.0, 
                                           T_init=1.0, T_min=0.05, cooling_rate=0.9998,
                                           max_steps_no_improve=10000, seed=target_n * 7)
            
            if score == 0:
                colors_out = []
                for i in range(n_new):
                    for j in range(i + 1, n_new):
                        colors_out.append(color_s[i][j])
                return {'n': n_new, 'colors': colors_out}
            
            # If not zero, continue trying
            adj = adj_s
            color = color_s
            n0 = n_new
        
        return None

    # Main search strategy
    # 1. Try extending from 32-vertex solution
    if time.time() < deadline - 10:
        ext_result = try_extend_from_32(time.time() + 5.0)
        if ext_result and ext_result['n'] >= 50:
            return ext_result
    
    # 2. Multiple SA runs with different initializations
    n_target = 50
    num_restarts = 15
    
    for attempt in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        time_left = deadline - time.time()
        if time_left < 3:
            break
        
        # Alternate between greedy and random initialization
        if attempt % 2 == 0:
            adj, color, score = greedy_init(n_target, seed=attempt * 1000 + 42)
        else:
            adj, color, score = random_init(n_target, seed=attempt * 1000 + 43)
        
        if score == 0:
            colors = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    colors.append(color[i][j])
            return {'n': n_target, 'colors': colors}
        
        alloc = min(4.0, time_left - 2.0)
        adj_s, color_s, score_s = run_sa(adj, color, n_target, time.time() + alloc,
                                         T_init=1.5, T_min=0.05, cooling_rate=0.9997,
                                         max_steps_no_improve=15000, seed=attempt * 500 + 99)
        
        if score_s == 0:
            colors = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    colors.append(color_s[i][j])
            return {'n': n_target, 'colors': colors}
    
    # 3. Try smaller n if 50 fails
    for n in range(49, 32, -1):
        if time.time() > deadline - 3:
            break
        
        time_left = deadline - time.time()
        if time_left < 2:
            break
        
        alloc = min(2.5, time_left - 1.0)
        
        adj, color, score = greedy_init(n, seed=n * 100 + 7)
        if score == 0:
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(color[i][j])
            return {'n': n, 'colors': colors}
        
        adj_s, color_s, score_s = run_sa(adj, color, n, time.time() + alloc,
                                         T_init=1.0, T_min=0.05, cooling_rate=0.9998,
                                         max_steps_no_improve=10000, seed=n * 200 + 13)
        
        if score_s == 0:
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(color_s[i][j])
            return {'n': n, 'colors': colors}
    
    return best_result
