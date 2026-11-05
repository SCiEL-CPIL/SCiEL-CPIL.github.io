import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n = 26
    num_colors = 5
    
    # colors[i][j] for i < j
    # Using 2D list for speed
    def random_init():
        col = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                c = random.randint(0, num_colors-1)
                col[i][j] = c
                col[j][i] = c
        return col

    def build_adj(col):
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                c = col[i][j]
                adj[i].add(j)
                adj[j].add(i)
        return adj

    def count_penalty(col, adj):
        """
        Counts total number of monochromatic C4s.
        A C4 is identified by a pair of vertices (u, v) and 2 common neighbors in the same color.
        We sum C(k, 2) for all pairs (u, v) and colors c where k = |N_c(u) ∩ N_c(v)|.
        Note: If u and v are adjacent in color c, it still counts as a C4 structure?
        Yes, the cycle u-a-v-b-u exists in the graph if edges exist.
        """
        total = 0
        for c in range(num_colors):
            # Build adjacency for color c
            # adj_c[u] is set of v such that col[u][v] == c
            adj_c = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if col[i][j] == c:
                        adj_c[i].add(j)
                        adj_c[j].add(i)
            
            for u in range(n):
                for v in range(u+1, n):
                    common = adj_c[u] & adj_c[v]
                    k = len(common)
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    def try_solve():
        col = random_init()
        adj = build_adj(col)
        current_penalty = count_penalty(col, adj)
        
        if current_penalty == 0:
            return col

        best_local_penalty = current_penalty
        best_local_col = [row[:] for row in col]
        
        # Fast update logic:
        # When changing color of (i, j) from c_old to c_new:
        # 1. Update adj[c_old] and adj[c_new]
        # 2. Only pairs involving i or j can change their C4 contribution.
        #    Pairs (i, k) and (j, k) for all k != i, j.
        
        def get_penalty_delta(i, j, c_old, c_new, col, adj):
            # This is complex to implement correctly with just sets without rebuilding.
            # Given n=26, rebuilding adj for the affected colors and recounting 
            # only the affected pairs is fast enough?
            # Let's try a simpler approach: 
            # Since n is small, we can just recalculate the penalty for the whole graph 
            # but optimize the C4 counting using bitmasks or lists.
            # Or, even simpler: 
            # Just run the full count_penalty but make it faster using numpy?
            # No, let's stick to the "only affected pairs" logic but implement it carefully.
            
            pass # Placeholder, will implement inline in the loop for speed

        # Simpler robust approach for n=26:
        # Full recalculation of penalty is O(5 * n^2 * n) = O(5 * 26^3) ~ 880,000 ops.
        # If we do 1000 iterations, that's 880M ops. Might be too slow in Python.
        # Let's optimize count_penalty.
        
        # Optimized count_penalty:
        def fast_count_penalty(col):
            total = 0
            for c in range(num_colors):
                # adj_c[u] as bitmask or list?
                # n=26 fits in int. Use bitmasks.
                adj_mask = [0] * n
                for i in range(n):
                    for j in range(i+1, n):
                        if col[i][j] == c:
                            adj_mask[i] |= (1 << j)
                            adj_mask[j] |= (1 << i)
                
                for u in range(n):
                    for v in range(u+1, n):
                        # common neighbors
                        common = adj_mask[u] & adj_mask[v]
                        k = bin(common).count('1')
                        if k >= 2:
                            total += k * (k - 1) // 2
            return total

        current_penalty = fast_count_penalty(col)
        
        no_improve = 0
        temp = 1.0
        
        while time.time() < deadline:
            if current_penalty == 0:
                return col
            
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j: continue
            if i > j: i, j = j, i
            
            old_c = col[i][j]
            new_c = random.randint(0, num_colors-1)
            if new_c == old_c: continue
            
            col[i][j] = new_c
            col[j][i] = new_c
            
            new_penalty = fast_count_penalty(col)
            
            delta = new_penalty - current_penalty
            if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / temp)):
                current_penalty = new_penalty
                no_improve = 0
                if current_penalty < best_local_penalty:
                    best_local_penalty = current_penalty
                    best_local_col = [row[:] for row in col]
            else:
                col[i][j] = old_c
                col[j][i] = old_c
                no_improve += 1
            
            if no_improve > 500:
                break
            temp *= 0.995
            
        return best_local_col

    best_sol = None
    while time.time() < deadline:
        sol = try_solve()
        if sol is not None:
            # Verify
            if fast_count_penalty(sol) == 0:
                best_sol = sol
                break
    
    if best_sol is not None:
        res_colors = []
        for i in range(n):
            for j in range(i+1, n):
                res_colors.append(best_sol[i][j])
        return {'n': n, 'colors': res_colors}
    
    # Fallback to n=11
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}
