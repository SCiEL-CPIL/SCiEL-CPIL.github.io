import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds to be safe
    
    def count_c4s_in_edge(u, v, c, adj):
        # Count C4s containing edge (u,v) in color c
        # A = neighbors of v in color c
        # B = neighbors of u in color c
        A = adj[c][v]
        B = adj[c][u]
        count = 0
        # Iterate through bits in A
        w = A
        while w:
            # Get lowest set bit
            lsb = w & -w
            w ^= lsb
            wi = lsb.bit_length() - 1
            # Count x in B such that (wi, x) is color c
            # i.e., x is in adj[c][wi]
            count += (B & adj[c][wi]).bit_count()
        return count

    def solve_n(n):
        if n < 2:
            return None
        
        # adj[c][u] is a bitmask of neighbors of u in color c
        # c in 0..4, u in 0..n-1
        adj = [[0] * n for _ in range(5)]
        colors = [0] * (n * (n - 1) // 2)
        
        def get_idx(i, j):
            if i > j: i, j = j, i
            # Lexicographic order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
            # Index for (i, j) with i < j:
            # Number of pairs (i', j') with i' < i is i * n - i * (i-1) / 2?
            # Let's just use a mapping or calculate.
            # Pairs with first index < i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
            # Pairs with first index == i and second index < j: j - i - 1
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

        def set_color(i, j, c):
            idx = get_idx(i, j)
            colors[idx] = c
            # Update adj
            # Clear old color? We assume we call this after clearing or we manage it carefully.
            # Better to have a function that changes color.
            pass

        def change_color(i, j, c_new):
            idx = get_idx(i, j)
            c_old = colors[idx]
            if c_old == c_new:
                return 0
            
            # Calculate delta
            # Remove C4s for c_old
            count_old = count_c4s_in_edge(i, j, c_old, adj)
            # Add C4s for c_new
            count_new = count_c4s_in_edge(i, j, c_new, adj)
            
            delta = count_new - count_old
            
            # Update state
            colors[idx] = c_new
            # Update adj masks
            # Remove i from adj[c_old][j] and j from adj[c_old][i]
            adj[c_old][j] &= ~(1 << i)
            adj[c_old][i] &= ~(1 << j)
            # Add i to adj[c_new][j] and j to adj[c_new][i]
            adj[c_new][j] |= (1 << i)
            adj[c_new][i] |= (1 << j)
            
            return delta

        # Initialize
        total_c4 = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 4)
                idx = get_idx(i, j)
                colors[idx] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        # Calculate initial total_c4
        # Sum over all edges (i,j) and colors c?
        # No, each C4 is counted 4 times (once for each edge)?
        # A C4 has 4 edges.
        # My count_c4s_in_edge counts C4s containing that specific edge.
        # So summing over all edges (i,j) the count for color colors[i,j] will count each C4 exactly 4 times.
        # So total_c4 = sum / 4.
        
        sum_counts = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j)]
                sum_counts += count_c4s_in_edge(i, j, c, adj)
        total_c4 = sum_counts // 4
        
        if total_c4 == 0:
            return colors[:]
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.995
        min_T = 0.01
        steps = 0
        max_steps_per_restart = 20000
        
        while T > min_T:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Pick new color
            c_new = random.randint(0, 4)
            
            delta = change_color(i, j, c_new)
            total_c4 += delta
            
            # Acceptance criterion
            if delta < 0 or random.random() < np.exp(-delta / T):
                pass
            else:
                # Revert
                c_old = c_new # Wait, change_color already updated. We need to revert.
                # It's easier to just track the old color before change.
                # Let's refactor change_color to return old color or just revert manually.
                # Actually, change_color updated adj and colors.
                # To revert, we need to know c_old.
                # Let's modify change_color to return (delta, c_old)
                pass
            
            # The revert logic is tricky with the current change_color implementation.
            # Let's rewrite the loop to be cleaner.
            
            break # Placeholder for logic fix
            
        # The above loop is flawed. Let's implement a proper SA loop.
        
        return None # Placeholder

    # This is getting complicated to write correctly in one go without testing.
    # Let's try a simpler approach: Backtracking with constraint propagation?
    # Or just use a known solution if I can find one online or derive it.
    # Since I can't browse, I will implement a robust SA.

    def solve_n_sa(n, time_limit):
        end_time = time.time() + time_limit
        if n < 2:
            return None
        
        adj = [[0] * n for _ in range(5)]
        colors = [0] * (n * (n - 1) // 2)
        
        def get_idx(i, j):
            if i > j: i, j = j, i
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

        def count_c4s_in_edge(u, v, c):
            A = adj[c][v]
            B = adj[c][u]
            count = 0
            w = A
            while w:
                lsb = w & -w
                w ^= lsb
                wi = lsb.bit_length() - 1
                count += (B & adj[c][wi]).bit_count()
            return count

        # Init
        total_c4 = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 4)
                idx = get_idx(i, j)
                colors[idx] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        sum_counts = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j)]
                sum_counts += count_c4s_in_edge(i, j, c)
        total_c4 = sum_counts // 4
        
        if total_c4 == 0:
            return colors[:]

        # SA
        T = 5.0
        cooling = 0.99
        min_T = 0.1
        best_total_c4 = total_c4
        best_colors = colors[:]
        
        while T > min_T and time.time() < end_time:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            c_old = colors[get_idx(i, j)]
            c_new = random.randint(0, 4)
            if c_new == c_old:
                continue
            
            # Calculate delta
            count_old = count_c4s_in_edge(i, j, c_old)
            count_new = count_c4s_in_edge(i, j, c_new)
            delta = count_new - count_old
            
            # Apply change
            idx = get_idx(i, j)
            colors[idx] = c_new
            adj[c_old][i] &= ~(1 << j)
            adj[c_old][j] &= ~(1 << i)
            adj[c_new][i] |= (1 << j)
            adj[c_new][j] |= (1 << i)
            
            total_c4 += delta
            
            if total_c4 < best_total_c4:
                best_total_c4 = total_c4
                best_colors = colors[:]
                if best_total_c4 == 0:
                    return best_colors[:]
            
            # Accept or reject
            if delta < 0 or random.random() < np.exp(-delta / T):
                pass
            else:
                # Revert
                colors[idx] = c_old
                adj[c_new][i] &= ~(1 << j)
                adj[c_new][j] &= ~(1 << i)
                adj[c_old][i] |= (1 << j)
                adj[c_old][j] |= (1 << i)
                total_c4 -= delta
            
            T *= cooling
        
        return best_colors if best_total_c4 == 0 else None

    # Try n=26 first
    result = solve_n_sa(26, 80)
    if result is not None:
        return {'n': 26, 'colors': result}
    
    # If failed, try n=25
    result = solve_n_sa(25, 30)
    if result is not None:
        return {'n': 25, 'colors': result}
        
    # Fallback to seed
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
