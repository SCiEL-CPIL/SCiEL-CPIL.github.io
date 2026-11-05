import time
import random
import math

def construct():
    start_time = time.time()
    # 120s limit, use 110s for safety
    deadline = start_time + 110
    
    def get_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def init_state(colors, n):
        """Builds adjacency sets and common neighbor counts."""
        m = n * (n - 1) // 2
        adj = [set() for _ in range(5)]
        # adj[c] is a list of sets? No, adj[c][i] is set of neighbors.
        adj = [[set() for _ in range(n)] for _ in range(5)]
        
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
        
        # common[c][i][j] = number of common neighbors of i and j in color c
        common = [[[0] * n for _ in range(n)] for _ in range(5)]
        total_violations = 0
        
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    # Count common neighbors
                    # adj[c][u] & adj[c][v]
                    cnt = len(adj[c][u].intersection(adj[c][v]))
                    common[c][u][v] = cnt
                    common[c][v][u] = cnt
                    if cnt >= 2:
                        total_violations += cnt * (cnt - 1) // 2
                        
        return adj, common, total_violations

    def solve_sa(n, max_time, seed):
        random.seed(seed)
        m = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(m)]
        
        adj, common, current_bad = init_state(colors, n)
        
        if current_bad == 0:
            return colors, True
        
        best_colors = list(colors)
        best_bad = current_bad
        
        T = 1.0
        cooling = 0.999
        time_limit = time.time() + max_time
        
        while time.time() < time_limit:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            idx = get_idx(i, j, n)
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            delta = 0
            
            # Update common neighbor counts and delta
            # For each k != i, j
            for k in range(n):
                if k == i or k == j:
                    continue
                
                # Effect on color old_c: edge (i,j) is removed
                # 1. Pair (i, k): j was a common neighbor if j in adj[old_c][k]
                if j in adj[old_c][k]:
                    old_cnt = common[old_c][i][k]
                    new_cnt = old_cnt - 1
                    delta -= (old_cnt * (old_cnt - 1) // 2) - (new_cnt * (new_cnt - 1) // 2)
                    common[old_c][i][k] = new_cnt
                    common[old_c][k][i] = new_cnt
                
                # 2. Pair (j, k): i was a common neighbor if i in adj[old_c][k]
                if i in adj[old_c][k]:
                    old_cnt = common[old_c][j][k]
                    new_cnt = old_cnt - 1
                    delta -= (old_cnt * (old_cnt - 1) // 2) - (new_cnt * (new_cnt - 1) // 2)
                    common[old_c][j][k] = new_cnt
                    common[old_c][k][j] = new_cnt
                
                # Effect on color new_c: edge (i,j) is added
                # 1. Pair (i, k): j is a common neighbor if j in adj[new_c][k]
                if j in adj[new_c][k]:
                    old_cnt = common[new_c][i][k]
                    new_cnt = old_cnt + 1
                    delta += (new_cnt * (new_cnt - 1) // 2) - (old_cnt * (old_cnt - 1) // 2)
                    common[new_c][i][k] = new_cnt
                    common[new_c][k][i] = new_cnt
                
                # 2. Pair (j, k): i is a common neighbor if i in adj[new_c][k]
                if i in adj[new_c][k]:
                    old_cnt = common[new_c][j][k]
                    new_cnt = old_cnt + 1
                    delta += (new_cnt * (new_cnt - 1) // 2) - (old_cnt * (old_cnt - 1) // 2)
                    common[new_c][j][k] = new_cnt
                    common[new_c][k][j] = new_cnt
            
            new_bad = current_bad + delta
            
            # Acceptance criterion
            if new_bad < current_bad or (delta >= 0 and random.random() < math.exp(-delta / T)):
                current_bad = new_bad
                colors[idx] = new_c
                
                # Update adjacency sets
                adj[old_c][i].remove(j)
                adj[old_c][j].remove(i)
                adj[new_c][i].add(j)
                adj[new_c][j].add(i)
                
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = list(colors)
                    if best_bad == 0:
                        return best_colors, True
            else:
                # Revert changes
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if j in adj[old_c][k]:
                        common[old_c][i][k] += 1
                        common[old_c][k][i] += 1
                    if i in adj[old_c][k]:
                        common[old_c][j][k] += 1
                        common[old_c][k][j] += 1
                    if j in adj[new_c][k]:
                        common[new_c][i][k] -= 1
                        common[new_c][k][i] -= 1
                    if i in adj[new_c][k]:
                        common[new_c][j][k] -= 1
                        common[new_c][k][j] -= 1
            
            T *= cooling
            
            # Reset temperature if stuck for a while? Or just let it cool.
            # Simple cooling is fine.
            
        return best_colors, (best_bad == 0)

    best_n = 1
    best_colors = [0] # Fallback
    
    # Try to find valid colorings for n from 26 down to 12
    # We prioritize higher n.
    for n in range(26, 11, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Allocate time based on n. Larger n is harder, but we start from 26.
        # If we fail 26 quickly, we move to 25.
        # Let's give 20 seconds per n if time permits, but we need to be fast.
        # Total time 110s. 15 values. ~7s each.
        time_per_n = 7.0
        
        for r in range(10): # Max 10 restarts
            if time.time() > deadline - 5:
                break
            seed = n * 1000 + r
            colors, is_valid = solve_sa(n, max_time=time_per_n, seed=seed)
            if is_valid:
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            # If we found a valid coloring for n, we can stop because we are maximizing n
            # and we are iterating downwards from the max possible (26).
            # However, we should check if we can go higher? No, 26 is the known max for 5 colors.
            # If we find 26, we are done.
            break

    # Fallback if we didn't find anything valid (should not happen for n < 26)
    if best_n < 12:
        n = 11
        # Cyclic construction for n=11 is not guaranteed valid for C4-free 5-coloring?
        # Actually, small n are easy.
        # Just use random search for small n with more time?
        # Or just return a known small valid one.
        # For n=4, K4 has no C4.
        n = 4
        best_colors = [0, 0, 0, 0, 0, 0]
        best_n = n

    return {'n': best_n, 'colors': best_colors}

