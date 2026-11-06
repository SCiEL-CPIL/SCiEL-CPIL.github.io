import numpy as np
import random
import time
from itertools import combinations, permutations

def construct():
    # Precompute base cycles for 6 vertices
    base_cycles = []
    for perm in permutations([1, 2, 3, 4, 5]):
        cycle = (0, perm[0], perm[1], perm[2], perm[3], perm[4])
        if cycle[1] < cycle[5]:
            base_cycles.append(cycle)
            
    def count_c6s_in_subset(adj_sub):
        count = 0
        for cycle in base_cycles:
            if adj_sub[cycle[0], cycle[1]] and adj_sub[cycle[1], cycle[2]] and \
               adj_sub[cycle[2], cycle[3]] and adj_sub[cycle[3], cycle[4]] and \
               adj_sub[cycle[4], cycle[5]] and adj_sub[cycle[5], cycle[0]]:
                count += 1
        return count

    def get_c6_count(colors, n):
        adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i, j] = True
                adj[c][j, i] = True
                idx += 1
        
        total_c6s = 0
        # Precompute subsets for this n
        # This is done once per n, but we call this function many times.
        # We should cache subsets per n.
        if not hasattr(get_c6_count, 'subset_cache'):
            get_c6_count.subset_cache = {}
        if n not in get_c6_count.subset_cache:
            get_c6_count.subset_cache[n] = list(combinations(range(n), 6))
        all_subsets = get_c6_count.subset_cache[n]
        
        for c in range(4):
            for subset in all_subsets:
                adj_sub = np.zeros((6, 6), dtype=bool)
                for i in range(6):
                    for j in range(6):
                        if i != j:
                            adj_sub[i, j] = adj[c][subset[i], subset[j]]
                total_c6s += count_c6s_in_subset(adj_sub)
        return total_c6s

    def find_valid_coloring(n, time_limit):
        random.seed(42)
        best_colors = None
        best_score = float('inf')
        start_time = time.time()
        
        colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        
        while time.time() - start_time < time_limit:
            current_c6s = get_c6_count(colors, n)
            if current_c6s == 0:
                return colors[:]
            
            if current_c6s < best_score:
                best_score = current_c6s
                best_colors = colors[:]
                
            # Local search step
            edge_idx = random.randint(0, len(colors) - 1)
            old_color = colors[edge_idx]
            
            best_new_color = old_color
            best_new_score = current_c6s
            
            for new_color in range(4):
                if new_color == old_color:
                    continue
                colors[edge_idx] = new_color
                new_score = get_c6_count(colors, n)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_color = new_color
                colors[edge_idx] = old_color
                
            if best_new_color != old_color:
                colors[edge_idx] = best_new_color
            else:
                # Random move
                new_color = random.choice([c for c in range(4) if c != old_color])
                colors[edge_idx] = new_color

        return best_colors if best_colors is not None else None

    # Try n=17, 16, 15...
    for n in [17, 16, 15, 14, 13, 12, 11, 10, 9, 8]:
        if time.time() - 0 > 110: # rough check
             pass
        # Allocate time
        if n == 17:
            limit = 100
        else:
            limit = 10
            
        result = find_valid_coloring(n, limit)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
