import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    max_n_target = 26
    best_n = 1
    best_colors = []
    
    # Fallback to seed solution if we don't find anything better
    seed_n = 11
    seed_colors = [min(j - i, seed_n - j + i) - 1 for i in range(seed_n) for j in range(i + 1, seed_n)]
    
    if best_n < seed_n:
        best_n = seed_n
        best_colors = seed_colors

    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
            
        n = 1
        neighbors = [[0] for _ in range(5)]
        
        color_order = list(range(5))
        random.shuffle(color_order)
        
        def try_add_vertex(n_current):
            new_v = n_current
            assigned = [None] * new_v
            temp_neigh = [0] * 5
            
            def backtrack(u):
                if u < 0:
                    return True
                
                for c in color_order:
                    if (neighbors[c][u] & temp_neigh[c]) == 0:
                        assigned[u] = c
                        temp_neigh[c] |= (1 << u)
                        
                        if backtrack(u - 1):
                            return True
                        
                        temp_neigh[c] &= ~(1 << u)
                return False
            
            if backtrack(new_v - 1):
                for u in range(new_v):
                    c = assigned[u]
                    neighbors[c][u] |= (1 << new_v)
                    neighbors[c].append(temp_neigh[c])
                return True
            return False
        
        current_n = 1
        while current_n < max_n_target:
            if time.time() - start_time > time_limit:
                break
            if try_add_vertex(current_n):
                current_n += 1
            else:
                break
        
        if current_n > best_n:
            best_n = current_n
            color_matrix = [[-1] * current_n for _ in range(current_n)]
            for c in range(5):
                for v in range(current_n):
                    mask = neighbors[c][v]
                    for u in range(current_n):
                        if mask & (1 << u):
                            if u < v:
                                color_matrix[u][v] = c
                            else:
                                color_matrix[v][u] = c
            
            best_colors = []
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    best_colors.append(color_matrix[i][j])
                    
        if best_n >= 26:
            break

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
