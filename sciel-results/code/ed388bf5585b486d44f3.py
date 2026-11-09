import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill

    # Precompute all 4-APs
    aps = []
    pos_to_aps = [[] for _ in range(N)]
    
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(N - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(aps)
            aps.append(ap)
            for pos in ap:
                pos_to_aps[pos].append(idx)
                
    def count_mono(colors):
        count = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                count += 1
        return count

    def thue_morse(n):
        return bin(n).count('1') % 2

    best_colors = np.array([thue_morse(i) for i in range(N)], dtype=np.int8)
    best_cost = count_mono(best_colors)
    
    def get_delta(colors, i):
        """Calculate change in cost if we flip colors[i]."""
        old_c = colors[i]
        new_c = 1 - old_c
        delta = 0
        for ap_idx in pos_to_aps[i]:
            ap = aps[ap_idx]
            others = [p for p in ap if p != i]
            n_old = 0
            n_new = 0
            for p in others:
                if colors[p] == old_c:
                    n_old += 1
                else:
                    n_new += 1
            
            before = 1 if n_old == 3 else 0
            after = 1 if n_new == 3 else 0
            delta += after - before
        return delta

    def hill_climb(colors):
        colors = colors.copy()
        cost = count_mono(colors)
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i = -1
            for i in range(N):
                d = get_delta(colors, i)
                if d < best_delta:
                    best_delta = d
                    best_i = i
            if best_delta < 0:
                colors[best_i] = 1 - colors[best_i]
                cost += best_delta
                improved = True
        return colors, cost

    # Run hill climb from Thue-Morse
    colors, cost = hill_climb(best_colors)
    if cost < best_cost:
        best_cost = cost
        best_colors = colors

    # Simulated Annealing with Random Restarts
    random.seed(42)
    
    # Multiple restarts to explore different basins
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        if r == 0:
            current_colors = best_colors.copy()
            current_cost = best_cost
        else:
            # Random restart
            current_colors = np.array([random.randint(0, 1) for _ in range(N)], dtype=np.int8)
            current_cost = count_mono(current_colors)
            
            # Optionally hill climb first to get to a local minimum
            current_colors, current_cost = hill_climb(current_colors)
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = current_colors.copy()
        
        # Annealing phase
        T = 50.0
        T_min = 0.1
        cooling = 0.95
        
        while T > T_min and time.time() - start_time < time_limit - 5:
            # Pick a random position to flip
            i = random.randint(0, N - 1)
            delta = get_delta(current_colors, i)
            
            # Metropolis acceptance criterion
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_colors[i] = 1 - current_colors[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = current_colors.copy()
            
            T *= cooling
    
    return best_colors.tolist()
