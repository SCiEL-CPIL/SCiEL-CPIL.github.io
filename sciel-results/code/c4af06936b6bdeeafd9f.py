import random
import math

def construct():
    n = 256
    # Precompute all 4-term arithmetic progressions
    all_aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(max_a):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            all_aps.append(ap)
    
    # Precompute for each position p, the list of (other1, other2, other3) for each AP containing p
    neighbors = [[] for _ in range(n)]
    for ap in all_aps:
        p0, p1, p2, p3 = ap
        neighbors[p0].append((p1, p2, p3))
        neighbors[p1].append((p0, p2, p3))
        neighbors[p2].append((p0, p1, p3))
        neighbors[p3].append((p0, p1, p2))

    def count_mono_4ap(color):
        count = 0
        for ap in all_aps:
            c0 = color[ap[0]]
            if c0 == color[ap[1]] == color[ap[2]] == color[ap[3]]:
                count += 1
        return count

    def delta_from_flip(color, p):
        old_c = color[p]
        new_c = 1 - old_c
        delta = 0
        for o1, o2, o3 in neighbors[p]:
            c1 = color[o1]
            c2 = color[o2]
            c3 = color[o3]
            if c1 == c2 == c3:
                if c1 == old_c:
                    delta -= 1
                else:
                    delta += 1
        return delta

    best_color = None
    best_score = float('inf')
    
    # Run multiple restarts
    # 50 runs * 200k iters = 10M flips.
    # Estimated time: ~10-20s.
    
    for run in range(50):
        random.seed(run * 1000)
        color = [random.randint(0, 1) for _ in range(n)]
        current_score = count_mono_4ap(color)
        
        T_start = 2.0
        T_end = 0.05
        iterations = 200000
        cooling_rate = (T_end / T_start) ** (1.0 / iterations)
        T = T_start
        
        best_run_score = current_score
        best_run_color = color[:]
        
        for _ in range(iterations):
            p = int(random.random() * n)
            delta = delta_from_flip(color, p)
            
            if delta < 0:
                color[p] = 1 - color[p]
                current_score += delta
                if current_score < best_run_score:
                    best_run_score = current_score
                    best_run_color = color[:]
            else:
                if delta == 0:
                    color[p] = 1 - color[p]
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        color[p] = 1 - color[p]
                        current_score += delta
            
            T *= cooling_rate
        
        if best_run_score < best_score:
            best_score = best_run_score
            best_color = best_run_color
            
    return best_color
