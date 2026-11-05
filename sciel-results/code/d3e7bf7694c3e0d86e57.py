import time
import random
import math

def construct():
    n = 256
    deadline = time.time() + 130  # Safety margin for 150s limit
    
    # 1. Precompute all 4-term APs
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # 2. Precompute neighbors for fast delta calculation
    # neighbors[i] contains a list of tuples (other1, other2, other3)
    neighbors = [[] for _ in range(n)]
    for a, b, c, d in ap_list:
        neighbors[a].append((b, c, d))
        neighbors[b].append((a, c, d))
        neighbors[c].append((a, b, d))
        neighbors[d].append((a, b, c))
        
    def count_mono(colors):
        count = 0
        for a, b, c, d in ap_list:
            if colors[a] == colors[b] == colors[c] == colors[d]:
                count += 1
        return count

    def get_delta(colors, i):
        old_color = colors[i]
        new_color = 1 - old_color
        delta = 0
        for o1, o2, o3 in neighbors[i]:
            c1 = colors[o1]
            c2 = colors[o2]
            c3 = colors[o3]
            
            if c1 == old_color and c2 == old_color and c3 == old_color:
                delta -= 1
            elif c1 == new_color and c2 == new_color and c3 == new_color:
                delta += 1
        return delta

    # Thue-Morse initialization
    def thue_morse(i):
        return bin(i).count('1') % 2
        
    best_colors = [thue_morse(i) for i in range(n)]
    best_score = count_mono(best_colors)
    
    rng = random.Random(42)
    
    T_init = 5.0
    T_min = 0.01
    cooling_rate = 0.9998
    
    current_colors = best_colors[:]
    current_score = best_score
    
    no_improve_steps = 0
    max_no_improve_steps = 5000
    
    while time.time() < deadline:
        T = T_init
        no_improve_steps = 0
        
        # Randomize start occasionally
        if rng.random() < 0.1:
             current_colors = [rng.randint(0, 1) for _ in range(n)]
             current_score = count_mono(current_colors)
        else:
             # Perturb best
             current_colors = best_colors[:]
             for _ in range(5):
                 pos = rng.randrange(n)
                 current_colors[pos] = 1 - current_colors[pos]
             current_score = count_mono(current_colors)

        while T > T_min and no_improve_steps < max_no_improve_steps:
            if time.time() > deadline:
                break
                
            pos = rng.randrange(n)
            delta = get_delta(current_colors, pos)
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_colors[pos] = 1 - current_colors[pos]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_colors = current_colors[:]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
                
            T *= cooling_rate
        
    return best_colors

# EVOLVE-BLOCK-END
