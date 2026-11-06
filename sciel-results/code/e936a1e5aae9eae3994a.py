import time
import random
import math

def construct():
    n = 256
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            aps.append((p0, p1, p2, p3))
    
    num_aps = len(aps)
    
    pos_aps = [[] for _ in range(n)]
    for idx, (p0, p1, p2, p3) in enumerate(aps):
        pos_aps[p0].append((idx, (p1, p2, p3)))
        pos_aps[p1].append((idx, (p0, p2, p3)))
        pos_aps[p2].append((idx, (p0, p1, p3)))
        pos_aps[p3].append((idx, (p0, p1, p2)))
        
    def count_mono(colors):
        cnt = 0
        for p0, p1, p2, p3 in aps:
            if colors[p0] == colors[p1] == colors[p2] == colors[p3]:
                cnt += 1
        return cnt

    def run_sa(initial_colors, time_limit):
        colors = initial_colors[:]
        count = count_mono(colors)
        best_colors = colors[:]
        best_count = count
        
        T = 5.0
        cooling = 0.99999
        
        while time.time() < time_limit:
            i = random.randint(0, n - 1)
            delta_e = 0
            for idx, others in pos_aps[i]:
                o0, o1, o2 = others
                if colors[o0] == colors[o1] == colors[o2]:
                    if colors[i] == colors[o0]:
                        delta_e -= 1
                    else:
                        delta_e += 1
            
            if delta_e <= 0:
                accept = True
            else:
                if T > 0:
                    prob = math.exp(-delta_e / T)
                else:
                    prob = 0
                accept = random.random() < prob
            
            if accept:
                colors[i] = 1 - colors[i]
                count += delta_e
                if count < best_count:
                    best_count = count
                    best_colors = colors[:]
            
            T *= cooling
            if T < 0.001:
                T = 0.001

        return best_colors, best_count

    deadline = time.time() + 140
    
    best_overall = None
    best_score = float('inf')
    
    # Generate diverse initializations
    inits = []
    # Thue-Morse is a strong candidate
    inits.append([bin(i).count('1') % 2 for i in range(n)])
    # Randoms
    for _ in range(4):
        inits.append([random.randint(0, 1) for _ in range(n)])
    # Seed
    inits.append([(i // 2) % 2 for i in range(n)])
    
    # Shuffle to mix good and bad starts? Or just run them in order.
    # Thue-Morse is likely best, so run it first or give it more time?
    # Let's just run 5 restarts with equal time.
    
    runs = 5
    time_per_run = (deadline - time.time()) / runs
    
    for k in range(min(runs, len(inits))):
        if time.time() > deadline - 2:
            break
        t_limit = time.time() + time_per_run
        colors, score = run_sa(inits[k], t_limit)
        if score < best_score:
            best_score = score
            best_overall = colors

    if best_overall is None:
        best_overall = inits[0]
        
    return best_overall
