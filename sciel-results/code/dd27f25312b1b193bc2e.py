import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin
    
    def time_left():
        return deadline - time.time()
    
    def try_solve(n):
        if n < 5:
            return None
        
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        # Map each K4 to its 4 triple indices
        k4_triples = []
        for k4 in k4s:
            ts = [triple_idx[tri] for tri in combinations(k4, 3)]
            k4_triples.append(ts)
        
        # Map each K5 to its 10 triple indices
        k5_triples = []
        for k5 in k5s:
            ts = [triple_idx[tri] for tri in combinations(k5, 3)]
            k5_triples.append(ts)
        
        # Map triple index to list of K4 indices and K5 indices
        triple_to_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4[t_idx].append(i)
        
        triple_to_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5[t_idx].append(i)
        
        # Structured initial coloring: mod 3
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
        
        # Initialize counts
        red_count_k4 = [0] * len(k4s)
        blue_count_k5 = [0] * len(k5s)
        
        for i, ts in enumerate(k4_triples):
            cnt = sum(1 for t_idx in ts if colors[t_idx] == 0)
            red_count_k4[i] = cnt
            
        for i, ts in enumerate(k5_triples):
            cnt = sum(1 for t_idx in ts if colors[t_idx] == 1)
            blue_count_k5[i] = cnt
            
        red_violations = sum(1 for c in red_count_k4 if c == 4)
        blue_violations = sum(1 for c in blue_count_k5 if c == 10)
        
        if red_violations == 0 and blue_violations == 0:
            return colors
        
        # Simulated annealing with violation targeting
        T = 2.0
        cooling = 0.9995
        min_T = 0.005
        max_steps = 300000
        
        for step in range(max_steps):
            if time_left() < 3.0:
                break
            
            if red_violations == 0 and blue_violations == 0:
                return colors
            
            # Targeted move: pick a triple from a violating set
            if random.random() < 0.7:
                # Pick from violations
                if red_violations > 0 and (blue_violations == 0 or random.random() < 0.5):
                    # Pick a random violating K4
                    violating_k4s = [i for i in range(len(k4s)) if red_count_k4[i] == 4]
                    if violating_k4s:
                        k4_i = random.choice(violating_k4s)
                        t_idx = random.choice(k4_triples[k4_i])
                    else:
                        t_idx = random.randint(0, num_triples - 1)
                else:
                    # Pick a random violating K5
                    violating_k5s = [i for i in range(len(k5s)) if blue_count_k5[i] == 10]
                    if violating_k5s:
                        k5_i = random.choice(violating_k5s)
                        t_idx = random.choice(k5_triples[k5_i])
                    else:
                        t_idx = random.randint(0, num_triples - 1)
            else:
                t_idx = random.randint(0, num_triples - 1)
            
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Calculate delta
            delta = 0
            
            # Update K4 counts
            for k4_idx in triple_to_k4[t_idx]:
                if old_color == 0:  # Red -> Blue
                    old_cnt = red_count_k4[k4_idx]
                    if old_cnt == 4:
                        delta -= 1
                    elif old_cnt == 3:
                        delta += 1
                else:  # Blue -> Red
                    old_cnt = red_count_k4[k4_idx]
                    if old_cnt == 3:
                        delta += 1
                    elif old_cnt == 4:
                        delta -= 1
            
            # Update K5 counts
            for k5_idx in triple_to_k5[t_idx]:
                if old_color == 1:  # Blue -> Red
                    old_cnt = blue_count_k5[k5_idx]
                    if old_cnt == 10:
                        delta -= 1
                    elif old_cnt == 9:
                        delta += 1
                else:  # Red -> Blue
                    old_cnt = blue_count_k5[k5_idx]
                    if old_cnt == 9:
                        delta += 1
                    elif old_cnt == 10:
                        delta -= 1
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.5:
                    accept = True
            else:
                if T > min_T:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                colors[t_idx] = new_color
                red_violations = max(0, red_violations)
                blue_violations = max(0, blue_violations)
                
                # Update counts
                for k4_idx in triple_to_k4[t_idx]:
                    if old_color == 0:
                        red_count_k4[k4_idx] -= 1
                    else:
                        red_count_k4[k4_idx] += 1
                    if red_count_k4[k4_idx] == 4:
                        red_violations += 1
                    elif red_count_k4[k4_idx] == 3:
                        red_violations -= 1
                for k5_idx in triple_to_k5[t_idx]:
                    if old_color == 1:
                        blue_count_k5[k5_idx] -= 1
                    else:
                        blue_count_k5[k5_idx] += 1
                    if blue_count_k5[k5_idx] == 10:
                        blue_violations += 1
                    elif blue_count_k5[k5_idx] == 9:
                        blue_violations -= 1
                
                # Clamp
                red_violations = max(0, red_violations)
                blue_violations = max(0, blue_violations)
            else:
                # Rollback not needed since we didn't apply
                pass
            
            # Cool down
            T = max(min_T, T * cooling)
        
        return None
    
    # Try n=34 first
    for n in [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time_left() < 5.0:
            break
        
        colors = try_solve(n)
        if colors is not None:
            return {'n': n, 'colors': colors}
    
    # Fallback: n=9 mod-3 coloring
    n_fallback = 9
    triples_fb = list(combinations(range(n_fallback), 3))
    colors_fb = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_fb]
    return {'n': n_fallback, 'colors': colors_fb}
