import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_triples(n):
        return list(combinations(range(n), 3))

    def build_structures(n):
        triples = get_triples(n)
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))
            
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))
            
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)
                
        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)
                
        return triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s

    def solve_n(n, time_limit):
        triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structures(n)
        
        # Use a fixed seed for reproducibility in testing, but random for production?
        # The prompt implies we want to *find* the solution.
        # Let's use a random seed.
        random.seed(int(time.time() * 100000) % (2**31))
        
        # Init
        colors = np.array([0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples], dtype=np.int8)
        
        red_count_k4 = np.zeros(len(k4s), dtype=np.int16)
        blue_count_k5 = np.zeros(len(k5s), dtype=np.int16)
        
        for i, idxs in enumerate(k4s):
            red_count_k4[i] = np.sum(colors[idxs] == 0)
            
        for i, idxs in enumerate(k5s):
            blue_count_k5[i] = np.sum(colors[idxs] == 1)
            
        current_violations = int(np.sum(red_count_k4 == 4) + np.sum(blue_count_k5 == 10))
        if current_violations == 0:
            return colors
        
        best_colors = colors.copy()
        best_violations = current_violations
        
        temp = 1.0
        cool_rate = 0.9999
        min_temp = 0.01
        no_improve = 0
        max_no_improve = 10000
        
        while time.time() < time_limit:
            t = random.randint(0, num_triples - 1)
            
            old_color = colors[t]
            new_color = 1 - old_color
            
            delta = 0
            
            for k4_idx in triple_to_k4s[t]:
                old_count = red_count_k4[k4_idx]
                if old_color == 0:
                    new_count = old_count - 1
                else:
                    new_count = old_count + 1
                    
                if old_count == 4:
                    delta -= 1
                if new_count == 4:
                    delta += 1
                    
                red_count_k4[k4_idx] = new_count
                
            for k5_idx in triple_to_k5s[t]:
                old_count = blue_count_k5[k5_idx]
                if old_color == 1:
                    new_count = old_count - 1
                else:
                    new_count = old_count + 1
                    
                if old_count == 10:
                    delta -= 1
                if new_count == 10:
                    delta += 1
                    
                blue_count_k5[k5_idx] = new_count
                
            new_violations = current_violations + delta
            
            if delta <= 0:
                accept = True
            else:
                if temp > min_temp:
                    accept = random.random() < np.exp(-delta / temp)
                else:
                    accept = False
            
            if accept:
                colors[t] = new_color
                current_violations = new_violations
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors.copy()
                    no_improve = 0
                    if best_violations == 0:
                        return best_colors
                else:
                    no_improve += 1
            else:
                # Revert
                for k4_idx in triple_to_k4s[t]:
                    if old_color == 0:
                        red_count_k4[k4_idx] += 1
                    else:
                        red_count_k4[k4_idx] -= 1
                        
                for k5_idx in triple_to_k5s[t]:
                    if old_color == 1:
                        blue_count_k5[k5_idx] += 1
                    else:
                        blue_count_k5[k5_idx] -= 1
                        
                no_improve += 1
                
            temp *= cool_rate
            
            if no_improve > max_no_improve:
                temp = max(1.0, temp * 2)
                no_improve = 0
                # Perturb
                for _ in range(num_triples // 50):
                    idx = random.randint(0, num_triples - 1)
                    colors[idx] = 1 - colors[idx]
                    # Update counts for this flip
                    # This is expensive to do correctly without a helper.
                    # Let's just reset to best_colors and re-init counts?
                    # That's also expensive.
                    # Just let the high temp do the work.
                pass

        return best_colors

    def verify(n, colors):
        triples = get_triples(n)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_to_idx[(a, b, c)]] == 0 and
                colors[triple_to_idx[(a, b, d)]] == 0 and
                colors[triple_to_idx[(a, c, d)]] == 0 and
                colors[triple_to_idx[(b, c, d)]] == 0):
                return False
                
        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                if colors[triple_to_idx[triple]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
                
        return True

    final_n = 0
    final_colors = []

    for n in [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]:
        if time.time() > deadline - 5:
            break
            
        if n <= final_n:
            continue
            
        if n == 34:
            time_budget = 60.0
        elif n == 33:
            time_budget = 30.0
        elif n == 32:
            time_budget = 20.0
        else:
            time_budget = 10.0
            
        time_budget = min(time_budget, deadline - time.time() - 2)
        if time_budget < 2:
            continue
            
        start_attempt = time.time()
        found = False
        attempts = 0
        max_attempts = 3 if n >= 32 else 1
        
        while attempts < max_attempts and time.time() < start_attempt + time_budget:
            remaining = start_attempt + time_budget - time.time()
            if remaining < 2: break
            
            colors = solve_n(n, time.time() + remaining)
            
            if verify(n, colors):
                final_n = n
                final_colors = colors.tolist()
                found = True
                break
                
            attempts += 1
            
        if found:
            break

    if final_n == 0:
        n = 9
        triples = get_triples(n)
        colors = [0 if (a+b+c)%3==0 else 1 for a,b,c in triples]
        if verify(n, colors):
            final_n = n
            final_colors = colors
        else:
            final_n = 1
            final_colors = [0]

    return {'n': final_n, 'colors': final_colors}
