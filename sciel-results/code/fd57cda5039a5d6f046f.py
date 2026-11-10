import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    def get_indices(n):
        idx = {}
        triples = []
        i = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx[(a,b,c)] = i
                    triples.append((a,b,c))
                    i += 1
        return idx, triples

    def precompute_structures(n, idx):
        k4_containing = [[] for _ in range(len(idx))]
        k5_containing = [[] for _ in range(len(idx))]
        
        # Precompute K4s
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        k4 = (a,b,c,d)
                        e1 = idx[(a,b,c)]
                        e2 = idx[(a,b,d)]
                        e3 = idx[(a,c,d)]
                        e4 = idx[(b,c,d)]
                        k4_containing[e1].append(k4)
                        k4_containing[e2].append(k4)
                        k4_containing[e3].append(k4)
                        k4_containing[e4].append(k4)
                        
        # Precompute K5s
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        for e in range(d+1, n):
                            k5 = (a,b,c,d,e)
                            vs = [a,b,c,d,e]
                            for combo in combinations(vs, 3):
                                e_idx = idx[combo]
                                k5_containing[e_idx].append(k5)
                                
        return k4_containing, k5_containing

    def is_valid(n, colors):
        idx, _ = get_indices(n)
        # Check red K4
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        if colors[idx[(a,b,c)]] == 0 and \
                           colors[idx[(a,b,d)]] == 0 and \
                           colors[idx[(a,c,d)]] == 0 and \
                           colors[idx[(b,c,d)]] == 0:
                            return False
        # Check blue K5
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        for e in range(d+1, n):
                            vs = [a,b,c,d,e]
                            all_blue = True
                            for combo in combinations(vs, 3):
                                if colors[idx[combo]] != 1:
                                    all_blue = False
                                    break
                            if all_blue:
                                return False
        return True

    def solve_sa(n, time_limit):
        idx, triples = get_indices(n)
        num_edges = len(triples)
        
        k4_containing, k5_containing = precompute_structures(n, idx)
        
        colors = [random.randint(0, 1) for _ in range(num_edges)]
        
        # Initial energy
        energy = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        if colors[idx[(a,b,c)]] == 0 and \
                           colors[idx[(a,b,d)]] == 0 and \
                           colors[idx[(a,c,d)]] == 0 and \
                           colors[idx[(b,c,d)]] == 0:
                            energy += 1
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        for e in range(d+1, n):
                            vs = [a,b,c,d,e]
                            all_blue = True
                            for combo in combinations(vs, 3):
                                if colors[idx[combo]] != 1:
                                    all_blue = False
                                    break
                            if all_blue:
                                energy += 1
                                
        best_energy = energy
        best_colors = colors[:]
        
        T = 1.0
        cooling = 0.99999
        min_T = 0.01
        stuck_counter = 0
        max_stuck = 5000
        
        while time.time() < time_limit:
            if T < min_T:
                break
            
            i = random.randint(0, num_edges - 1)
            old_color = colors[i]
            new_color = 1 - old_color
            
            delta = 0
            
            for k4 in k4_containing[i]:
                a,b,c,d = k4
                e1 = idx[(a,b,c)]
                e2 = idx[(a,b,d)]
                e3 = idx[(a,c,d)]
                e4 = idx[(b,c,d)]
                
                was_red = True
                for e_idx in [e1, e2, e3, e4]:
                    col = old_color if e_idx == i else colors[e_idx]
                    if col != 0:
                        was_red = False
                        break
                
                is_red = True
                for e_idx in [e1, e2, e3, e4]:
                    col = new_color if e_idx == i else colors[e_idx]
                    if col != 0:
                        is_red = False
                        break
                
                if was_red and not is_red:
                    delta -= 1
                elif not was_red and is_red:
                    delta += 1
            
            for k5 in k5_containing[i]:
                vs = list(k5)
                edges = [idx[combo] for combo in combinations(vs, 3)]
                
                was_blue = True
                for e_idx in edges:
                    col = old_color if e_idx == i else colors[e_idx]
                    if col != 1:
                        was_blue = False
                        break
                
                is_blue = True
                for e_idx in edges:
                    col = new_color if e_idx == i else colors[e_idx]
                    if col != 1:
                        is_blue = False
                        break
                
                if was_blue and not is_blue:
                    delta -= 1
                elif not was_blue and is_blue:
                    delta += 1
            
            if delta <= 0 or random.random() < pow(2.71828, -delta / T):
                colors[i] = new_color
                energy += delta
                if energy < best_energy:
                    best_energy = energy
                    best_colors = colors[:]
                    stuck_counter = 0
                    if best_energy == 0:
                        return best_colors
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
                
            T *= cooling
            if stuck_counter > max_stuck:
                stuck_counter = 0
                colors = best_colors[:]
                for _ in range(10):
                    j = random.randint(0, num_edges - 1)
                    colors[j] = 1 - colors[j]
                T = 1.0
        
        return best_colors

    target_ns = [34, 33, 32, 31, 30, 25, 20]
    
    for n in target_ns:
        if time.time() > deadline - 10:
            break
            
        time_limit = deadline - time.time() - 2
        if time_limit < 1:
            continue
            
        colors = solve_sa(n, time_limit)
        
        if is_valid(n, colors):
            return {'n': n, 'colors': colors}
            
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
