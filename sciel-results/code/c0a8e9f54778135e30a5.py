import time
import math
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_score = -1
    best_set = [0, 1]
    
    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        if len(U) < 2:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        
        n = len(U)
        if n > 50:
            u_arr = np.array(U)
            sum_arr = u_arr[:, None] + u_arr[None, :]
            diff_arr = u_arr[:, None] - u_arr[None, :]
            n_sum = len(np.unique(sum_arr))
            n_diff = len(np.unique(diff_arr))
        else:
            sumset = set()
            diffset = set()
            for a in U:
                for b in U:
                    sumset.add(a + b)
                    diffset.add(a - b)
            n_sum = len(sumset)
            n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
            
        base = 2 * max_u + 1
        if base <= 1:
            return 0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if len(cand) < 2:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Known optimal Golomb rulers
    # Source: OEIS A005292 (lengths) and related sequences for marks
    # n=1: [0]
    # n=2: [0, 1]
    # n=3: [0, 1, 3]
    # n=4: [0, 1, 4, 6]
    # n=5: [0, 1, 4, 10, 12]
    # n=6: [0, 1, 4, 10, 18, 21]
    # n=7: [0, 1, 4, 10, 18, 23, 31]
    # n=8: [0, 1, 4, 10, 18, 23, 31, 45]
    # n=9: [0, 1, 4, 10, 18, 23, 31, 45, 52]
    # n=10: [0, 1, 4, 10, 18, 23, 31, 45, 52, 62]
    # n=11: [0, 1, 4, 10, 18, 23, 31, 45, 52, 62, 76]
    # n=12: [0, 1, 4, 10, 18, 23, 31, 45, 52, 62, 76, 90] (Length 90 is for n=12? Let's check. L(12)=90? No, L(11)=76, L(12)=? )
    # Actually, L(12) is 90? No, L(12) is likely around 90-100.
    # Let's use the ones we are sure about.
    
    golomb_rulers = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 18, 21],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 18, 23, 31, 45],
        [0, 1, 4, 10, 18, 23, 31, 45, 52],
        [0, 1, 4, 10, 18, 23, 31, 45, 52, 62],
        [0, 1, 4, 10, 18, 23, 31, 45, 52, 62, 76]
    ]
    
    for gr in golomb_rulers:
        try_set(gr)

    # 2. Exhaustive search for small n and M
    # We limit M to 30 and n to 7 to keep it fast but catch n=6,7 optimal rulers
    max_m_exhaustive = 30
    max_n_exhaustive = 7
    
    for n in range(2, max_n_exhaustive + 1):
        if time.time() > deadline - 30:
            break
        range_m = range(1, max_m_exhaustive + 1)
        for combo in combinations(range_m, n - 1):
            if time.time() > deadline - 30:
                break
            cand = [0] + list(combo)
            try_set(cand)

    # 3. Local Search
    def local_search(initial_set, max_steps=2000):
        nonlocal best_score, best_set
        current_set = list(initial_set)
        current_score = compute_score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
            
        for step in range(max_steps):
            if time.time() > deadline - 5:
                return
            
            neighbor = list(current_set)
            move_type = random.choice(['add', 'remove', 'move', 'expand'])
            
            if move_type == 'add':
                max_val = max(current_set)
                limit = max_val + 20 if max_val < 100 else max_val + 10
                candidates = [x for x in range(1, limit + 1) if x not in current_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                neighbor.append(x)
                
            elif move_type == 'remove':
                if len(current_set) <= 2:
                    continue
                removable = [x for x in current_set if x != 0]
                if not removable:
                    continue
                x = random.choice(removable)
                neighbor.remove(x)
                
            elif move_type == 'move':
                if len(current_set) <= 2:
                    continue
                removable = [x for x in current_set if x != 0]
                if not removable:
                    continue
                x = random.choice(removable)
                neighbor.remove(x)
                max_val = max(neighbor) if neighbor else 1
                limit = max_val + 20 if max_val < 100 else max_val + 10
                candidates = [y for y in range(1, limit + 1) if y not in neighbor]
                if not candidates:
                    continue
                y = random.choice(candidates)
                neighbor.append(y)
            
            elif move_type == 'expand':
                # Specifically try to add a large number to increase diameter
                # This might help if the current set is "too compact" but has good ratio
                # However, increasing diameter hurts the denominator.
                # So this is risky. Better to use 'add' with reasonable limits.
                max_val = max(current_set)
                # Try adding a number that is a multiple or related to existing structure?
                # Just pick a random number in [max_val, max_val + 50]
                candidates = [x for x in range(max_val + 1, max_val + 51) if x not in current_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                neighbor.append(x)

            neighbor_score = compute_score(neighbor)
            
            if neighbor_score > current_score:
                current_set = neighbor
                current_score = neighbor_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
            else:
                # Simulated Annealing
                # Accept worse moves with probability exp(-delta/T)
                # T decreases over time
                T = 50 / (step + 1)
                delta = current_score - neighbor_score
                if delta > 0:
                    prob = math.exp(-delta / T)
                else:
                    prob = 1.0
                    
                if random.random() < prob:
                    current_set = neighbor
                    current_score = neighbor_score

    # Run local search from multiple starting points
    # 1. From Golomb rulers
    for gr in golomb_rulers:
        if time.time() > deadline - 20:
            break
        local_search(gr, max_steps=500)
        
    # 2. From best exhaustive set
    if time.time() < deadline - 20:
        local_search(best_set, max_steps=1000)

    # 3. From random sets with larger diameters
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        n = random.randint(5, 15)
        max_m = random.randint(20, 100)
        rand_set = [0] + random.sample(range(1, max_m + 1), n - 1)
        local_search(rand_set, max_steps=200)
        
    return list(best_set)

construct()
