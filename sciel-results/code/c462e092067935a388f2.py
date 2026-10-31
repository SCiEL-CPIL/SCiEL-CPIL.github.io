import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
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
        
        # Use sets for sum and diff
        # Optimization: If n is small, list comprehension is fast
        n = len(U)
        if n > 50:
            # Fallback to numpy for large sets, though unlikely to be optimal for this score
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

    # 1. Initialize with some known good structures (Golomb Rulers)
    # Small optimal Golomb rulers
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

    # 2. Local Search
    def local_search(initial_set, max_steps=1000):
        nonlocal best_score, best_set
        current_set = list(initial_set)
        current_score = compute_score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
            
        for step in range(max_steps):
            if time.time() > deadline:
                return
            
            # Generate a neighbor
            neighbor = list(current_set)
            move_type = random.choice(['add', 'remove', 'move'])
            
            if move_type == 'add':
                # Try adding a small number
                max_val = max(current_set)
                # Search range around current max
                candidates = [x for x in range(1, max_val + 20) if x not in current_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                neighbor.append(x)
                
            elif move_type == 'remove':
                if len(current_set) <= 2:
                    continue
                # Don't remove 0
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
                candidates = [y for y in range(1, max_val + 20) if y not in neighbor]
                if not candidates:
                    continue
                y = random.choice(candidates)
                neighbor.append(y)
            
            neighbor_score = compute_score(neighbor)
            
            # Accept if better or with small probability (Simulated Annealing style)
            if neighbor_score > current_score:
                current_set = neighbor
                current_score = neighbor_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
            else:
                # Small probability of accepting worse move to escape local minima
                # Probability decreases with "temperature" (step number)
                temp = 100 / (step + 1)
                if random.random() < temp * (current_score - neighbor_score) * -1:
                    current_set = neighbor
                    current_score = neighbor_score

    # Run local search from multiple starting points
    # 1. From Golomb rulers
    for gr in golomb_rulers:
        if time.time() > deadline:
            break
        local_search(gr, max_steps=500)
        
    # 2. From random sets
    for _ in range(50):
        if time.time() > deadline:
            break
        n = random.randint(4, 15)
        max_m = random.randint(20, 100)
        # Generate random set
        rand_set = [0] + random.sample(range(1, max_m + 1), n - 1)
        local_search(rand_set, max_steps=200)
        
    return list(best_set)

construct()
