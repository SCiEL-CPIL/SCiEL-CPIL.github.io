import time
import random

DEADLINE = 85.0

def reciprocal_sum(selected):
    return sum(1.0/x for x in selected)

def can_add_fast(selected_list, sel_set, x):
    """Check if adding x creates a 3-AP with existing elements"""
    for a in selected_list:
        # b = 2*a - x: check if b exists and b != x, b != a
        b = 2*a - x
        if b in sel_set and b != x and b != a:
            return False
        # b = 2*x - a: check if b exists and b != x, b != a
        b = 2*x - a
        if b in sel_set and b != x and b != a:
            return False
    return True

def is_3ap_free(selected):
    sel_set = set(selected)
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            a, b = selected[i], selected[j]
            c = 2*b - a
            if c in sel_set and c != a and c != b:
                return False
    return True

def greedy_construct(order):
    selected = []
    sel_set = set()
    for x in order:
        if 1 <= x <= 256 and x not in sel_set:
            if can_add_fast(selected, sel_set, x):
                selected.append(x)
                sel_set.add(x)
    return selected

def base_p_no_zero(p, n):
    result = []
    for x in range(1, n+1):
        t = x
        valid = True
        while t > 0:
            if t % p == 0:
                valid = False
                break
            t //= p
        if valid:
            result.append(x)
    return result

def local_search(selected, time_limit=75.0):
    current = list(selected)
    current_set = set(current)
    current_sum = reciprocal_sum(current)
    best_sum = current_sum
    best_state = current[:]
    
    start_time = time.time()
    T = 0.05
    cooling = 0.9995
    min_T = 1e-7
    
    all_candidates = list(range(1, 257))
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
        
        r = random.random()
        
        if r < 0.6 and len(current) > 1:
            # Swap
            idx = random.randint(0, len(current)-1)
            to_remove = current[idx]
            to_add = random.choice(all_candidates)
            if to_add == to_remove or to_add in current_set:
                T *= cooling
                continue
            
            # Check: remove to_remove, add to_add
            new_set = set(current_set)
            new_set.remove(to_remove)
            new_set.add(to_add)
            
            # Check 3-AP-free efficiently
            valid = True
            # Check all pairs in new_set
            new_list = list(new_set)
            for i in range(len(new_list)):
                for j in range(i+1, len(new_list)):
                    a, b = new_list[i], new_list[j]
                    c = 2*b - a
                    if c in new_set and c != a and c != b:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                T *= cooling
                continue
            
            new_sum = current_sum - 1.0/to_remove + 1.0/to_add
            delta = new_sum - current_sum
            
            if delta > 0 or random.random() < (2.718281828 ** (delta / T)):
                current = list(new_set)
                current_set = new_set
                current_sum = new_sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_state = current[:]
        elif r < 0.85:
            # Add
            to_add = random.choice(all_candidates)
            if to_add in current_set:
                T *= cooling
                continue
            
            if can_add_fast(current, current_set, to_add):
                new_set = set(current_set)
                new_set.add(to_add)
                new_sum = current_sum + 1.0/to_add
                delta = new_sum - current_sum
                
                if delta > 0 or random.random() < (2.718281828 ** (delta / T)):
                    current = list(new_set)
                    current_set = new_set
                    current_sum = new_sum
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_state = current[:]
        else:
            # Remove (rare)
            if len(current) <= 1:
                T *= cooling
                continue
            idx = random.randint(0, len(current)-1)
            to_remove = current[idx]
            new_set = set(current_set)
            new_set.remove(to_remove)
            new_sum = current_sum - 1.0/to_remove
            delta = new_sum - current_sum
            
            if random.random() < (2.718281828 ** (delta / T)):
                current = list(new_set)
                current_set = new_set
                current_sum = new_sum
        
        T *= cooling
    
    return best_state, best_sum

def construct():
    start_time = time.time()
    random.seed(42)
    
    candidates = []
    
    # Greedy ascending
    candidates.append(greedy_construct(list(range(1, 257))))
    # Greedy descending
    candidates.append(greedy_construct(list(range(256, 0, -1))))
    # Base-p constructions
    for p in [3, 5, 7, 11]:
        c = base_p_no_zero(p, 256)
        if c:
            candidates.append(c)
    # Randomized greedy with bias toward small numbers
    for seed in range(1, 6):
        random.seed(seed)
        order = list(range(1, 257))
        # Weighted shuffle: smaller numbers more likely to come first
        weights = [1.0/x for x in order]
        total = sum(weights)
        sampled = []
        rem = order[:]
        rem_w = weights[:]
        for _ in range(256):
            r = random.random() * sum(rem_w)
            cum = 0
            for i in range(len(rem)):
                cum += rem_w[i]
                if cum >= r:
                    sampled.append(rem[i])
                    rem.pop(i)
                    rem_w.pop(i)
                    break
        candidates.append(greedy_construct(sampled))
    
    best_candidate = None
    best_score = 0
    for c in candidates:
        if not c or not is_3ap_free(c):
            continue
        s = reciprocal_sum(c)
        if s > best_score:
            best_score = s
            best_candidate = c
    
    if best_candidate is None:
        return [1]
    
    elapsed = time.time() - start_time
    time_left = DEADLINE - elapsed - 5.0
    if time_left > 2.0:
        best_state, best_sum = local_search(best_candidate, time_limit=time_left)
        if best_sum > best_score:
            best_candidate = best_state
            best_score = best_sum
    
    if not is_3ap_free(best_candidate):
        best_candidate = [1]
    
    return best_candidate
