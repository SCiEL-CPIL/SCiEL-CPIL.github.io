import math
import random
import time

def construct():
    n = 256
    nums = list(range(2, n + 1))
    
    # Build conflict graph
    conflict = {a: set() for a in nums}
    for i, a in enumerate(nums):
        for b in nums[i+1:]:
            g = math.gcd(a, b)
            l = a * b // g
            if l <= n:
                conflict[a].add(b)
                conflict[b].add(a)
                
    weight = {a: 1.0/a for a in nums}
    
    # Helper to check if a node can be added to a set
    def can_add(a, current_set):
        return not (conflict[a] & current_set)
        
    # Helper to calculate score
    def score(s):
        return sum(weight[a] for a in s)
        
    # Initial solution: Random Greedy
    def initial_solution():
        s = set()
        available = nums[:]
        random.shuffle(available)
        for a in available:
            if can_add(a, s):
                s.add(a)
        return s
        
    # Simulated Annealing
    start_time = time.time()
    timeout = 55.0 # Leave 5s buffer
    
    current_set = initial_solution()
    current_score = score(current_set)
    best_set = set(current_set)
    best_score = current_score
    
    T = 1.0
    cooling = 0.9995
    
    # Precompute weights for faster access? Not necessary for 255 items.
    
    while T > 0.01 and (time.time() - start_time) < timeout:
        # Generate a random neighbor
        move_type = random.random()
        new_set = set(current_set)
        
        if move_type < 0.3:
            # Try Add
            candidates = [a for a in nums if a not in new_set and can_add(a, new_set)]
            if candidates:
                a = random.choice(candidates)
                new_set.add(a)
            else:
                continue # No valid add, skip step
        elif move_type < 0.5:
            # Try Remove
            if not new_set:
                continue
            a = random.choice(list(new_set))
            new_set.remove(a)
        else:
            # Try Swap
            if not new_set:
                continue
            a = random.choice(list(new_set))
            # Find candidates for b
            # b must not be in new_set, and b must be compatible with (new_set - {a})
            temp_set = new_set - {a}
            candidates = [b for b in nums if b not in new_set and can_add(b, temp_set)]
            if candidates:
                b = random.choice(candidates)
                new_set.remove(a)
                new_set.add(b)
            else:
                continue # No valid swap
        
        new_score = score(new_set)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
                
        T *= cooling
        
    return list(best_set)
