import time
import random
import math

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. We use 120s to be safe.
    deadline = start_time + 120.0
    
    # Helper to generate initial sets
    def generate_initial_sets():
        sets = []
        
        # 1. Simple interval
        sets.append(set(range(1, 33)))
        
        # 2. Sums of two squares (known to have good additive properties)
        # We need 32 distinct values in 1..512.
        # i^2 + j^2 for small i,j
        candidates = set()
        for i in range(1, 20):
            for j in range(1, 20):
                val = i*i + j*j
                if 1 <= val <= 512:
                    candidates.add(val)
        if len(candidates) >= 32:
            # Pick 32 randomly or greedily? Random is fine for diversity
            sets.append(set(random.sample(sorted(candidates), 32)))
            
        # 3. Union of two progressions
        # e.g., 1, 4, 7, ... and 2, 5, 8, ...
        # Let's try a grid-like structure
        # x = i * 10 + j for i in 1..5, j in 1..6 -> 30 elements
        # Add 2 more
        grid_set = set()
        for i in range(1, 6):
            for j in range(1, 7):
                grid_set.add(i * 10 + j)
        if len(grid_set) >= 32:
             sets.append(set(random.sample(sorted(grid_set), 32)))
             
        # 4. Random
        sets.append(set(random.sample(range(1, 513), 32)))
        
        # 5. Another random
        sets.append(set(random.sample(range(1, 513), 32)))
        
        return sets

    # Efficient score calculation
    def get_score(S):
        S_set = S
        S_sorted = sorted(S)
        d_set = set()
        n = len(S_sorted)
        for i in range(n):
            a = S_sorted[i]
            for j in range(i + 1, n):
                b = S_sorted[j]
                d = b - a
                c = b + d
                if c in S_set:
                    d_set.add(d)
        return len(d_set)

    best_set = None
    best_score = -1
    
    # We will run multiple SA instances or a long SA with restarts
    # Given the time budget, we can afford several restarts.
    
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        # Check time
        if time.time() >= deadline - 5.0:
            break
            
        # Select an initial set
        # For the first restart, try the structured ones, then random
        if r == 0:
            init_sets = generate_initial_sets()
            # Evaluate all initial sets and pick the best one to start SA from
            current_set = max(init_sets, key=get_score)
        else:
            current_set = set(random.sample(range(1, 513), 32))
            
        current_score = get_score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
            
        # Simulated Annealing Parameters
        # Start temperature high enough to accept bad moves, cool down
        start_temp = 10.0
        end_temp = 0.01
        cooling_rate = 0.999 # Adjust based on expected steps
        
        # Estimate steps: we want to use time_per_restart
        # Let's just run until time limit for this restart
        restart_deadline = start_time + (r + 1) * time_per_restart
        
        temp = start_temp
        
        # Pre-calculate list of elements for faster choice? 
        # set -> list conversion is O(N). N=32. It's fine.
        
        while time.time() < restart_deadline:
            # Generate neighbor
            # Strategy: 50% random replacement, 50% local nudge
            if random.random() < 0.5:
                # Random replacement
                elem_to_replace = random.choice(list(current_set))
                # Pick a new value not in set
                # Efficient way: pick random 1..512, if in set, retry (rare)
                new_val = random.randint(1, 512)
                if new_val in current_set:
                    continue
            else:
                # Local nudge
                elem_to_replace = random.choice(list(current_set))
                delta = random.randint(-10, 10)
                if delta == 0:
                    continue
                new_val = elem_to_replace + delta
                if new_val < 1 or new_val > 512:
                    continue
                if new_val in current_set:
                    continue
            
            # Apply move
            new_set = set(current_set)
            new_set.remove(elem_to_replace)
            new_set.add(new_val)
            
            new_score = get_score(new_set)
            delta_score = new_score - current_score
            
            if delta_score >= 0:
                # Accept improving or equal
                current_set = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            else:
                # Accept worsening with probability exp(delta_score / temp)
                if temp > 0:
                    prob = math.exp(delta_score / temp)
                    if random.random() < prob:
                        current_set = new_set
                        current_score = new_score
            
            # Cool down
            temp *= cooling_rate
            if temp < end_temp:
                temp = end_temp

    return sorted(best_set)
