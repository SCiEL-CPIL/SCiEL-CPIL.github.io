import time
import random
import numpy as np

def construct():
    n = 12
    # Precompute subset masks
    masks = np.zeros((1 << n, n), dtype=np.int32)
    for i in range(n):
        masks[:, i] = (np.arange(1 << n) >> i) & 1

    # Initial sequence: Powers of 2
    seq = [1 << i for i in range(n)]
    best_seq = seq[:]
    best_max = max(seq)
    
    PENALTY = 10000
    
    def get_cost(arr):
        arr_np = np.array(arr, dtype=np.int32)
        sums = masks @ arr_np
        unique_sums = np.unique(sums)
        num_unique = len(unique_sums)
        num_collisions = (1 << n) - num_unique
        if len(set(arr)) != n:
            num_collisions += 100
        return max(arr) + PENALTY * num_collisions

    def is_valid(arr):
        arr_np = np.array(arr, dtype=np.int32)
        sums = masks @ arr_np
        return len(np.unique(sums)) == (1 << n) and len(set(arr)) == n

    deadline = time.time() + 85
    current_seq = seq[:]
    current_cost = get_cost(current_seq)
    
    temperature = 1000.0
    cooling_rate = 0.999
    
    for iteration in range(2000000):
        if time.time() > deadline:
            break
            
        if temperature < 1.0:
            temperature = 1.0
            
        idx = random.randint(0, n - 1)
        
        move_type = random.random()
        
        if move_type < 0.8:
            step = random.choice([-1, 1])
            new_val = current_seq[idx] + step
        else:
            # Larger jump, biased towards reducing max
            if current_seq[idx] > 100:
                new_val = random.randint(1, current_seq[idx])
            else:
                new_val = random.randint(1, 500)
        
        if new_val <= 0:
            continue
            
        temp_seq = current_seq[:]
        temp_seq[idx] = new_val
        
        if len(set(temp_seq)) != n:
            continue
            
        new_cost = get_cost(temp_seq)
        
        if new_cost < current_cost:
            current_seq = temp_seq
            current_cost = new_cost
            if is_valid(current_seq):
                new_max = max(current_seq)
                if new_max < best_max:
                    best_max = new_max
                    best_seq = current_seq[:]
        else:
            delta = new_cost - current_cost
            if delta > 0:
                prob = np.exp(-delta / temperature)
                if random.random() < prob:
                    current_seq = temp_seq
                    current_cost = new_cost
                    if is_valid(current_seq):
                        new_max = max(current_seq)
                        if new_max < best_max:
                            best_max = new_max
                            best_seq = current_seq[:]
        
        temperature *= cooling_rate
        
    return best_seq
