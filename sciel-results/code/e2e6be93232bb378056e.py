import numpy as np
import random
import time

def construct():
    """
    Constructs a set of 32 distinct integers in 1..512 maximizing the number
    of distinct differences d > 0 such that a, a+d, a+2d are in the set.
    Uses Simulated Annealing with vectorized scoring.
    """
    
    def vectorized_score(mask):
        """
        mask: boolean array of size 513 (indices 0..512), where mask[i] is True if i is in the set.
        Returns the number of distinct differences d > 0 forming a 3-AP.
        """
        found_d = 0
        for d in range(1, 256):
            max_a = 512 - 2 * d
            if max_a < 1:
                break
            if np.any(mask[1:max_a+1] & mask[1+d:max_a+1+d] & mask[1+2*d:max_a+1+2*d]):
                found_d += 1
        return found_d

    random.seed(42)
    
    # Initial state: Random set
    current_set = set(random.sample(range(1, 513), 32))
    current_mask = np.zeros(513, dtype=bool)
    for x in current_set:
        current_mask[x] = True
        
    current_score = vectorized_score(current_mask)
    
    best_set = set(current_set)
    best_score = current_score
    
    # Simulated Annealing Parameters
    start_temp = 10.0
    end_temp = 0.001
    iterations = 50000
    start_time = time.time()
    time_limit = 110.0
    
    for i in range(iterations):
        if time.time() - start_time > time_limit:
            break
            
        temp = start_temp * (end_temp / start_temp) ** (i / iterations)
        
        # Generate a random move
        if len(current_set) == 0:
            break
        to_remove = random.choice(list(current_set))
        to_add = random.randint(1, 512)
        while to_add in current_set:
            to_add = random.randint(1, 512)
            
        new_set = set(current_set)
        new_set.remove(to_remove)
        new_set.add(to_add)
        
        new_mask = current_mask.copy()
        new_mask[to_remove] = False
        new_mask[to_add] = True
        
        new_score = vectorized_score(new_mask)
        
        delta = new_score - current_score
        if delta > 0 or (random.random() < np.exp(delta / temp)):
            current_set = new_set
            current_mask = new_mask
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
    
    return sorted(best_set)
