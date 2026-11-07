"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    n_elements = 12
    universe_size = 1 << n_elements  # 4096
    family_size = 40
    
    all_masks = list(range(universe_size))
    
    def count_comparable(family):
        count = 0
        n = len(family)
        for i in range(n):
            a = family[i]
            for j in range(i + 1, n):
                b = family[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count

    random.seed(42)
    current_family = random.sample(all_masks, family_size)
    current_score = count_comparable(current_family)
    best_family = current_family.copy()
    best_score = current_score
    
    start_time = time.time()
    timeout = 55
    temperature = 10.0
    cooling = 0.9999
    
    while time.time() - start_time < timeout:
        idx = random.randint(0, family_size - 1)
        old_set = current_family[idx]
        
        if random.random() < 0.7:
            bit_to_flip = random.randint(0, n_elements - 1)
            new_set = old_set ^ (1 << bit_to_flip)
        else:
            new_set = random.choice(all_masks)
            
        if new_set == old_set:
            continue
            
        is_dup = False
        for k in range(family_size):
            if k != idx and current_family[k] == new_set:
                is_dup = True
                break
        
        if is_dup:
            continue
            
        candidate_family = current_family.copy()
        candidate_family[idx] = new_set
        candidate_score = count_comparable(candidate_family)
        
        delta = candidate_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / temperature):
            current_family = candidate_family
            current_score = candidate_score
            
            if current_score > best_score:
                best_score = current_score
                best_family = current_family.copy()
                
        temperature *= cooling
        if temperature < 0.1:
            temperature = 0.1

    result = []
    for mask in best_family:
        subset = []
        for i in range(n_elements):
            if mask & (1 << i):
                subset.append(i)
        result.append(subset)
        
    return result
# EVOLVE-BLOCK-END
