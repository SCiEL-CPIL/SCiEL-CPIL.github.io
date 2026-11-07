import time
import random

def construct():
    n = 12
    m = 40
    deadline = time.time() + 85.0
    
    def is_comp(a, b):
        return (a & b) == a or (a & b) == b
        
    def calc_score(masks):
        score = 0
        for i in range(m):
            for j in range(i+1, m):
                if is_comp(masks[i], masks[j]):
                    score += 1
        return score

    all_masks = list(range(1 << n))
    
    # Initialization: Random
    current = random.sample(all_masks, m)
    current_set = set(current)
    current_score = calc_score(current)
    
    best_score = current_score
    best_solution = current[:]
    
    # Simulated Annealing
    temperature = 1.0
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline:
        i = random.randint(0, m - 1)
        old_mask = current[i]
        
        new_mask = random.choice(all_masks)
        if new_mask in current_set:
            continue
            
        delta = 0
        for x in current:
            if x == old_mask:
                continue
            if is_comp(new_mask, x):
                delta += 1
            if is_comp(old_mask, x):
                delta -= 1
                
        if delta > 0:
            # Always accept
            current[i] = new_mask
            current_set.remove(old_mask)
            current_set.add(new_mask)
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_solution = current[:]
        else:
            # Accept with probability exp(delta / temperature)
            # delta is negative
            if random.random() < pow(2.71828, delta / temperature):
                current[i] = new_mask
                current_set.remove(old_mask)
                current_set.add(new_mask)
                current_score += delta
                
        temperature *= cooling
        if temperature < 0.01:
            temperature = 1.0 # Restart cooling
            
    return [[b for b in range(n) if (mask >> b) & 1] for mask in best_solution]
