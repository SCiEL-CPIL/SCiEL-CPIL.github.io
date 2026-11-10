import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try n=34 first
    n = 34
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Map triple to index
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # Generate 4-sets and their triple masks
    four_sets = list(combinations(range(n), 4))
    masks_4 = []
    affected_4sets = [[] for _ in range(num_triples)]
    for i, s in enumerate(four_sets):
        # Get the 4 triples from this 4-set
        # Triples are (a,b,c), (a,b,d), (a,c,d), (b,c,d)
        a, b, c, d = s
        t1 = triple_index[(a,b,c)]
        t2 = triple_index[(a,b,d)]
        t3 = triple_index[(a,c,d)]
        t4 = triple_index[(b,c,d)]
        mask = (1 << t1) | (1 << t2) | (1 << t3) | (1 << t4)
        masks_4.append(mask)
        # Update affected lists
        affected_4sets[t1].append(i)
        affected_4sets[t2].append(i)
        affected_4sets[t3].append(i)
        affected_4sets[t4].append(i)
        
    # Generate 5-sets and their triple masks
    five_sets = list(combinations(range(n), 5))
    masks_5 = []
    affected_5sets = [[] for _ in range(num_triples)]
    for i, s in enumerate(five_sets):
        # Get the 10 triples from this 5-set
        # All combinations of 3 from the 5
        ts = [triple_index[t] for t in combinations(s, 3)]
        mask = 0
        for t_idx in ts:
            mask |= (1 << t_idx)
            affected_5sets[t_idx].append(i)
        masks_5.append(mask)
        
    # Initialize colors
    # Seed: algebraic
    red_bits = 0
    blue_bits = 0
    colors = [0] * num_triples
    for i, (a, b, c) in enumerate(triples):
        if (a + b + c) % 3 == 0:
            colors[i] = 0
            red_bits |= (1 << i)
        else:
            colors[i] = 1
            blue_bits |= (1 << i)
            
    # Calculate initial violations
    def count_violations(red_bits, blue_bits):
        v4 = 0
        for mask in masks_4:
            if (red_bits & mask) == mask:
                v4 += 1
        v5 = 0
        for mask in masks_5:
            if (blue_bits & mask) == mask:
                v5 += 1
        return v4 + v5
        
    current_violations = count_violations(red_bits, blue_bits)
    best_violations = current_violations
    best_red_bits = red_bits
    best_blue_bits = blue_bits
    best_colors = colors[:]
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Number of steps per temperature
    # We have ~110 seconds.
    # Each step is fast.
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random triple to flip
        t = random.randint(0, num_triples - 1)
        
        # Calculate change in violations
        # We need to check affected 4-sets and 5-sets
        
        # Save old bits
        old_red = red_bits
        old_blue = blue_bits
        
        # Flip bit t
        if colors[t] == 0:
            # Red to Blue
            red_bits = red_bits & ~(1 << t)
            blue_bits = blue_bits | (1 << t)
            colors[t] = 1
        else:
            # Blue to Red
            blue_bits = blue_bits & ~(1 << t)
            red_bits = red_bits | (1 << t)
            colors[t] = 0
            
        # Calculate delta V
        delta_v = 0
        
        # Check affected 4-sets
        # For each 4-set in affected_4sets[t], check if it was a violation and is not now, or vice versa
        for idx in affected_4sets[t]:
            mask = masks_4[idx]
            was_viol = ((old_red & mask) == mask)
            is_viol = ((red_bits & mask) == mask)
            if was_viol and not is_viol:
                delta_v -= 1
            elif not was_viol and is_viol:
                delta_v += 1
                
        # Check affected 5-sets
        for idx in affected_5sets[t]:
            mask = masks_5[idx]
            was_viol = ((old_blue & mask) == mask)
            is_viol = ((blue_bits & mask) == mask)
            if was_viol and not is_viol:
                delta_v -= 1
            elif not was_viol and is_viol:
                delta_v += 1
                
        new_violations = current_violations + delta_v
        
        # Accept/Reject
        if delta_v < 0 or random.random() < pow(2.718281828, -delta_v / T):
            current_violations = new_violations
            if current_violations < best_violations:
                best_violations = current_violations
                best_red_bits = red_bits
                best_blue_bits = blue_bits
                best_colors = colors[:]
                if current_violations == 0:
                    break
        else:
            # Revert
            red_bits = old_red
            blue_bits = old_blue
            colors[t] = 1 - colors[t] # Revert flip
            
        T *= cooling_rate
        
    # If we found a valid solution, return it
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
    else:
        # If no valid solution for n=34, try n=33? 
        # Or just return the best found (which is invalid, score 0).
        # To be safe, we should try to find a valid solution for a smaller n.
        # But n=34 is known to be valid. SA should find it.
        # If it fails, it's a failure.
        # We can try a quick restart with n=33 if we have time.
        
        if time.time() < deadline - 10:
             # Try n=33
             n = 33
             triples = list(combinations(range(n), 3))
             num_triples = len(triples)
             triple_index = {t: i for i, t in enumerate(triples)}
             
             four_sets = list(combinations(range(n), 4))
             masks_4 = []
             affected_4sets = [[] for _ in range(num_triples)]
             for i, s in enumerate(four_sets):
                 a, b, c, d = s
                 t1 = triple_index[(a,b,c)]
                 t2 = triple_index[(a,b,d)]
                 t3 = triple_index[(a,c,d)]
                 t4 = triple_index[(b,c,d)]
                 mask = (1 << t1) | (1 << t2) | (1 << t3) | (1 << t4)
                 masks_4.append(mask)
                 affected_4sets[t1].append(i)
                 affected_4sets[t2].append(i)
                 affected_4sets[t3].append(i)
                 affected_4sets[t4].append(i)
                 
             five_sets = list(combinations(range(n), 5))
             masks_5 = []
             affected_5sets = [[] for _ in range(num_triples)]
             for i, s in enumerate(five_sets):
                 ts = [triple_index[t] for t in combinations(s, 3)]
                 mask = 0
                 for t_idx in ts:
                     mask |= (1 << t_idx)
                     affected_5sets[t_idx].append(i)
                 masks_5.append(mask)
                 
             red_bits = 0
             blue_bits = 0
             colors = [0] * num_triples
             for i, (a, b, c) in enumerate(triples):
                 if (a + b + c) % 3 == 0:
                     colors[i] = 0
                     red_bits |= (1 << i)
                 else:
                     colors[i] = 1
                     blue_bits |= (1 << i)
                     
             current_violations = count_violations(red_bits, blue_bits)
             best_violations = current_violations
             best_red_bits = red_bits
             best_blue_bits = blue_bits
             best_colors = colors[:]
             
             T = 1.0
             while time.time() < deadline - 2:
                 if T < min_T:
                     break
                 t = random.randint(0, num_triples - 1)
                 old_red = red_bits
                 old_blue = blue_bits
                 
                 if colors[t] == 0:
                     red_bits = red_bits & ~(1 << t)
                     blue_bits = blue_bits | (1 << t)
                     colors[t] = 1
                 else:
                     blue_bits = blue_bits & ~(1 << t)
                     red_bits = red_bits | (1 << t)
                     colors[t] = 0
                     
                 delta_v = 0
                 for idx in affected_4sets[t]:
                     mask = masks_4[idx]
                     was_viol = ((old_red & mask) == mask)
                     is_viol = ((red_bits & mask) == mask)
                     if was_viol and not is_viol: delta_v -= 1
                     elif not was_viol and is_viol: delta_v += 1
                     
                 for idx in affected_5sets[t]:
                     mask = masks_5[idx]
                     was_viol = ((old_blue & mask) == mask)
                     is_viol = ((blue_bits & mask) == mask)
                     if was_viol and not is_viol: delta_v -= 1
                     elif not was_viol and is_viol: delta_v += 1
                     
                 new_violations = current_violations + delta_v
                 if delta_v < 0 or random.random() < pow(2.718281828, -delta_v / T):
                     current_violations = new_violations
                     if current_violations < best_violations:
                         best_violations = current_violations
                         best_red_bits = red_bits
                         best_blue_bits = blue_bits
                         best_colors = colors[:]
                         if current_violations == 0:
                             break
                 else:
                     red_bits = old_red
                     blue_bits = old_blue
                     colors[t] = 1 - colors[t]
                     
                 T *= cooling_rate
                 
             if best_violations == 0:
                 return {'n': n, 'colors': best_colors}
                 
        # If all else fails, return the best n=34 attempt (likely invalid, score 0)
        # Or a known valid small n?
        # The seed n=9 is valid.
        # Let's return the n=9 seed if we fail completely?
        # No, the task says "Maximize n".
        # If we fail n=34, we should return n=9 to get score 1.0?
        # "1.0 equals the seed program".
        # So returning n=9 gives 1.0.
        # Returning invalid n=34 gives 0.
        # So we should fallback to n=9 if n=34 fails.
        
        n = 9
        triples = list(combinations(range(n), 3))
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]}

# EVOLVE-BLOCK-END
