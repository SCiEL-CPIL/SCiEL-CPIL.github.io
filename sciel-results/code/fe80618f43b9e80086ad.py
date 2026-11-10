import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    max_mask = (1 << n) - 1
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    def calc_score(masks):
        score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if is_comparable(masks[i], masks[j]):
                    score += 1
        return score

    # Initialization: 
    # Create a base chain
    masks = []
    cur = 0
    for i in range(n + 1):
        masks.append(cur)
        cur |= (1 << i)
    # Fill rest randomly
    used = set(masks)
    while len(masks) < m:
        r = random.randint(0, max_mask)
        if r not in used:
            masks.append(r)
            used.add(r)
            
    # Optimize
    current_score = calc_score(masks)
    best_score = current_score
    best_masks = masks[:]
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    random.seed(42)
    
    # Precompute comparability for speed
    # For each mask, precompute which other masks it's comparable with
    # But since masks change, we'll compute delta on the fly
    
    while time.time() < deadline and T > min_T:
        # Pick random index
        i = random.randint(0, m - 1)
        old_mask = masks[i]
        
        # Generate candidate
        if random.random() < 0.8:
            new_mask = old_mask
            # Flip 1 to 2 bits (smaller mutations for more local search)
            for _ in range(random.randint(1, 2)):
                bit = random.randint(0, n - 1)
                new_mask ^= (1 << bit)
        else:
            # Random restart - pick a mask near existing ones
            base = random.choice(masks)
            new_mask = base
            for _ in range(random.randint(1, 3)):
                bit = random.randint(0, n - 1)
                new_mask ^= (1 << bit)
            
        if new_mask == old_mask:
            continue
            
        if new_mask in used:
            continue
             
        # Calculate Delta
        delta = 0
        for j in range(m):
            if j == i:
                continue
            m_j = masks[j]
            old_c = 1 if is_comparable(old_mask, m_j) else 0
            new_c = 1 if is_comparable(new_mask, m_j) else 0
            delta += new_c - old_c
            
        # Accept?
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept
            used.remove(old_mask)
            used.add(new_mask)
            masks[i] = new_mask
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_masks = masks[:]
            
            T *= cooling
    
    # Final polish: hill climbing on best_masks
    polish_deadline = deadline - 2.0
    polish_masks = best_masks[:]
    polish_score = best_score
    polish_used = set(polish_masks)
    
    improved = True
    while improved and time.time() < polish_deadline:
        improved = False
        for i in range(m):
            old_mask = polish_masks[i]
            best_new = old_mask
            best_delta = 0
            # Try all single-bit flips
            for bit in range(n):
                new_mask = old_mask ^ (1 << bit)
                if new_mask in polish_used:
                    continue
                delta = 0
                for j in range(m):
                    if j == i:
                        continue
                    m_j = polish_masks[j]
                    old_c = 1 if is_comparable(old_mask, m_j) else 0
                    new_c = 1 if is_comparable(new_mask, m_j) else 0
                    delta += new_c - old_c
                if delta > best_delta:
                    best_delta = delta
                    best_new = new_mask
            if best_delta > 0:
                polish_used.remove(old_mask)
                polish_used.add(best_new)
                polish_masks[i] = best_new
                polish_score += best_delta
                improved = True
                if polish_score > best_score:
                    best_score = polish_score
                    best_masks = polish_masks[:]
    
    # Final conversion
    result = []
    for mask in best_masks:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
