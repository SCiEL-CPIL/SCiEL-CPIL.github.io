import time
import random

def construct():
    N = 32
    R = 512
    MAX_D = 255
    
    # Initialize random set
    selected = set(random.sample(range(1, R + 1), N))
    sel_list = list(selected)
    bits = 0
    for x in selected:
        bits |= (1 << x)
    
    def score(bits):
        count = 0
        for d in range(1, MAX_D + 1):
            if (bits & (bits >> d) & (bits >> 2 * d)):
                count += 1
        return count

    current_score = score(bits)
    best_score = current_score
    best_selected = set(selected)
    best_bits = bits
    
    start_time = time.time()
    deadline = 140.0 # 10s buffer
    
    T = 1.0
    cooling = 0.99995
    
    all_values = list(range(1, R + 1))
    
    while time.time() - start_time < deadline:
        # Pick a random element to remove
        idx = random.randint(0, N - 1)
        x = sel_list[idx]
        
        # Pick a random element to add
        y = random.choice(all_values)
        while y in selected:
            y = random.choice(all_values)
            
        # New bits
        new_bits = bits ^ (1 << x) ^ (1 << y)
        new_score = score(new_bits)
        
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < pow(2.71828, delta / T):
            # Accept
            selected.remove(x)
            selected.add(y)
            sel_list[idx] = y
            bits = new_bits
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_selected = set(selected)
                best_bits = bits
        
        T *= cooling
        if T < 0.001:
            T = 0.001
            
    return sorted(best_selected)
