import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 85.0  # 5s buffer before 90s kill
    
    # 1. Generate initial valid sequence using greedy
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    best_seq = values[:]
    best_len = len(best_seq)
    
    # Helper to count product collisions
    def count_collisions(seq):
        if not seq:
            return 0
        prods = []
        for i in range(len(seq)):
            p = 1
            for j in range(i, len(seq)):
                p *= seq[j]
                prods.append(p)
        return len(prods) - len(set(prods))
    
    current_seq = best_seq[:]
    current_collisions = 0  # Greedy result is valid
    
    # SA Parameters
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    restart_T = 5.0
    
    all_numbers = list(range(2, 129))
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = restart_T
            # Optional: perturb current state or reset to best?
            # Resetting to best might be too conservative. 
            # Let's just continue with higher T.
            
        # Generate a neighbor
        move_type = random.choice(['add', 'remove', 'swap'])
        new_seq = current_seq[:]
        
        if move_type == 'add':
            candidates = [x for x in all_numbers if x not in new_seq]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_seq.append(x)
            new_seq.sort()
            
        elif move_type == 'remove':
            if not new_seq:
                continue
            idx = random.randint(0, len(new_seq) - 1)
            new_seq.pop(idx)
            
        else: # swap
            if not new_seq:
                continue
            candidates = [x for x in all_numbers if x not in new_seq]
            if not candidates:
                continue
            idx = random.randint(0, len(new_seq) - 1)
            new_val = random.choice(candidates)
            new_seq[idx] = new_val
            new_seq.sort()
        
        # Evaluate new_seq
        new_collisions = count_collisions(new_seq)
        new_len = len(new_seq)
        
        # Energy: Minimize (Collisions * 1000 - Length * 100)
        # We want low collisions, high length.
        curr_energy = (current_collisions * 1000) - (len(current_seq) * 100)
        new_energy = (new_collisions * 1000) - (new_len * 100)
        
        delta_E = new_energy - curr_energy
        
        # Accept if better, or with probability exp(-delta_E/T)
        accept = False
        if delta_E < 0:
            accept = True
        elif T > 0:
            # Avoid overflow/underflow issues with exp
            if delta_E / T < 700: # exp(-700) is ~0
                prob = pow(2.718281828, -delta_E / T)
                if random.random() < prob:
                    accept = True
        
        if accept:
            current_seq = new_seq
            current_collisions = new_collisions
            
            # Update best valid sequence
            if new_collisions == 0 and new_len > best_len:
                best_seq = new_seq
                best_len = new_len
                
        T *= cooling
        
    return best_seq
# EVOLVE-BLOCK-END
