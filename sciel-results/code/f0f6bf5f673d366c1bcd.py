import random
import time

def _score_mask(mask):
    """Count distinct d > 0 such that exists a: a, a+d, a+2d all in mask."""
    count = 0
    for d in range(1, 257):
        if (mask & (mask >> d) & (mask >> 2 * d)) != 0:
            count += 1
    return count

def _list_to_mask(lst):
    mask = 0
    for x in lst:
        mask |= (1 << (x - 1))
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    random.seed(123)
    
    # Initial set
    cur_list = random.sample(range(1, 513), 32)
    cur_mask = _list_to_mask(cur_list)
    cur_score = _score_mask(cur_mask)
    best_score = cur_score
    best_list = cur_list[:]
    
    T = 5.0
    T_min = 0.01
    cooling = 0.9998
    
    iters = 0
    while time.time() < deadline:
        if iters % 200 == 0 and time.time() > deadline:
            break
        
        # Move: replace one element with a random new one
        idx = random.randint(0, 31)
        old_val = cur_list[idx]
        new_val = random.randint(1, 512)
        
        # Skip if same or already in set
        if new_val == old_val:
            iters += 1
            continue
        if cur_mask & (1 << (new_val - 1)):
            iters += 1
            continue
        
        new_mask = cur_mask & ~(1 << (old_val - 1))
        new_mask |= (1 << (new_val - 1))
        
        new_score = _score_mask(new_mask)
        delta = new_score - cur_score
        
        accept = False
        if delta >= 0:
            accept = True
        else:
            # Metropolis criterion
            ratio = delta / max(T, 1e-10)
            if random.random() < (2.718 ** ratio):
                accept = True
        
        if accept:
            cur_list[idx] = new_val
            cur_mask = new_mask
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_list = cur_list[:]
        
        T *= cooling
        if T < T_min:
            T = T_min
        iters += 1
    
    return best_list
