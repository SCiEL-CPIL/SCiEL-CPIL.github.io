import time
import random

N = 18
MIN_DISTANCE = 6
DEADLINE = 85  # seconds, safety margin before 90s kill

def _generate_offsets(min_dist):
    """Generate all bitmasks with popcount < min_dist."""
    offsets = []
    for mask in range(2**N):
        if mask.bit_count() < min_dist:
            offsets.append(mask)
    return offsets

def _greedy_bitset(deadline, offsets, strategy='lowest', seed=0):
    """Greedy construction using bitset representation."""
    random.seed(seed)
    available = (1 << (2**N)) - 1
    selected = []
    
    start_time = time.time()
    
    while available:
        if time.time() > deadline:
            break
        
        if strategy == 'lowest':
            lowbit = available & -available
            w = lowbit.bit_length() - 1
        elif strategy == 'highest':
            w = available.bit_length() - 1
        elif strategy == 'random':
            while True:
                pos = random.randint(0, 2**N - 1)
                if (available >> pos) & 1:
                    w = pos
                    break
        else:
            w = 0
        
        selected.append(w)
        
        conflict_mask = 0
        for offset in offsets:
            neighbor = w ^ offset
            conflict_mask |= (1 << neighbor)
        
        available &= ~conflict_mask
    
    return selected

def _local_improve_bitset(code, offsets, deadline):
    """Try to add more words using bitset."""
    if not code:
        return code
    
    selected_set = set(code)
    available = 0
    for w in range(2**N):
        if w not in selected_set:
            available |= (1 << w)
    
    improved = list(code)
    
    while available:
        if time.time() > deadline:
            break
        
        lowbit = available & -available
        w = lowbit.bit_length() - 1
        
        can_add = True
        for s in improved:
            if (w ^ s).bit_count() < MIN_DISTANCE:
                can_add = False
                break
        
        if can_add:
            improved.append(w)
            conflict_mask = 0
            for offset in offsets:
                neighbor = w ^ offset
                conflict_mask |= (1 << neighbor)
            available &= ~conflict_mask
        else:
            available &= ~(1 << w)
    
    return improved

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    offsets = _generate_offsets(MIN_DISTANCE)
    
    best_code = []
    best_len = 0
    
    strategies = ['lowest', 'highest', 'random', 'random', 'random']
    seeds = [0, 1, 42, 123, 999]
    
    for i in range(len(strategies)):
        if time.time() > deadline - 5:
            break
        
        code = _greedy_bitset(deadline - 2, offsets, strategy=strategies[i], seed=seeds[i])
        
        if len(code) > best_len:
            best_len = len(code)
            best_code = code
    
    if best_code and time.time() < deadline - 2:
        improved = _local_improve_bitset(best_code, offsets, deadline - 1)
        if len(improved) > best_len:
            best_code = improved
    
    return best_code
