import random
import time

def score_set(S):
    S = set(S)
    diffs = set()
    for a in S:
        for d in range(1, 512):
            if a + d in S and a + 2*d in S:
                diffs.add(d)
    return len(diffs)

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Initial guess: 4 blocks of 8
    # L1=1, L2=100, L3=300, L4=500
    L = [1, 100, 300, 500]
    S = []
    for l in L:
        S.extend(range(l, l+8))
    
    best_S = S[:]
    best_score = score_set(best_S)
    
    current_S = best_S[:]
    current_score = best_score
    
    T = 10.0
    T_min = 0.1
    
    while time.time() - start_time < timeout:
        # Move: Pick a block and shift it
        block_idx = random.randint(0, 3)
        # Get indices of elements in this block
        # Since blocks are contiguous, we can just pick a new start
        # But we need to maintain distinctness.
        # Easier: Pick one element and move it to a random position.
        idx = random.randint(0, 31)
        old_val = current_S[idx]
        
        # Try to move it to a new random position
        new_val = random.randint(1, 512)
        
        # Check if new_val is already in S (excluding old_val)
        if new_val == old_val:
            continue
        if new_val in current_S:
            continue
            
        current_S[idx] = new_val
        new_score = score_set(current_S)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_S = current_S[:]
        else:
            # Accept with probability
            delta = new_score - current_score
            if random.random() < pow(2.71828, delta / T):
                current_score = new_score
            else:
                current_S[idx] = old_val
        
        T *= 0.99
        
    return sorted(best_S)
