import random
import time
import collections

N = 12
FULL_MASK = (1 << N) - 1

def calc_threshold_fast(trans):
    """
    trans: list of two lists of destination states.
    Uses bitmask BFS.
    """
    # Precompute transition masks? No, we need to apply letter to each state in subset.
    # For a subset mask M, next_mask = OR of (1 << trans[l][s]) for s in M.
    
    start_mask = FULL_MASK
    if start_mask == 1: # Single state
        return 0
        
    visited = {start_mask: 0}
    queue = collections.deque([start_mask])
    
    # Precompute single state transitions for each letter
    # t0[s] = 1 << trans[0][s]
    # t1[s] = 1 << trans[1][s]
    t0 = [1 << trans[0][s] for s in range(N)]
    t1 = [1 << trans[1][s] for s in range(N)]
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        # Apply letter 0
        # next_mask = OR of t0[s] for all s in current
        # We can iterate bits
        m = current
        next_mask = 0
        while m:
            lsb = m & -m
            s = lsb.bit_length() - 1
            next_mask |= t0[s]
            m ^= lsb
            
        if next_mask == 1:
            return dist + 1
        if next_mask not in visited:
            visited[next_mask] = dist + 1
            queue.append(next_mask)
            
        # Apply letter 1
        m = current
        next_mask = 0
        while m:
            lsb = m & -m
            s = lsb.bit_length() - 1
            next_mask |= t1[s]
            m ^= lsb
            
        if next_mask == 1:
            return dist + 1
        if next_mask not in visited:
            visited[next_mask] = dist + 1
            queue.append(next_mask)
            
    return -1

def construct():
    start_time = time.time()
    timeout = 85.0
    
    # Initialization
    best_trans = None
    best_score = -1
    
    # Try to find a good initial automaton
    # Use a "slide" automaton as a strong seed
    # a: i -> (i+1) % N
    # b: i -> 0
    seed_a = [(i + 1) % N for i in range(N)]
    seed_b = [0] * N
    seed_trans = [seed_a, seed_b]
    seed_score = calc_threshold_fast(seed_trans)
    
    if seed_score > best_score:
        best_score = seed_score
        best_trans = [seed_a[:], seed_b[:]]
        
    # Random initialization
    for _ in range(100):
        if time.time() - start_time > 5.0:
            break
        trans = [[random.randint(0, N-1) for _ in range(N)] for _ in range(2)]
        score = calc_threshold_fast(trans)
        if score > best_score:
            best_score = score
            best_trans = [row[:] for row in trans]
            
    if best_trans is None:
        return seed_trans

    current_trans = [row[:] for row in best_trans]
    current_score = best_score
    
    # Hill Climbing
    while time.time() - start_time < timeout:
        letter = random.randint(0, 1)
        state = random.randint(0, N-1)
        old_val = current_trans[letter][state]
        new_val = random.randint(0, N-1)
        
        if new_val == old_val:
            continue
            
        current_trans[letter][state] = new_val
        
        new_score = calc_threshold_fast(current_trans)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_trans = [row[:] for row in current_trans]
        else:
            current_trans[letter][state] = old_val

    return best_trans
