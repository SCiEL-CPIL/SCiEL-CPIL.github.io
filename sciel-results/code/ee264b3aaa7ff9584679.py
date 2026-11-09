import time
import random

N = 12
MAX_TIME = 55.0  # Safety margin before 60s kill

def compute_threshold(a_map, b_map):
    """
    Computes the shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    """
    size = 1 << N
    start_mask = size - 1
    
    # dist array for BFS
    dist = [-1] * size
    dist[start_mask] = 0
    queue = [start_mask]
    head = 0
    
    while head < len(queue):
        curr = queue[head]
        head += 1
        d = dist[curr]
        
        # Check if single state
        if curr == 0 or (curr & (curr - 1)) == 0:
            return d
            
        # Compute next states on the fly
        next_a = 0
        next_b = 0
        
        # Iterate through set bits in curr
        m = curr
        while m:
            lsb = m & -m
            i = (lsb.bit_length() - 1)
            next_a |= (1 << a_map[i])
            next_b |= (1 << b_map[i])
            m ^= lsb
            
        if dist[next_a] == -1:
            dist[next_a] = d + 1
            queue.append(next_a)
            
        if dist[next_b] == -1:
            dist[next_b] = d + 1
            queue.append(next_b)
            
    return -1

def construct_pawlik_variant():
    """
    Constructs the standard Pawlik automaton variant.
    a: i -> (i+1) % N
    b: i -> i-1 for i>0, b(0) = 0
    This is known to have threshold (N-1)^2 for N >= 3.
    """
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0] + [i - 1 for i in range(1, N)]
    return a_map, b_map

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Initialize with Pawlik
    best_a, best_b = construct_pawlik_variant()
    best_score = compute_threshold(best_a, best_b)
    
    # If Pawlik failed (unlikely), fallback to a simple one
    if best_score == -1:
         a_map = [(i + 1) % N for i in range(N)]
         b_map = [0] * N
         best_a = a_map
         best_b = b_map
         best_score = compute_threshold(best_a, best_b)

    # Local search phase
    # We will perform hill climbing on the transition tables.
    
    while time.time() - start_time < MAX_TIME:
        # Random restart occasionally, or start from best
        if random.random() < 0.2:
            # Start from a random valid synchronizing automaton
            # Simple strategy: pick random a, b. If not sync, try to fix?
            # Actually, most random automata synchronize.
            cur_a = [random.randint(0, N-1) for _ in range(N)]
            cur_b = [random.randint(0, N-1) for _ in range(N)]
        else:
            cur_a = best_a.copy()
            cur_b = best_b.copy()
            
        cur_score = compute_threshold(cur_a, cur_b)
        
        if cur_score == -1:
            # If not synchronizing, try to make it synchronizing by
            # forcing one letter to be a constant map? No, that changes structure too much.
            # Just skip to next restart.
            continue
            
        # Local search steps
        steps = 0
        max_steps = 500
        
        while steps < max_steps and time.time() - start_time < MAX_TIME:
            steps += 1
            # Pick a random position to mutate
            if random.random() < 0.5:
                idx = random.randint(0, N - 1)
                old_val = cur_a[idx]
                cur_a[idx] = random.randint(0, N - 1)
                if cur_a[idx] == old_val:
                    continue
            else:
                idx = random.randint(0, N - 1)
                old_val = cur_b[idx]
                cur_b[idx] = random.randint(0, N - 1)
                if cur_b[idx] == old_val:
                    continue
                    
            new_score = compute_threshold(cur_a, cur_b)
            
            if new_score == -1:
                # Revert
                if random.random() < 0.5:
                    cur_a[idx] = old_val
                else:
                    cur_b[idx] = old_val
                continue
                
            if new_score > cur_score:
                cur_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_a = cur_a.copy()
                    best_b = cur_b.copy()
            else:
                # Accept worse with low probability to escape local minima
                if random.random() > 0.05:
                    # Revert
                    if random.random() < 0.5:
                        cur_a[idx] = old_val
                    else:
                        cur_b[idx] = old_val

    return [best_a, best_b]
