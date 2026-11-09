import time
import random
import math

N = 12
MAX_TIME = 55.0  # Safety margin before 60s kill

def compute_threshold_fast(a_map, b_map):
    """
    Computes the shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    Optimized with precomputed transition tables for all 2^N subsets.
    """
    size = 1 << N
    start_mask = size - 1
    
    # Precompute transitions for all masks
    trans_a = [0] * size
    trans_b = [0] * size
    
    for mask in range(size):
        ma = 0
        mb = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            ma |= (1 << a_map[i])
            mb |= (1 << b_map[i])
            m ^= lsb
        trans_a[mask] = ma
        trans_b[mask] = mb
        
    # BFS
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
            
        next_a = trans_a[curr]
        next_b = trans_b[curr]
        
        if dist[next_a] == -1:
            dist[next_a] = d + 1
            queue.append(next_a)
            
        if dist[next_b] == -1:
            dist[next_b] = d + 1
            queue.append(next_b)
            
    return -1

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Start with Pawlik automaton
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0] + [i - 1 for i in range(1, N)]
    
    best_a = a_map.copy()
    best_b = b_map.copy()
    best_score = compute_threshold_fast(best_a, best_b)
    
    # If best_score is -1, we need a valid one. Pawlik is always valid for N>=2.
    if best_score == -1:
         # Fallback to a known simple synchronizing automaton if Pawlik fails (unlikely)
         a_map = [i+1 for i in range(N-1)] + [0]
         b_map = [0]*N
         best_a = a_map
         best_b = b_map
         best_score = compute_threshold_fast(best_a, best_b)

    # Simulated Annealing with Random Restarts
    # We will run multiple independent restarts until time runs out.
    
    # Parameters for SA
    T_init = 5.0
    T_min = 0.1
    alpha = 0.999
    
    while time.time() - start_time < MAX_TIME:
        # Initialize current state
        if random.random() < 0.1:
            # Occasionally start from best known to refine
            cur_a = best_a.copy()
            cur_b = best_b.copy()
            cur_score = best_score
        else:
            # Random restart
            # Bias towards permutations or simple structures? 
            # For now, uniform random is okay but might be slow to find sync.
            # Let's try: a_map is a permutation, b_map is random.
            # Or just fully random.
            cur_a = [random.randint(0, N-1) for _ in range(N)]
            cur_b = [random.randint(0, N-1) for _ in range(N)]
            cur_score = compute_threshold_fast(cur_a, cur_b)
            
            if cur_score == -1:
                continue
        
        # Local Search Phase (Simulated Annealing)
        # Limit steps per restart to avoid getting stuck in one local opt for too long
        max_steps = 200
        steps = 0
        T = T_init
        
        while steps < max_steps and time.time() - start_time < MAX_TIME:
            steps += 1
            prev_a = cur_a.copy()
            prev_b = cur_b.copy()
            
            # Mutate: change one transition
            if random.random() < 0.5:
                i = random.randint(0, N - 1)
                cur_a[i] = random.randint(0, N - 1)
            else:
                i = random.randint(0, N - 1)
                cur_b[i] = random.randint(0, N - 1)
                
            new_score = compute_threshold_fast(cur_a, cur_b)
            
            if new_score == -1:
                # Reject non-synchronizing
                cur_a = prev_a
                cur_b = prev_b
                continue
            
            # Accept if better, or with probability exp(delta/T) if worse
            if new_score > cur_score:
                cur_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_a = cur_a.copy()
                    best_b = cur_b.copy()
            else:
                # Metropolis criterion
                delta = new_score - cur_score
                if delta < 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        # Accept
                        cur_score = new_score
                    else:
                        # Revert
                        cur_a = prev_a
                        cur_b = prev_b
            
            T = max(T_min, T * alpha)
    
    return [best_a, best_b]
