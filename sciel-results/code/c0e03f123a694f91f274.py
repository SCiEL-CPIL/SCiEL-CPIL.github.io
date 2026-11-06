import time
import random
import numpy as np
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _compute_threshold_fast(trans0, trans1):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Optimized for speed using precomputed transition tables for all subsets.
    Returns -1 if not synchronizing.
    """
    # Precompute transition masks for each state and letter
    # t0[i] = 1 << dest0[i]
    t0 = [1 << trans0[i] for i in range(N)]
    t1 = [1 << trans1[i] for i in range(N)]
    
    # Precompute next subset for all 2^N subsets for both letters
    # This is the key optimization: O(2^N * N) precomputation, then O(1) BFS steps
    # 4096 subsets * 12 states * 2 letters = ~100k operations, very fast
    
    next_subsets = np.zeros((1 << N, 2), dtype=np.int32)
    
    # We can vectorize this a bit or just loop. Looping 4096 times is fine.
    for s in range(1 << N):
        # Compute next for letter 0
        res0 = 0
        # Use bit manipulation to iterate set bits
        temp_s = s
        while temp_s:
            lsb = temp_s & -temp_s
            idx = (lsb.bit_length() - 1)
            res0 |= t0[idx]
            temp_s ^= lsb
        next_subsets[s, 0] = res0
        
        # Compute next for letter 1
        res1 = 0
        temp_s = s
        while temp_s:
            lsb = temp_s & -temp_s
            idx = (lsb.bit_length() - 1)
            res1 |= t1[idx]
            temp_s ^= lsb
        next_subsets[s, 1] = res1

    # BFS
    visited = np.full(1 << N, -1, dtype=np.int32)
    queue = deque()
    
    start = FULL_MASK
    visited[start] = 0
    queue.append(start)
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        # Check if singleton
        if current != 0 and (current & (current - 1)) == 0:
            return dist
            
        for letter in range(2):
            nxt = next_subsets[current, letter]
            if visited[nxt] == -1:
                visited[nxt] = dist + 1
                queue.append(nxt)
                
    return -1

def _generate_cerny_seed():
    """
    Generates a standard Cerny automaton which has threshold N-1.
    """
    trans0 = [(i + 1) % N for i in range(N)]
    trans1 = [0] * N
    for i in range(1, N):
        trans1[i] = i - 1
    return trans0, trans1

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    rng = random.Random(12345)
    
    # Initialize with Cerny seed
    current_0, current_1 = _generate_cerny_seed()
    current_score = _compute_threshold_fast(current_0, current_1)
    
    best_0, best_1 = current_0[:], current_1[:]
    best_score = current_score
    
    # Simulated Annealing parameters
    temperature = 5.0  # Start high to allow bad moves
    cooling_rate = 0.999
    min_temp = 0.1
    
    stagnation_counter = 0
    
    while time.time() < deadline:
        # Create a copy for mutation
        mut_0 = current_0[:]
        mut_1 = current_1[:]
        
        # Mutation Strategy: 
        # 1. Pick a random state and letter
        # 2. Sometimes pick a "smart" mutation: e.g., change a transition to a random other state, 
        #    or specifically try to break a cycle or create a sink.
        
        letter = rng.randint(0, 1)
        state = rng.randint(0, N - 1)
        
        if letter == 0:
            old_val = mut_0[state]
            new_val = rng.randint(0, N - 1)
            if new_val == old_val:
                continue
            mut_0[state] = new_val
            score = _compute_threshold_fast(mut_0, mut_1)
        else:
            old_val = mut_1[state]
            new_val = rng.randint(0, N - 1)
            if new_val == old_val:
                continue
            mut_1[state] = new_val
            score = _compute_threshold_fast(mut_0, mut_1)
            
        # Acceptance Criterion (Simulated Annealing)
        if score > current_score:
            # Always accept improvement
            current_0, current_1 = mut_0, mut_1
            current_score = score
            stagnation_counter = 0
            
            if score > best_score:
                best_score = score
                best_0, best_1 = current_0[:], current_1[:]
                # Optional: cool down faster when we find a new best
                # temperature *= 0.95 
        else:
            # Accept worse move with probability exp((new_score - old_score) / temperature)
            # Note: score can be -1 (non-synchronizing). 
            # If score is -1, we generally don't want to accept it unless we are stuck and want to explore.
            # But -1 is very bad. Let's treat -1 as a very large negative delta.
            delta = score - current_score
            if delta < 0:
                # If the new state is non-synchronizing (-1), probability should be very low
                # unless temperature is high.
                # Let's map -1 to a large negative value for the exponent to make it rare.
                if score == -1:
                    delta = -100.0 
                
                if temperature > min_temp:
                    prob = np.exp(delta / temperature)
                    if rng.random() < prob:
                        current_0, current_1 = mut_0, mut_1
                        current_score = score
                        # Do NOT reset stagnation counter on worse moves, but do cool down
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1
            
            # Cool down
            temperature *= cooling_rate
            
            # If we are stuck in a local minimum for too long, restart or kick
            if stagnation_counter > 500:
                # Restart from best
                current_0, current_1 = best_0[:], best_1[:]
                current_score = best_score
                stagnation_counter = 0
                temperature = max(temperature, 2.0) # Reheat slightly
                
                # Occasionally generate a completely new random structure
                if rng.random() < 0.2:
                    # Random permutation for 0
                    t0 = list(range(N))
                    rng.shuffle(t0)
                    # Random function for 1
                    t1 = [rng.randint(0, N - 1) for _ in range(N)]
                    rs_score = _compute_threshold_fast(t0, t1)
                    if rs_score > 0: # Must be synchronizing
                        current_0, current_1 = t0, t1
                        current_score = rs_score
                        if current_score > best_score:
                            best_score = current_score
                            best_0, best_1 = current_0[:], current_1[:]

    return [best_0, best_1]

# EVOLVE-BLOCK-END
