"""
Constructs a 12-state binary deterministic automaton maximizing shortest reset word length.
Uses Steepest Ascent Hill Climbing with Restarts to escape local optima.
"""
import time
import random
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

# EVOLVE-BLOCK-START

def _threshold_fast(a, b):
    """
    Compute the exact reset threshold using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    """
    visited = set()
    queue = deque()
    
    visited.add(FULL_MASK)
    queue.append((FULL_MASK, 0))
    
    while queue:
        mask, depth = queue.popleft()
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return depth
        
        # Apply 'a'
        new_mask_a = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_a |= 1 << a[i]
        
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append((new_mask_a, depth + 1))
            
        # Apply 'b'
        new_mask_b = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_b |= 1 << b[i]
            
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append((new_mask_b, depth + 1))
            
    return -1

def _random_automaton():
    return [random.randint(0, N-1) for _ in range(N)], [random.randint(0, N-1) for _ in range(N)]

def _hill_climb(a, b, deadline):
    """
    Perform steepest ascent hill climbing starting from (a, b).
    Returns the best (a, b, score) found during this climb.
    """
    cur_a = a[:]
    cur_b = b[:]
    cur_score = _threshold_fast(cur_a, cur_b)
    
    if cur_score < 0:
        return cur_a, cur_b, cur_score
        
    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score
    
    stuck_count = 0
    max_stuck = 500 # If we don't improve in 500 random probes, we consider it stuck
    
    while time.time() < deadline:
        if stuck_count > max_stuck:
            break
            
        # Pick a random state and input
        state = random.randint(0, N - 1)
        input_idx = random.randint(0, 1) # 0 for 'a', 1 for 'b'
        
        old_dest = cur_a[state] if input_idx == 0 else cur_b[state]
        
        best_new_dest = old_dest
        best_new_score = cur_score
        
        # Try all possible destinations for this transition
        for dest in range(N):
            if dest == old_dest:
                continue
                
            if input_idx == 0:
                cur_a[state] = dest
            else:
                cur_b[state] = dest
                
            new_score = _threshold_fast(cur_a, cur_b)
            
            if new_score > best_new_score:
                best_new_score = new_score
                best_new_dest = dest
                
            # Restore original to test next destination
            if input_idx == 0:
                cur_a[state] = old_dest
            else:
                cur_b[state] = old_dest
        
        if best_new_score > cur_score:
            # Apply the best move
            if input_idx == 0:
                cur_a[state] = best_new_dest
            else:
                cur_b[state] = best_new_dest
            cur_score = best_new_score
            stuck_count = 0
            
            if cur_score > best_score:
                best_score = cur_score
                best_a = cur_a[:]
                best_b = cur_b[:]
        else:
            stuck_count += 1
            
    return best_a, best_b, best_score

def construct():
    """
    Construct a 12-state binary automaton maximizing reset threshold.
    Uses Hill Climbing with Restarts.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_a = None
    best_b = None
    best_score = -1
    
    # Initial random starts
    num_initials = 5
    for i in range(num_initials):
        a_init, b_init = _random_automaton()
        score = _threshold_fast(a_init, b_init)
        if score > best_score:
            best_score = score
            best_a = a_init
            best_b = b_init
            
    # If we didn't find a synchronizing one, force one?
    # Usually random ones sync. If not, we can use a known one.
    if best_score < 0:
        # Known synchronizing automaton (Cherny-like)
        a_known = [(i + 1) % N for i in range(N)]
        b_known = [0] * N
        for i in range(1, N):
            b_known[i] = (i + 1) % N
        best_a = a_known
        best_b = b_known
        best_score = _threshold_fast(best_a, best_b)
        
    # Main loop: Hill Climb from best, then restart if stuck or time allows
    while time.time() < deadline:
        # Hill climb from current best
        a_climb, b_climb, score_climb = _hill_climb(best_a, best_b, deadline)
        
        if score_climb > best_score:
            best_score = score_climb
            best_a = a_climb
            best_b = b_climb
        else:
            # No improvement from best, try a random restart
            # Perturb the best solution or start fresh?
            # Let's try a random restart to explore new basins
            if random.random() < 0.5:
                a_init, b_init = _random_automaton()
            else:
                # Perturb best
                a_init = best_a[:]
                b_init = best_b[:]
                # Randomly flip a few transitions
                for _ in range(random.randint(1, 3)):
                    if random.random() < 0.5:
                        a_init[random.randint(0, N-1)] = random.randint(0, N-1)
                    else:
                        b_init[random.randint(0, N-1)] = random.randint(0, N-1)
                        
            score_init = _threshold_fast(a_init, b_init)
            if score_init > best_score:
                best_score = score_init
                best_a = a_init
                best_b = b_init
                
            # If the random/perturbed start was worse, we might just loop hill climbing on best again
            # To avoid infinite loop of no-improvement hill climbs, we rely on the 'stuck' break in _hill_climb
            # But if _hill_climb returns immediately (stuck=0 but no improvement), we need to force a change
            # The logic above: if score_climb == best_score, we go to else.
            # If we keep getting stuck at best_score, we will keep restarting.
            
            # To prevent tight loop if _hill_climb is very fast and always stuck:
            # We can add a small sleep or just let it run. The deadline guard is the ultimate stop.
            
    return [best_a, best_b]

# EVOLVE-BLOCK-END
