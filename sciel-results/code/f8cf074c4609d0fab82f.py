"""Simulated Annealing search for a 12-state binary automaton with long reset threshold."""
import random
import time
import numpy as np
from collections import deque

N = 12
MAX_TIME = 55.0  # Leave 5s margin for overhead
START_TEMP = 10.0
COOLING = 0.9999
MIN_TEMP = 0.1
RANDOM_SEED = 42

def _get_score(transitions):
    """
    Computes the maximum shortest word length needed to merge any pair of distinct states.
    This is a lower bound on the reset threshold and a good proxy for it.
    Returns -1000 if the automaton is not synchronizing.
    """
    t0 = transitions[0]
    t1 = transitions[1]
    max_dist = 0
    
    # Check all pairs
    for i in range(N):
        for j in range(i + 1, N):
            visited = set()
            visited.add((i, j))
            queue = deque([(i, j, 0)])
            found = False
            dist = 0
            
            while queue:
                u, v, d = queue.popleft()
                if u == v:
                    dist = d
                    found = True
                    break
                
                # Try letter 0
                nu = t0[u]
                nv = t0[v]
                if nu == nv:
                    dist = d + 1
                    found = True
                    break
                # Normalize pair for visited check (order doesn't matter for merging)
                a, b = (nu, nv) if nu < nv else (nv, nu)
                if a != b and (a, b) not in visited:
                    visited.add((a, b))
                    queue.append((a, b, d + 1))
                
                # Try letter 1
                nu = t1[u]
                nv = t1[v]
                if nu == nv:
                    dist = d + 1
                    found = True
                    break
                a, b = (nu, nv) if nu < nv else (nv, nu)
                if a != b and (a, b) not in visited:
                    visited.add((a, b))
                    queue.append((a, b, d + 1))
            
            if not found:
                return -1000
            if dist > max_dist:
                max_dist = dist
                
    return max_dist

def construct():
    """
    Use simulated annealing to find a 12-state binary automaton with
    a long shortest reset word.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    start_time = time.time()
    
    # Initialize with a random automaton
    current = [[random.randint(0, N - 1) for _ in range(N)] for _ in range(2)]
    
    current_score = _get_score(current)
    
    # If not synchronizing, mutate until it is
    while current_score < 0:
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        current[letter][state] = random.randint(0, N - 1)
        current_score = _get_score(current)
        
    best = [row[:] for row in current]
    best_score = current_score
    
    temp = START_TEMP
    
    while temp > MIN_TEMP:
        if time.time() - start_time > MAX_TIME:
            break
            
        # Create a neighbor
        neighbor = [row[:] for row in current]
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        old_val = neighbor[letter][state]
        new_val = random.randint(0, N - 1)
        if new_val == old_val:
            temp *= COOLING
            continue
        neighbor[letter][state] = new_val
        
        neighbor_score = _get_score(neighbor)
        
        # Accept or reject
        if neighbor_score >= current_score:
            current = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best = [row[:] for row in current]
                best_score = current_score
        else:
            delta = current_score - neighbor_score
            # If neighbor is non-synchronizing (negative score), delta is large positive
            # exp(-large/temp) is 0, so reject.
            if delta < 0:
                accept = True
            else:
                # Avoid overflow
                if delta / temp > 100:
                    accept_prob = 0.0
                else:
                    accept_prob = np.exp(-delta / temp)
                accept = (random.random() < accept_prob)
            
            if accept:
                current = neighbor
                current_score = neighbor_score
                if current_score > best_score:
                    best = [row[:] for row in current]
                    best_score = current_score
        
        temp *= COOLING
        
    return best

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
