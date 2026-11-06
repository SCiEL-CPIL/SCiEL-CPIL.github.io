import time
import random
import math
import numpy as np

N = 17
TOTAL_INPUTS = 1 << N  # 131072
DEADLINE = 85.0  # Safety margin before 90s kill

# Precompute initial state bitmasks for verification
# Each wire i has a bitmask where bit x is set iff input x has wire i = 1
# For wire i, the pattern is: blocks of 2^i zeros followed by 2^i ones
_INITIAL_STATE = [0] * N
for i in range(N):
    block = 1 << i
    val = 0
    for m in range(block, TOTAL_INPUTS, 2 * block):
        bits_to_set = (1 << block) - 1
        val |= bits_to_set << m
    _INITIAL_STATE[i] = val

def _is_valid(network):
    """
    Check if network sorts all 0/1 inputs using zero-one principle.
    Uses bitmask simulation for speed.
    """
    if not network:
        return N <= 1
    
    state = list(_INITIAL_STATE)
    st = state
    for i, j in network:
        si = st[i]
        sj = st[j]
        st[i] = si & sj
        st[j] = si | sj
        
    # Check sorted: for each adjacent pair, no wire i has 1 where wire i+1 has 0
    for i in range(N - 1):
        if st[i] & ~st[i + 1]:
            return False
            
    return True

def _construct_odd_even_transposition():
    """Standard odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _greedy_removal(network, deadline):
    """Aggressively remove redundant comparators."""
    current = list(network)
    current_len = len(current)
    
    while time.time() < deadline:
        n = len(current)
        if n <= 1:
            break
        
        # Try random positions
        order = list(range(n))
        random.shuffle(order)
        
        removed = False
        for idx in order:
            if time.time() > deadline:
                break
            candidate = current[:idx] + current[idx+1:]
            if _is_valid(candidate):
                current = candidate
                removed = True
                break
        
        if not removed:
            break
    
    return current

def _simulated_annealing(network, start_time, deadline):
    """
    Simulated annealing with diverse moves to escape local minima.
    """
    current = list(network)
    current_len = len(current)
    best = list(current)
    best_len = current_len
    
    T = 3.0
    cooling_rate = 0.9995
    min_T = 0.1
    
    stuck_counter = 0
    max_stuck = 2000
    
    while time.time() < deadline:
        if T < min_T:
            break
        
        if stuck_counter > max_stuck:
            T = max(T, 1.5)
            stuck_counter = 0
        
        # Generate a candidate move
        move_type = random.random()
        
        if move_type < 0.45:
            # Delete a random comparator
            if len(current) > 20:
                idx = random.randint(0, len(current) - 1)
                candidate = current[:idx] + current[idx+1:]
                new_len = current_len - 1
            else:
                continue
        elif move_type < 0.55:
            # Add a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            idx = random.randint(0, len(current))
            candidate = current[:idx] + [(i, j)] + current[idx:]
            new_len = current_len + 1
        elif move_type < 0.75:
            # Swap two comparators
            if len(current) < 2:
                continue
            idx1 = random.randint(0, len(current) - 1)
            idx2 = random.randint(0, len(current) - 1)
            if idx1 == idx2:
                continue
            candidate = list(current)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            new_len = current_len
        else:
            # Modify a comparator (change one of its wires)
            if len(current) == 0:
                continue
            idx = random.randint(0, len(current) - 1)
            i, j = current[idx]
            if random.random() < 0.5:
                # Change wire i
                new_i = random.randint(0, N - 1)
                if new_i == j:
                    continue
                new_pair = (min(new_i, j), max(new_i, j))
            else:
                # Change wire j
                new_j = random.randint(0, N - 1)
                if new_j == i:
                    continue
                new_pair = (min(i, new_j), max(i, new_j))
            candidate = list(current)
            candidate[idx] = new_pair
            new_len = current_len
        
        if not _is_valid(candidate):
            continue
        
        delta = new_len - current_len
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current = candidate
            current_len = new_len
            if current_len < best_len:
                best = list(current)
                best_len = current_len
                stuck_counter = 0
            else:
                stuck_counter += 1
        else:
            stuck_counter += 1
        
        T *= cooling_rate
    
    return best

def construct():
    start_time = time.time()
    
    # Start with odd-even transposition
    network = _construct_odd_even_transposition()
    
    if not _is_valid(network):
        return network
    
    # Phase 1: Aggressive greedy removal (first 20 seconds)
    phase1_deadline = start_time + 20.0
    network = _greedy_removal(network, phase1_deadline)
    
    # Phase 2: Simulated annealing for the remaining time
    best = _simulated_annealing(network, start_time, start_time + DEADLINE)
    
    return best
