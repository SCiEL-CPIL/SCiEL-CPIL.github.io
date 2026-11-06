"""
Size-efficient sorting networks on 17 wires – Enhanced Simulated Annealing approach.
Uses bitmask verification with multi-phase search: aggressive deletion, then fine-tuning.
"""

import time
import random
import math

N = 17
TOTAL_INPUTS = 1 << N  # 131072

# Precompute initial state bitmasks for verification
INITIAL_STATE = [0] * N
for i in range(N):
    block = 1 << i
    val = 0
    for m in range(block, TOTAL_INPUTS, 2 * block):
        bits_to_set = (1 << block) - 1
        val |= bits_to_set << m
    INITIAL_STATE[i] = val

def _is_valid(network):
    """
    Check if network sorts all 0/1 inputs using zero-one principle.
    Uses bitmask simulation for speed.
    """
    if not network:
        return N <= 1
    
    state = list(INITIAL_STATE)
    st = state
    for i, j in network:
        si = st[i]
        sj = st[j]
        st[i] = si & sj
        st[j] = si | sj
        
    for i in range(N - 1):
        if st[i] & ~st[i + 1]:
            return False
            
    return True

def _construct_odd_even_transposition():
    """Standard odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _try_redundant_removal(network, max_checks=None):
    """
    Try to remove a comparator that is redundant (removing it still keeps network valid).
    Returns (new_network, removed_idx) if successful, else (None, None).
    """
    n = len(network)
    if n <= 10:
        return None, None
    
    # Try random positions first, then systematic
    order = list(range(n))
    random.shuffle(order)
    
    checks = 0
    if max_checks is None:
        max_checks = n
    
    for idx in order:
        if checks >= max_checks:
            break
        candidate = network[:idx] + network[idx+1:]
        if _is_valid(candidate):
            return candidate, idx
        checks += 1
    
    return None, None

def construct():
    """
    Construct a sorting network with as few comparators as possible.
    Uses multi-phase Simulated Annealing: aggressive deletion, then fine-tuning.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Start with Odd-Even Transposition
    current_network = _construct_odd_even_transposition()
    best_network = list(current_network)
    best_len = len(current_network)
    current_len = len(current_network)
    
    if not _is_valid(current_network):
        return best_network
    
    # Phase 1: Aggressive redundant removal (no temperature needed)
    phase1_deadline = start_time + 15.0
    while time.time() < phase1_deadline:
        new_net, idx = _try_redundant_removal(current_network)
        if new_net is not None:
            current_network = new_net
            current_len = len(current_network)
            if current_len < best_len:
                best_network = list(current_network)
                best_len = current_len
        else:
            break  # No more redundant comparators
    
    # Phase 2: Simulated Annealing for fine-tuning
    T = 5.0  # Lower initial temperature since we're closer to optimum
    cooling_rate = 0.998
    min_T = 0.05
    
    stuck_counter = 0
    max_stuck = 5000  # Reset if stuck too long
    
    while time.time() < deadline:
        if T < min_T:
            break
        
        if stuck_counter > max_stuck:
            # Reset temperature to explore more
            T = max(T, 2.0)
            stuck_counter = 0
        
        # Generate a candidate move
        move_type = random.random()
        
        if move_type < 0.5:
            # Delete a random comparator
            if len(current_network) > 15:
                idx = random.randint(0, len(current_network) - 1)
                candidate = current_network[:idx] + current_network[idx+1:]
                new_len = current_len - 1
            else:
                continue
        elif move_type < 0.65:
            # Add a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            idx = random.randint(0, len(current_network))
            candidate = current_network[:idx] + [(i, j)] + current_network[idx:]
            new_len = current_len + 1
        elif move_type < 0.8:
            # Swap two comparators
            if len(current_network) < 2:
                continue
            idx1 = random.randint(0, len(current_network) - 1)
            idx2 = random.randint(0, len(current_network) - 1)
            if idx1 == idx2:
                continue
            candidate = list(current_network)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            new_len = current_len
        else:
            # Modify a comparator (change wires)
            if len(current_network) == 0:
                continue
            idx = random.randint(0, len(current_network) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate = list(current_network)
            candidate[idx] = (i, j)
            new_len = current_len
        
        # Verify candidate
        if not _is_valid(candidate):
            stuck_counter += 1
            continue
            
        # Accept or reject based on SA
        delta_cost = new_len - current_len
        if delta_cost < 0:
            # Improvement
            current_network = candidate
            current_len = new_len
            stuck_counter = 0
            if current_len < best_len:
                best_network = list(current_network)
                best_len = current_len
        else:
            # Worse or same
            if delta_cost == 0:
                if random.random() < 0.3:
                    current_network = candidate
            else:
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    current_network = candidate
                    current_len = new_len
            stuck_counter += 1
        
        # Cool down
        T *= cooling_rate
    
    return best_network
