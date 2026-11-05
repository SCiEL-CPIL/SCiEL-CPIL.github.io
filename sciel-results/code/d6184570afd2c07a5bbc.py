"""Size-efficient sorting network search for N=16 wires.

Combined approach:
- Start from the known 64-comparator Batcher bitonic sorter.
- Use fast bit-parallel verification (zero-one principle over all 65536 boolean inputs).
- Use a "remove_redundant" pass that computes all states once and removes all
  no-op comparators in a single sweep (much faster than repeated full re-verifications).
- Local search: insert random comparators, then clean via remove_redundant.
- Deadline guard to ensure we finish well before the 90s limit.
"""

import time
import random

N = 16
NUM_INPUTS = 1 << N  # 65536
ALL_ONES = (1 << NUM_INPUTS) - 1

# Pre-compute initial states for bit-parallel verification
_initial_states = None

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            # For input j (0..65535), wire i has value (j >> i) & 1
            # Build a 65536-bit integer where bit j = (j >> i) & 1
            # Pattern: period 2^(i+1), lower half 0, upper half 1
            val = 0
            period = 1 << (i + 1)
            half = 1 << i
            period_val = ((1 << half) - 1) << half
            num_periods = NUM_INPUTS // period
            shift = 0
            for _ in range(num_periods):
                val |= period_val << shift
                shift += period
            _initial_states.append(val)
    return _initial_states

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    state = _get_initial_states()[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def bitonic_sorter(n):
    """
    Generate Batcher's bitonic sorting network for n wires (n must be power of 2).
    Returns a list of (i, j) comparator pairs.
    """
    if n <= 1:
        return []
    
    comparators = []
    
    def bitonic_sort(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(half, start, ascending)
        bitonic_sort(half, start + half, not ascending)
        bitonic_merge(half, start, ascending)
    
    def bitonic_merge(size, start, ascending):
        if size <= 1:
            return
        for i in range(size):
            a = start + i
            b = start + i + size
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        bitonic_merge(size // 2, start, ascending)
        bitonic_merge(size // 2, start + size, ascending)
    
    bitonic_sort(n, 0, True)
    return comparators

def odd_even_network():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def compute_states(net):
    """
    Returns (states, final_state) where states[k] is the state of all wires
    before comparator k is applied. states[0] is the initial state.
    """
    states = [None] * (len(net) + 1)
    current_state = _get_initial_states()[:]
    states[0] = current_state
    
    for k, (i, j) in enumerate(net):
        next_state = current_state[:]
        min_val = current_state[i] & current_state[j]
        max_val = current_state[i] | current_state[j]
        next_state[i] = min_val
        next_state[j] = max_val
        states[k+1] = next_state
        current_state = next_state
        
    return states, current_state

def remove_redundant(net):
    """
    Removes all comparators that are no-ops in the current network.
    Returns (new_net, final_state).
    """
    if not net:
        return [], _get_initial_states()[:]
        
    states, final_state = compute_states(net)
    to_remove = set()
    
    for k, (i, j) in enumerate(net):
        # Check if wire i <= wire j for all inputs before this comparator
        if states[k][i] & ~states[k][j] == 0:
            to_remove.add(k)
            
    if not to_remove:
        return net, final_state
        
    new_net = [net[k] for k in range(len(net)) if k not in to_remove]
    return new_net, final_state

def greedy_remove(net, deadline):
    """
    Try to remove each comparator one by one. If removal keeps the network valid,
    remove it and continue. Returns the reduced network.
    """
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                return net
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                net = candidate
                changed = True
                break
    return net

def construct():
    deadline = time.time() + 50  # 50s to leave 40s margin for overhead
    
    # Start from bitonic sorter (64 comparators for n=16)
    net = bitonic_sorter(N)
    best_net = list(net)
    best_count = len(net)
    
    # Verify initial network and clean
    net, final_state = remove_redundant(net)
    
    # Check if sorted
    is_sorted = True
    for i in range(N - 1):
        if final_state[i] & ~final_state[i + 1]:
            is_sorted = False
            break
            
    if not is_sorted:
        # Fallback to odd-even if bitonic has a bug
        best_net = odd_even_network()
        best_count = len(best_net)
        net, _ = remove_redundant(best_net)
    else:
        best_net = list(net)
        best_count = len(net)
    
    # More aggressive local search with insertions
    random.seed(123)
    
    # Try multiple restarts with random insertions
    # Increase iterations since remove_redundant is fast
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Try inserting a random comparator at a random position
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(net))
        candidate = net[:pos] + [(i, j)] + net[pos:]
        
        # Clean the candidate
        candidate, candidate_final = remove_redundant(candidate)
        
        # Check if candidate is valid (sorted)
        is_sorted = True
        for k in range(N - 1):
            if candidate_final[k] & ~candidate_final[k + 1]:
                is_sorted = False
                break
                
        if is_sorted and len(candidate) < len(net):
            net = candidate
            best_net = list(net)
            best_count = len(net)
    
    # Final check: try a few more aggressive passes
    for _ in range(3):
        if time.time() > deadline:
            break
        net = greedy_remove(net, deadline)
        if len(net) < best_count:
            best_net = list(net)
            best_count = len(net)
    
    # Final cleanup
    best_net, _ = remove_redundant(best_net)
    
    return best_net
