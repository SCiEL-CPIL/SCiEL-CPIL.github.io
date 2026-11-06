"""Size-efficient sorting network search for N=16 wires.

Combines the direct greedy removal of the seed program with simulated annealing
inspired structural moves (move/swap/insert) to escape local minima.
Uses fast bit-parallel verification.
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

def shell_sort_network():
    """Shell sort variant."""
    comparators = []
    gaps = [8, 4, 2, 1]
    for gap in gaps:
        for i in range(gap, N):
            comparators.append((i - gap, i))
    return comparators

def merge_sort_network(n):
    """Merge sort network."""
    if n <= 1:
        return []
    comparators = []
    
    def merge_sort(size, start):
        if size <= 1:
            return
        half = size // 2
        merge_sort(half, start)
        merge_sort(half, start + half)
        merge(half, start)
    
    def merge(size, start):
        if size <= 1:
            return
        for i in range(size):
            a = start + i
            b = start + i + size
            comparators.append((a, b))
        merge(size // 2, start)
        merge(size // 2, start + size)
    
    merge_sort(n, 0)
    return comparators

def remove_redundant(net):
    """
    Removes all comparators that are no-ops in the current network.
    Returns (new_net, final_state).
    """
    if not net:
        return [], _get_initial_states()[:]
        
    current_state = _get_initial_states()[:]
    new_net = []
    
    for (i, j) in net:
        min_val = current_state[i] & current_state[j]
        max_val = current_state[i] | current_state[j]
        
        # If the comparator is a no-op (min == state[i] and max == state[j] for all inputs)
        # i.e., state[i] is already <= state[j] for all inputs
        if (current_state[i] & ~current_state[j]) == 0:
            # No-op, skip adding to new_net
            pass
        else:
            # Not a no-op, update state and add
            new_net.append((i, j))
            current_state[i] = min_val
            current_state[j] = max_val
        # Note: Even if it's a no-op, the state doesn't change, so we don't need to update.
        # But wait, if it IS a no-op, state[i] and state[j] remain unchanged.
        # So we just skip the update.
        
    return new_net, current_state

def is_sorted_state(state):
    """Check if the final state is sorted (non-decreasing)."""
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def construct():
    deadline = time.time() + 85
    random.seed(42)
    
    best_net = None
    best_count = float('inf')
    
    # Generate multiple starting points
    start_nets = []
    
    # 1. Bitonic sorter
    bn = bitonic_sorter(N)
    bn_clean, bn_final = remove_redundant(bn)
    if is_sorted_state(bn_final):
        start_nets.append(bn_clean)
    
    # 2. Odd-even transposition
    oen = odd_even_network()
    oen_clean, oen_final = remove_redundant(oen)
    if is_sorted_state(oen_final):
        start_nets.append(oen_clean)
        
    # 3. Shell sort
    sn = shell_sort_network()
    sn_clean, sn_final = remove_redundant(sn)
    if is_sorted_state(sn_final):
        start_nets.append(sn_clean)
        
    # 4. Merge sort
    mn = merge_sort_network(N)
    mn_clean, mn_final = remove_redundant(mn)
    if is_sorted_state(mn_final):
        start_nets.append(mn_clean)
        
    # Select the best starting net
    for net in start_nets:
        if len(net) < best_count:
            best_count = len(net)
            best_net = net
            
    if best_net is None:
        # Fallback, should not happen
        return bitonic_sorter(N)
        
    current_net = list(best_net)
    
    # Local Search / Simulated Annealing
    # We try to modify current_net to find a smaller valid network.
    
    while time.time() < deadline:
        # First, clean the current network
        current_net, current_final = remove_redundant(current_net)
        
        if not is_sorted_state(current_final):
            # Should not happen if we only accept valid moves, but safeguard
            current_net = list(best_net)
            continue
            
        # Update best if improved
        if len(current_net) < best_count:
            best_count = len(current_net)
            best_net = list(current_net)
            
        # If we have reached the theoretical minimum or reference, stop?
        # Reference is 60. If we hit 60, we can stop or try to beat it.
        # For now, keep searching.
        
        if len(current_net) <= 60:
            break
            
        # Generate a candidate move
        # 1. Move a comparator
        # 2. Swap two comparators
        # 3. Insert a comparator (rarely, to escape deep minima)
        
        move_type = random.random()
        candidate = list(current_net)
        
        if not candidate:
            break
            
        if move_type < 0.5:
            # Move
            if len(candidate) > 1:
                idx = random.randint(0, len(candidate) - 1)
                new_idx = random.randint(0, len(candidate) - 1)
                if idx != new_idx:
                    comp = candidate.pop(idx)
                    candidate.insert(new_idx, comp)
            else:
                continue # Can't move in 1-element list
        elif move_type < 0.8:
            # Swap
            if len(candidate) > 1:
                i, j = random.sample(range(len(candidate)), 2)
                candidate[i], candidate[j] = candidate[j], candidate[i]
            else:
                continue
        else:
            # Insert
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(candidate))
            candidate.insert(pos, (i, j))
            
        # Verify candidate
        # We can use remove_redundant to both clean and get final state
        candidate_clean, candidate_final = remove_redundant(candidate)
        
        if is_sorted_state(candidate_final):
            # Valid move
            # Acceptance criteria:
            # 1. If smaller, always accept.
            # 2. If same size, accept with probability 0.5 (explore).
            # 3. If larger, reject (we want to minimize).
            
            if len(candidate_clean) < len(current_net):
                current_net = candidate_clean
            elif len(candidate_clean) == len(current_net):
                if random.random() < 0.5:
                    current_net = candidate_clean
            # else: reject
        
    # Final cleanup
    best_net, _ = remove_redundant(best_net)
    
    return best_net
