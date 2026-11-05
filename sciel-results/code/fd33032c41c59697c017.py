"""Size-efficient sorting network search for N=16 wires.

Strategy:
- Start from known good constructions (bitonic, odd-even merge, odd-even transposition).
- Use a fast bit-parallel verifier over all 65536 Boolean inputs.
- Aggressive local search:
  * Remove no-op comparators in one sweep.
  * Try swapping adjacent comparators, inserting, and removing.
  * Use a temperature-based acceptance to escape local minima.
- Multi-start with several seeds, each given a time slice.
- Deadline guard: return best found before 85s.
"""

import time
import random

N = 16
NUM_INPUTS = 1 << N  # 65536

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
        states[k + 1] = next_state
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

def is_sorted_state(final_state):
    """Check if the final state is sorted."""
    for i in range(N - 1):
        if final_state[i] & ~final_state[i + 1]:
            return False
    return True

def bitonic_sorter(n):
    """Generate Batcher's bitonic sorting network for n wires (n must be power of 2)."""
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

def odd_even_merge(n):
    """Odd-even merge sort network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    comparators = []
    def oe_merge(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        # Merge two sorted sequences of size half
        for i in range(half):
            a = start + i
            b = start + i + half
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        oe_merge(half, start, ascending)
        oe_merge(half, start + half, ascending)
    # First, sort odd and even positions
    def oe_sort(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        oe_sort(half, start, ascending)
        oe_sort(half, start + half, ascending)
        oe_merge(half, start, ascending)
    oe_sort(n, 0, True)
    return comparators

def odd_even_transposition(n):
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def random_network(n, seed):
    """Generate a random network of n comparators."""
    rng = random.Random(seed)
    comparators = []
    for _ in range(n):
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def local_search(net, deadline, seed):
    """
    Perform local search starting from net.
    Returns the best network found.
    """
    rng = random.Random(seed)
    best_net = list(net)
    best_count = len(net)
    current_net = list(net)
    
    # Initial cleanup
    current_net, final_state = remove_redundant(current_net)
    if is_sorted_state(final_state):
        best_net = list(current_net)
        best_count = len(current_net)
    
    # Local search loop
    temp = 0.5  # initial temperature for simulated annealing
    cooling = 0.999
    
    while time.time() < deadline:
        r = rng.random()
        
        # Try to remove a random comparator
        if len(current_net) > 0 and r < 0.5:
            idx = rng.randint(0, len(current_net) - 1)
            candidate = current_net[:idx] + current_net[idx + 1:]
            candidate, candidate_final = remove_redundant(candidate)
            if is_sorted_state(candidate_final):
                if len(candidate) < best_count:
                    best_net = list(candidate)
                    best_count = len(candidate)
                current_net = candidate
                temp = max(0.1, temp * cooling)
            else:
                # Accept worse solutions with probability
                if rng.random() < temp:
                    current_net = candidate
                temp = max(0.1, temp * cooling)
        
        # Try swapping two adjacent comparators
        elif len(current_net) > 1 and r < 0.7:
            idx = rng.randint(0, len(current_net) - 2)
            candidate = current_net[:]
            candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
            candidate, candidate_final = remove_redundant(candidate)
            if is_sorted_state(candidate_final):
                if len(candidate) < best_count:
                    best_net = list(candidate)
                    best_count = len(candidate)
                current_net = candidate
                temp = max(0.1, temp * cooling)
            else:
                if rng.random() < temp:
                    current_net = candidate
                temp = max(0.1, temp * cooling)
        
        # Try inserting a random comparator
        else:
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(current_net))
            candidate = current_net[:pos] + [(i, j)] + current_net[pos:]
            candidate, candidate_final = remove_redundant(candidate)
            if is_sorted_state(candidate_final):
                if len(candidate) < best_count:
                    best_net = list(candidate)
                    best_count = len(candidate)
                current_net = candidate
                temp = max(0.1, temp * cooling)
            else:
                if rng.random() < temp:
                    current_net = candidate
                temp = max(0.1, temp * cooling)
        
        # Periodically re-seed from best
        if rng.random() < 0.01:
            current_net = list(best_net)
    
    return best_net

def construct():
    deadline = time.time() + 80  # 80s to leave 10s margin for overhead
    
    # Multi-seed portfolio
    seeds = [
        bitonic_sorter(N),
        odd_even_merge(N),
        odd_even_transposition(N),
    ]
    
    # Add some random seeds
    for s in range(5):
        seeds.append(random_network(80, seed=100 + s))
    
    best_net = None
    best_count = float('inf')
    
    # For each seed, do local search
    for seed_idx, seed_net in enumerate(seeds):
        if time.time() > deadline:
            break
        
        # Verify the seed
        seed_net, final_state = remove_redundant(seed_net)
        if not is_sorted_state(final_state):
            # Try to fix by adding comparators
            # Just skip this seed if it's not valid after cleanup
            continue
        
        current_count = len(seed_net)
        if current_count < best_count:
            best_net = list(seed_net)
            best_count = current_count
        
        # Do local search from this seed
        search_deadline = min(deadline, time.time() + 15)  # 15s per seed
        found = local_search(seed_net, search_deadline, seed=seed_idx * 1000)
        if len(found) < best_count:
            best_net = found
            best_count = len(found)
    
    # Final cleanup
    if best_net is not None:
        best_net, _ = remove_redundant(best_net)
    else:
        # Fallback
        best_net = odd_even_transposition(N)
        best_net, _ = remove_redundant(best_net)
    
    return best_net
