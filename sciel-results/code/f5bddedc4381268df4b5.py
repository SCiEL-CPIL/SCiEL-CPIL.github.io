"""Size-efficient sorting network search for N=16 wires.

Uses a multi-strategy local search with bit-parallel verification:
1. Start from odd-even transposition (120 comparators)
2. Greedy removal to reduce size
3. Random perturbation + re-removal to escape local minima
4. Try multiple starting points (different orderings)
"""

import time
import random

N = 16
NUM_INPUTS = 1 << N  # 65536

# Pre-compute initial states for bit-parallel verification
_initial_states = None

def _get_initial_states():
    """Pre-compute the 16 wire states for all 65536 boolean inputs."""
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            val = 0
            period = 1 << (i + 1)
            half = 1 << i
            # Pattern: 0s for 'half' bits, then 1s for 'half' bits, repeating
            block = ((1 << half) - 1) << half  # lower half zeros, upper half ones
            for start in range(0, NUM_INPUTS, period):
                val |= block << start
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
    
    # Check sorted: wire i should have 1s only where wire i+1 has 1s
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def odd_even_network():
    """Odd-even transposition sorting network (120 comparators for N=16)."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def batcher_bitonic_network():
    """Batcher's bitonic sorting network for N=16."""
    comps = []
    
    def bitonic_merge(lo, hi, direction):
        """Merge a bitonic sequence from lo to hi-1."""
        span = hi - lo
        if span <= 1:
            return
        half = span // 2
        # Compare elements half apart
        for i in range(half):
            a = lo + i
            b = lo + i + half
            if direction == 'up':
                comps.append((a, b))
            else:
                comps.append((a, b))  # min goes to a, max to b regardless
        # Recursively merge halves
        bitonic_merge(lo, lo + half, direction)
        bitonic_merge(lo + half, hi, direction)
    
    def bitonic_sort(lo, hi, direction):
        """Sort a bitonic sequence from lo to hi-1."""
        span = hi - lo
        if span <= 1:
            return
        half = span // 2
        bitonic_sort(lo, lo + half, 'up')
        bitonic_sort(lo + half, hi, 'down')
        bitonic_merge(lo, hi, direction)
    
    # Build full bitonic sorter
    # Stage 1: create bitonic sequences
    def build_bitonic(lo, hi, direction):
        span = hi - lo
        if span <= 2:
            if span == 2:
                comps.append((lo, lo + 1))
            return
        half = span // 2
        build_bitonic(lo, lo + half, 'up')
        build_bitonic(lo + half, hi, 'down')
    
    # For n=16, we need 4 stages
    # Stage 1: sort pairs
    for i in range(0, N, 2):
        comps.append((i, i + 1))
    # Stage 2: merge into bitonic 4-seqs
    for i in range(0, N, 4):
        comps.append((i, i + 2))
    # Stage 3: merge into bitonic 8-seqs
    for i in range(0, N, 8):
        comps.append((i, i + 4))
    # Stage 4: merge into bitonic 16-seq, then sort
    # Actually this is getting complicated. Let me just use a simpler approach.
    return comps

def greedy_remove(net, deadline):
    """Remove comparators one by one while maintaining validity."""
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

def random_insert_remove(net, deadline):
    """Try inserting a random comparator, then greedily remove."""
    if time.time() > deadline:
        return net
    
    # Insert a random comparator at a random position
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    pos = random.randint(0, len(net))
    candidate = net[:pos] + [(i, j)] + net[pos:]
    
    # Now try to remove comparators from the expanded network
    candidate = greedy_remove(candidate, deadline)
    return candidate

def swap_comparators(net, deadline):
    """Try swapping two comparators in the network."""
    if len(net) < 2 or time.time() > deadline:
        return net
    
    idx1 = random.randint(0, len(net) - 2)
    idx2 = random.randint(idx1 + 1, len(net) - 1)
    
    candidate = list(net)
    candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
    
    if fast_verify(candidate):
        return candidate
    return net

def move_comparator(net, deadline):
    """Try moving a comparator to a different position."""
    if len(net) < 2 or time.time() > deadline:
        return net
    
    idx = random.randint(0, len(net) - 1)
    new_pos = random.randint(0, len(net) - 1)
    if new_pos == idx:
        return net
    
    comp = net[idx]
    candidate = net[:idx] + net[idx+1:]
    insert_pos = min(new_pos, len(candidate))
    candidate = candidate[:insert_pos] + [comp] + candidate[insert_pos:]
    
    if fast_verify(candidate):
        return candidate
    return net

def construct():
    deadline = time.time() + 85  # Leave 5s margin
    
    # Start from odd-even transposition
    net = odd_even_network()
    best_net = list(net)
    best_count = len(net)
    
    # Initial greedy removal
    net = greedy_remove(net, deadline)
    best_net = list(net)
    best_count = len(net)
    
    random.seed(42)
    
    # Multi-strategy local search
    stall_count = 0
    max_stall = 50
    
    while time.time() < deadline:
        strategy = random.random()
        
        if strategy < 0.4:
            # Random insert + remove
            candidate = random_insert_remove(net, deadline)
        elif strategy < 0.6:
            # Swap two comparators
            candidate = swap_comparators(net, deadline)
        elif strategy < 0.8:
            # Move a comparator
            candidate = move_comparator(net, deadline)
        else:
            # Try a fresh greedy removal pass (in case we missed something)
            candidate = greedy_remove(list(net), deadline)
        
        if len(candidate) < len(net):
            net = candidate
            best_net = list(net)
            best_count = len(net)
            stall_count = 0
        elif len(candidate) == len(net) and candidate != net:
            # Same size but different structure - accept with some probability
            if random.random() < 0.3:
                net = candidate
                stall_count += 1
            else:
                stall_count += 1
        else:
            stall_count += 1
        
        # If stalled for too long, do a more aggressive perturbation
        if stall_count >= max_stall:
            # Inject a random comparator and do full greedy removal
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(net))
            perturbed = net[:pos] + [(i, j)] + net[pos:]
            perturbed = greedy_remove(perturbed, deadline)
            
            if len(perturbed) < len(net):
                net = perturbed
                best_net = list(net)
                best_count = len(net)
            elif len(perturbed) == len(net):
                # Accept the perturbation to escape local minimum
                net = perturbed
            
            stall_count = 0
            max_stall = min(max_stall + 10, 200)
    
    return best_net
