"""Size-efficient sorting network on 18 wires.
Uses bit-parallel verification for speed, greedy forward construction,
and aggressive pruning to minimize comparator count.
"""
import time
import random

N = 18
_TOTAL_INPUTS = 1 << N  # 262144

# Deadline: evaluator kills at 90s; keep a 5s safety margin.
_DEADLINE = 85.0

def _time_left():
    return _DEADLINE - time.time()


def _init_constants():
    """Precompute initial bit-vectors for all 2^N inputs as a list of Python ints."""
    initials = []
    for mask in range(_TOTAL_INPUTS):
        initials.append(mask)
    return initials


def _precompute_targets():
    """For each input mask, the sorted output is the mask with bits set to the
    lowest k positions where k = popcount(mask). We precompute this as a list."""
    targets = []
    for mask in range(_TOTAL_INPUTS):
        k = bin(mask).count('1')
        target = (1 << k) - 1
        targets.append(target)
    return targets


def _popcount_lut():
    """Build a 256-entry popcount lookup table."""
    lut = [0] * 256
    for i in range(256):
        lut[i] = bin(i).count('1')
    return lut


_POP_LUT = _popcount_lut()

def _popcount_fast(x):
    """Fast popcount using LUT for 32-bit chunks."""
    c = 0
    while x:
        c += _POP_LUT[x & 0xFF]
        x >>= 8
    return c


def _apply_comparator_parallel(state_list, i, j):
    """Apply a single comparator (i, j) to all states in parallel.
    state_list is a list of ints, one per input.
    For each state, if bit i > bit j, swap them.
    """
    mask_i = 1 << i
    mask_j = 1 << j
    # For each state, check if bit i is set and bit j is not
    # If so, we need to swap: clear bit i, set bit j
    # If bit j is set and bit i is not, clear bit j, set bit i
    # If both same, no change
    
    for idx in range(len(state_list)):
        s = state_list[idx]
        bi = (s >> i) & 1
        bj = (s >> j) & 1
        if bi != bj:
            if bi > bj:
                # swap: clear i, set j
                s = (s & ~mask_i) | mask_j
            else:
                # swap: clear j, set i
                s = (s & ~mask_j) | mask_i
            state_list[idx] = s


def _verify_network_bitparallel(net, initials, targets):
    """Verify network sorts all inputs using bit-parallel simulation.
    Returns True if all inputs are sorted correctly."""
    # Copy initials
    states = list(initials)
    
    for i, j in net:
        _apply_comparator_parallel(states, i, j)
    
    # Check all states match targets
    for idx in range(len(states)):
        if states[idx] != targets[idx]:
            return False
    return True


def _count_unsorted(net, initials, targets):
    """Count how many inputs are not sorted by the network."""
    states = list(initials)
    for i, j in net:
        _apply_comparator_parallel(states, i, j)
    count = 0
    for idx in range(len(states)):
        if states[idx] != targets[idx]:
            count += 1
    return count


def _odd_even_transposition(n):
    """Odd-even transposition sorting network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps


def _bitonic_merge(n, k, offset=0, reverse=False):
    """Generate comparators for bitonic merge of n elements."""
    comps = []
    if k == 1:
        return comps
    half = k // 2
    comps.extend(_bitonic_merge(n, half, offset, reverse))
    comps.extend(_bitonic_merge(n, half, offset + half, not reverse))
    for i in range(half):
        if not reverse:
            comps.append((offset + i, offset + i + half))
        else:
            comps.append((offset + i + half, offset + i))
    return comps


def _bitonic_sort_network(n):
    """Generate a bitonic sort network for n wires."""
    comps = []
    # Bitonic sort: recursively sort halves, then merge
    # For n=18, this isn't a power of 2, so we pad to 32 or use a different approach
    # Actually, let's just use odd-even transposition as the seed
    return _odd_even_transposition(n)


def _greedy_build(initials, targets, rng, time_limit=30.0):
    """Greedy forward construction: start empty, add comparators that reduce unsorted count."""
    net = []
    current_unsorted = _TOTAL_INPUTS  # all unsorted initially
    
    # All possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_comps)
    
    start_time = time.time()
    for comp in all_comps:
        if time.time() - start_time > time_limit:
            break
        if current_unsorted == 0:
            break
        
        trial = net + [comp]
        new_unsorted = _count_unsorted(trial, initials, targets)
        if new_unsorted < current_unsorted:
            net = trial
            current_unsorted = new_unsorted
    
    return net


def _prune(net, initials, targets, rng, time_limit=20.0):
    """Try to remove redundant comparators."""
    net = list(net)
    best = net
    best_len = len(net)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            trial = net[:idx] + net[idx + 1:]
            if _verify_network_bitparallel(trial, initials, targets):
                net = trial
                best = trial
                best_len = len(net)
                improved = True
                break
        if not improved:
            break
    
    return best


def _mutate(net, initials, targets, rng, time_limit=20.0):
    """Local search: try replacing, adding, or swapping comparators."""
    net = list(net)
    best = net
    best_len = len(net)
    start_time = time.time()
    
    all_comps = set((i, j) for i in range(N) for j in range(i + 1, N))
    
    while time.time() - start_time < time_limit:
        move_type = rng.choice(['remove', 'add', 'swap', 'replace'])
        
        if move_type == 'remove' and len(net) > 1:
            idx = rng.randrange(len(net))
            trial = net[:idx] + net[idx + 1:]
            
        elif move_type == 'add':
            comp = rng.choice(list(all_comps))
            pos = rng.randrange(len(net) + 1)
            trial = net[:pos] + [comp] + net[pos:]
            
        elif move_type == 'swap' and len(net) >= 2:
            idx = rng.randrange(len(net) - 1)
            trial = net[:]
            trial[idx], trial[idx + 1] = trial[idx + 1], trial[idx]
            
        elif move_type == 'replace' and len(net) > 0:
            idx = rng.randrange(len(net))
            comp = rng.choice(list(all_comps))
            trial = net[:idx] + [comp] + net[idx + 1:]
        else:
            continue
        
        if _verify_network_bitparallel(trial, initials, targets):
            if len(trial) < best_len:
                best = trial
                best_len = len(trial)
                net = trial
            elif len(trial) == best_len and rng.random() < 0.3:
                net = trial
    
    return best


def construct():
    """Construct a sorting network on 18 wires, minimizing comparator count."""
    start_time = time.time()
    rng = random.Random(42)
    
    # Precompute
    initials = _precompute_targets()  # We'll use masks as initials
    # Actually, initials should be the raw bit patterns
    initials = list(range(_TOTAL_INPUTS))
    targets = _precompute_targets()
    
    # Phase 1: Start with a known valid network (odd-even transposition)
    net = _odd_even_transposition(N)
    assert _verify_network_bitparallel(net, initials, targets), "Seed network invalid!"
    
    # Phase 2: Prune aggressively
    if _time_left() > 15:
        net = _prune(net, initials, targets, rng, time_limit=min(25.0, _time_left() - 12))
    
    # Phase 3: Local search / mutation
    if _time_left() > 15:
        net = _mutate(net, initials, targets, rng, time_limit=min(25.0, _time_left() - 12))
    
    # Phase 4: Final prune after mutation
    if _time_left() > 10:
        net = _prune(net, initials, targets, rng, time_limit=min(15.0, _time_left() - 8))
    
    # Final verification
    if not _verify_network_bitparallel(net, initials, targets):
        # Fallback
        net = _odd_even_transposition(N)
    
    return net
