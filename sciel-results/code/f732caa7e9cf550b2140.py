"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 18

# EVOLVE-BLOCK-START

# We use Python big integers to represent the state of all 2^N inputs.
# Bit 'm' in the integer corresponds to input mask 'm'.
# Wire i state is an integer M_i.

def _precompute_targets():
    """
    Precompute the target state for each wire for all 2^N inputs.
    Target_i has bit 'm' set if and only if the sorted output for input 'm' 
    has a 1 on wire i.
    Wire 0 is the minimum, Wire N-1 is the maximum.
    For input m with k ones, sorted output has 0s in positions 0..N-1-k and 1s in N-k..N-1.
    So wire i is 1 iff k >= N - i.
    """
    targets = [0] * N
    # Precompute popcount for all masks to speed up target generation
    popcounts = [0] * (1 << N)
    for m in range(1, 1 << N):
        popcounts[m] = popcounts[m >> 1] + (m & 1)
        
    # Initialize masks for each possible popcount k
    masks_by_popcount = [0] * (N + 1)
    for m in range(1 << N):
        k = popcounts[m]
        masks_by_popcount[k] |= (1 << m)
        
    # Construct targets
    # Wire i is 1 if popcount >= N - i
    for i in range(N):
        threshold = N - i
        target_mask = 0
        for k in range(threshold, N + 1):
            target_mask |= masks_by_popcount[k]
        targets[i] = target_mask
        
    return targets

def _precompute_initials():
    """
    Precompute initial states for all 2^N inputs.
    initials[i] has bit 'm' set if the i-th bit of input m is 1.
    """
    initials = [0] * N
    for i in range(N):
        mask = 0
        for m in range(1 << N):
            if (m >> i) & 1:
                mask |= (1 << m)
        initials[i] = mask
    return initials

_TARGETS = None
_INITIALS = None

def _init_constants():
    global _TARGETS, _INITIALS
    if _TARGETS is None:
        _TARGETS = _precompute_targets()
        _INITIALS = _precompute_initials()

def _verify_network_bitparallel(comparators):
    """
    Verifies the network using bit-parallel simulation.
    """
    if _TARGETS is None:
        _init_constants()
    
    # Copy initials
    states = list(_INITIALS)
    
    for i, j in comparators:
        si = states[i]
        sj = states[j]
        states[i] = si & sj
        states[j] = si | sj
        
    # Check against targets
    for i in range(N):
        if states[i] != _TARGETS[i]:
            return False
    return True

def _seed_network():
    """Odd-Even Transposition Sort."""
    comps = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _prune(comps, deadline):
    """
    Aggressively remove redundant comparators.
    We try to remove comparators from the end first, as they are often redundant.
    If a removal fails, we move to the previous one.
    """
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing from the end first
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network_bitparallel(candidate):
                comps = candidate
                improved = True
                break
    return comps

def _mutate(comps, rng):
    """
    Mutate a valid network.
    """
    comps = list(comps)
    action = rng.random()
    
    if action < 0.4:
        # Swap
        if len(comps) > 1:
            i, j = rng.sample(range(len(comps)), 2)
            comps[i], comps[j] = comps[j], comps[i]
    elif action < 0.7:
        # Insert
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        pos = rng.randint(0, len(comps))
        comps.insert(pos, (i, j))
    else:
        # Delete
        if len(comps) > 1:
            idx = rng.randint(0, len(comps) - 1)
            comps.pop(idx)
            
    return comps

def construct():
    _init_constants()
    
    # 90s limit, keep 5s safety margin
    deadline = time.time() + 85.0
    rng = random.Random(42)
    
    best = _seed_network()
    best_size = len(best)
    
    # Initial Pruning
    best = _prune(best, deadline)
    best_size = len(best)
    
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to remove any comparator
        for idx in range(len(best) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify_network_bitparallel(candidate):
                best = candidate
                best_size -= 1
                improved = True
                break # Restart scan from end
        
        if not improved:
            # 2. Try random mutations to escape local minima
            # We perform a batch of mutations. If any leads to a valid network 
            # that can be pruned to a smaller size, we accept it.
            for _ in range(50):
                if time.time() > deadline:
                    break
                mutated = _mutate(best, rng)
                # Verify mutated
                if _verify_network_bitparallel(mutated):
                    # If valid, try to prune it
                    pruned = _prune(mutated, deadline)
                    if len(pruned) < best_size:
                        best = pruned
                        best_size = len(best)
                        improved = True
                        break
    return best

# EVOLVE-BLOCK-END
