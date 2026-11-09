import time
import random
import numpy as np

N = 18

# EVOLVE-BLOCK-START

def _precompute_targets_np():
    """
    Precompute target states for all 2^N inputs using NumPy.
    Returns a 2D array of shape (N, 1<<N) with dtype uint8.
    targets[i, m] is 1 if wire i should be 1 for input m.
    """
    targets = np.zeros((N, 1 << N), dtype=np.uint8)
    # Popcount for all masks
    masks = np.arange(1 << N, dtype=np.uint32)
    # Efficient popcount using bit tricks or just np.unpackbits if needed, 
    # but for 18 bits, a simple loop or vectorized bit count is fine.
    # Actually, np.bitwise_count is not standard in older numpy. 
    # Let's use a vectorized approach: sum of bits.
    # bits = (masks[:, None] >> np.arange(N)) & 1
    # This creates an (2^N, N) array. 262144 * 18 * 4 bytes = ~18MB. Acceptable.
    bits = (masks[:, None] >> np.arange(N, dtype=np.uint32)) & 1
    popcounts = bits.sum(axis=1)
    
    # For wire i (0-indexed), it is 1 in sorted output if popcount >= N - i
    # i=0 -> popcount >= N
    # i=N-1 -> popcount >= 1
    for i in range(N):
        threshold = N - i
        targets[i, :] = (popcounts >= threshold).astype(np.uint8)
    return targets

def _precompute_initials_np():
    """
    Precompute initial states for all 2^N inputs using NumPy.
    Returns a 2D array of shape (N, 1<<N) with dtype uint8.
    initials[i, m] is the value of wire i for input m.
    """
    initials = np.zeros((N, 1 << N), dtype=np.uint8)
    masks = np.arange(1 << N, dtype=np.uint32)
    for i in range(N):
        initials[i, :] = ((masks >> i) & 1).astype(np.uint8)
    return initials

_TARGETS = None
_INITIALS = None

def _init_constants():
    global _TARGETS, _INITIALS
    if _TARGETS is None:
        _TARGETS = _precompute_targets_np()
        _INITIALS = _precompute_initials_np()

def _verify_network_np(comparators):
    """
    Verifies the network using NumPy vectorized operations.
    """
    if _TARGETS is None:
        _init_constants()
    
    # Copy initials (N x 2^N)
    states = _INITIALS.copy()
    
    for i, j in comparators:
        # Compare-exchange: min to i, max to j
        # np.minimum and np.maximum are element-wise
        si = states[i]
        sj = states[j]
        states[i] = np.minimum(si, sj)
        states[j] = np.maximum(si, sj)
        
    # Check against targets
    # np.array_equal is fast for uint8 arrays
    return np.array_equal(states, _TARGETS)

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
    """
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing from the end first, as they are often redundant in transposition networks
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network_np(candidate):
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
        # Iterate backwards to handle indices safely
        for idx in range(len(best) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify_network_np(candidate):
                best = candidate
                best_size -= 1
                improved = True
                break # Restart scan from end
        
        if not improved:
            # 2. Try random mutations to escape local minima
            # Increase attempts per iteration to make better use of faster verification
            for _ in range(20):
                if time.time() > deadline:
                    break
                mutated = _mutate(best, rng)
                # Verify mutated
                if _verify_network_np(mutated):
                    # If valid, try to prune it
                    pruned = _prune(mutated, deadline)
                    if len(pruned) < best_size:
                        best = pruned
                        best_size = len(best)
                        improved = True
                        break
    return best

# EVOLVE-BLOCK-END
