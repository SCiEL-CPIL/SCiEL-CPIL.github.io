N = 18
# EVOLVE-BLOCK-START
import time
import random

# Precompute constants for fast verification
# We use Python big integers to represent the state of all 2^18 inputs.
# Bit 'm' in the integer corresponds to input mask 'm'.
# Wire i state is an integer M_i.

def _precompute_targets():
    """
    Precompute the target state for each wire for all 2^18 inputs.
    Target_i has bit 'm' set if and only if the sorted output for input 'm' 
    has a 1 on wire i.
    Wire 0 is the minimum, Wire N-1 is the maximum.
    For input m with k ones, sorted output has 0s in positions 0..N-1-k and 1s in N-k..N-1.
    So wire i is 1 iff k >= N - i.
    """
    targets = [0] * N
    # To speed up, we can group by popcount?
    # Or just iterate. 2^18 is 262144.
    # Iterating 262144 times in Python is ~0.05s.
    for m in range(1 << N):
        # Count bits
        k = bin(m).count('1')
        # For each wire i, if k >= N - i, set bit m in targets[i]
        # This is N operations per m. 18 * 262144 = 4.7M ops. ~0.5s.
        # We can optimize: 
        # targets[N-1] gets bit m if k >= 1
        # targets[N-2] gets bit m if k >= 2
        # ...
        # targets[0] gets bit m if k >= N
        
        # Let's update targets from bottom up or just set them.
        # Actually, we can build the masks more efficiently.
        # Let's just do the direct loop, it's a one-time cost.
        for i in range(N):
            if k >= N - i:
                targets[i] |= (1 << m)
    return targets

# Initial masks: Wire i has bit m set if the i-th bit of input m is 1.
def _precompute_initials():
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
        # Compare-exchange: 
        # min goes to i, max goes to j
        # For 0-1 inputs:
        # new_i = states[i] & states[j]  (1 only if both were 1)
        # new_j = states[i] | states[j]  (1 if either was 1)
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

def _random_network(rng):
    """Generate a random network (for restarts, though seed is better)."""
    comps = []
    # Random depth
    depth = rng.randint(10, 30)
    for _ in range(depth):
        # Random non-conflicting comparators? 
        # Just random pairs for now, verifier will reject invalid sorts.
        # But we want valid sorts. 
        # Better to start from seed and mutate.
        pass
    return comps

def _mutate(comps, rng):
    """
    Mutate a valid network.
    Operations:
    1. Swap two comparators.
    2. Insert a random comparator (might break validity, but we verify).
    3. Delete a random comparator (might break validity).
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
    
    deadline = time.time() + 55 # 5s safety margin
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
        # We iterate backwards to handle indices safely if we modify, 
        # but since we verify and break on success, we can just break.
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
            # 2. Try to swap adjacent comparators to escape local minima?
            # Or random mutation followed by pruning?
            # Let's try a few random mutations.
            for _ in range(10):
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

def _prune(comps, deadline):
    """
    Aggressively remove redundant comparators.
    """
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network_bitparallel(candidate):
                comps = candidate
                improved = True
                break
    return comps

# EVOLVE-BLOCK-END
