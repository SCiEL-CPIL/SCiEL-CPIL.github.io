import time
import random
import sys

N = 18
TOTAL_INPUTS = 1 << N

# Precompute constants for fast verification using bit-parallelism
# State is a list of N integers. Integer i represents the state of wire i 
# across all 2^N inputs. Bit 'm' in integer i is 1 if wire i is 1 for input m.

def _precompute_targets():
    """
    Precompute the target state for each wire for all 2^N inputs.
    """
    targets = [0] * N
    for m in range(TOTAL_INPUTS):
        k = bin(m).count('1')
        start_wire = N - k
        for i in range(start_wire, N):
            targets[i] |= (1 << m)
    return targets

def _precompute_initials():
    """
    Initial state: Wire i has bit m set if the i-th bit of input m is 1.
    """
    initials = [0] * N
    for i in range(N):
        mask = 0
        # Bit i of m is 1.
        # Pattern: 000...000 111...111 000...000 111...111 ...
        # Period is 2^(i+1). First half 0, second half 1.
        # We can construct this faster using bit manipulation or loops.
        # Loop is fine for 18 wires.
        for m in range(TOTAL_INPUTS):
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
    Returns True if the network sorts all inputs.
    """
    if _TARGETS is None:
        _init_constants()
    
    states = list(_INITIALS)
    
    for i, j in comparators:
        si = states[i]
        sj = states[j]
        # min goes to i, max goes to j
        # For 0-1 values:
        # min(a,b) = a & b
        # max(a,b) = a | b
        states[i] = si & sj
        states[j] = si | sj
        
    for i in range(N):
        if states[i] != _TARGETS[i]:
            return False
    return True

def _count_inversions_bitparallel(states):
    """
    Counts the total number of inversions (unsorted pairs) across all inputs.
    """
    inv = 0
    for i in range(N):
        si = states[i]
        for j in range(i + 1, N):
            sj = states[j]
            inv += (si & ~sj).bit_count()
    return inv

def _seed_network():
    """Odd-Even Transposition Sort."""
    comps = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _prune_comparators(comps, deadline):
    """
    Aggressively remove redundant comparators.
    Uses randomized order to escape local minima of "first found removal".
    """
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle indices to try different removal orders
        indices = list(range(len(comps)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                return comps
            # Try removing comparator at idx
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network_bitparallel(candidate):
                comps = candidate
                improved = True
                break # Restart scan with new length
    return comps

def construct():
    _init_constants()
    
    # Deadline: 90s limit, keep 5s margin
    deadline = time.time() + 85.0
    
    # 1. Start with Seed Network
    network = _seed_network()
    best_network = network
    best_size = len(network)
    
    # 2. Aggressive Pruning Phase
    # Goal: Get to a local minimum size quickly.
    prune_deadline = time.time() + 10.0 # Give 10s to pruning
    network = _prune_comparators(network, prune_deadline)
    best_network = network
    best_size = len(network)
    
    # 3. Simulated Annealing / Local Search
    # We try to reduce size further by finding non-adjacent redundancies
    # or by reordering/inserting to enable further deletions.
    
    current_network = network
    current_size = len(network)
    
    # Pre-compute all possible comparator pairs for insertion
    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    T = 1.0
    no_improve_count = 0
    max_no_improve = 50 # Reset/shuffle if stuck
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart from best known to avoid drifting too far
            current_network = list(best_network)
            current_size = best_size
            no_improve_count = 0
            T = 1.0 # Reset temperature
            
        # Generate neighbor
        r = random.random()
        new_network = current_network[:]
        
        if r < 0.5:
            # Remove a random comparator
            if len(current_network) > 0:
                idx = random.randint(0, len(current_network) - 1)
                new_network = current_network[:idx] + current_network[idx+1:]
        elif r < 0.8:
            # Insert a random comparator
            # Heuristic: Pick a pair that is "far apart" or random
            comp = random.choice(all_comps)
            pos = random.randint(0, len(current_network))
            new_network = current_network[:pos] + [comp] + current_network[pos:]
        else:
            # Swap two comparators
            if len(current_network) > 1:
                i, j = random.sample(range(len(current_network)), 2)
                new_network[i], new_network[j] = new_network[j], new_network[i]
        
        new_size = len(new_network)
        
        # Verify
        is_valid = _verify_network_bitparallel(new_network)
        
        if is_valid:
            # Accept if better, or with some probability if same/worse
            if new_size < current_size:
                current_network = new_network
                best_network = new_network
                best_size = new_size
                T = max(0.1, T * 0.95) # Cool down
                no_improve_count = 0
            else:
                # Accept with probability p to escape local minima
                delta = new_size - current_size
                if delta == 0:
                    p = 0.1
                else:
                    p = min(1.0, 0.1 * (1.0 / T)) # Simple heuristic
                
                if random.random() < p:
                    current_network = new_network
                    current_size = new_size
                    T = max(0.1, T * 0.95)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1

    return best_network

# Ensure the function is available at module level as required
# The task says "Keep the function names and signatures of the seed program."
# The seed program has `construct()`.

