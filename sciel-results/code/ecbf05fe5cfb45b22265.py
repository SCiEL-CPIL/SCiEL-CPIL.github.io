"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
DEADLINE = time.time() + 85  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def _verify_network(comparators):
    """Verify that the network sorts all 2^18 boolean inputs using fast bitwise checks."""
    inputs = np.arange(1 << N, dtype=np.uint32)
    
    for i, j in comparators:
        mask_i = np.uint32(1) << i
        mask_j = np.uint32(1) << j
        # Calculate swap mask: bit i is 1 and bit j is 0
        # bi = (inputs >> i) & 1
        # bj = (inputs >> j) & 1
        # swap_mask = bi & ~bj
        swap_mask = ((inputs >> i) & 1) & ~((inputs >> j) & 1)
        # Note: ~ on uint32 in numpy might behave unexpectedly if not masked, 
        # but since we only care about the bit position, and we use it as a boolean mask, 
        # we should be careful. 
        # Actually, ((inputs >> i) & 1) is 0 or 1.
        # ~((inputs >> j) & 1) is -1 or 0 in signed, or all 1s or 0 in unsigned?
        # In numpy, ~ on uint32 flips all bits.
        # So ~1 is 0xFFFFFFFF, ~0 is 0xFFFFFFFF.
        # So swap_mask will be 0 or 0xFFFFFFFF.
        # We need a boolean mask.
        # Let's use explicit boolean logic to be safe and clear.
        bi = (inputs >> i) & 1
        bj = (inputs >> j) & 1
        swap_mask = (bi == 1) & (bj == 0)
        
        # Apply swap in-place
        idx = np.flatnonzero(swap_mask)
        if len(idx) > 0:
            inputs[idx] ^= (mask_i | mask_j)
    
    # Check if sorted
    # A sequence is sorted if there are no '10' patterns in the bits.
    # i.e., for all k, if bit k is 1, bit k+1 must be 1.
    # Violation: bit k is 1 and bit k+1 is 0.
    is_unsorted = np.zeros(1 << N, dtype=bool)
    for k in range(N - 1):
        b_k = (inputs >> k) & 1
        b_k1 = (inputs >> (k + 1)) & 1
        is_unsorted |= (b_k == 1) & (b_k1 == 0)
        
    return not is_unsorted.any()

def _make_seed_network():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _local_search_remove(comparators, budget=30):
    """Aggressively try to remove comparators."""
    comps = list(comparators)
    start_time = time.time()
    
    improved = True
    while improved and time.time() < start_time + budget:
        improved = False
        # Iterate in reverse to handle removals
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > start_time + budget:
                break
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network(candidate):
                comps = candidate
                improved = True
                break
    
    return comps

def _simulated_annealing(start_comps, budget=60):
    """Simulated annealing to minimize comparator count."""
    comps = list(start_comps)
    current_count = len(comps)
    best_comps = list(comps)
    best_count = current_count
    
    T = 10.0
    cooling = 0.995
    start_time = time.time()
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < start_time + budget and T > 0.01:
        # Random move
        move_type = random.random()
        
        if move_type < 0.3 and len(comps) > 1:
            # Remove a random comparator
            idx = random.randint(0, len(comps) - 1)
            new_comps = comps[:idx] + comps[idx+1:]
            delta = -1
        elif move_type < 0.6:
            # Add a random comparator
            i, j = random.choice(all_pairs)
            new_comps = comps + [(i, j)]
            delta = 1
        elif move_type < 0.8:
            # Swap two comparators
            if len(comps) >= 2:
                idx1 = random.randint(0, len(comps) - 1)
                idx2 = random.randint(0, len(comps) - 1)
                if idx1 != idx2:
                    new_comps = list(comps)
                    new_comps[idx1], new_comps[idx2] = new_comps[idx2], new_comps[idx1]
                    delta = 0
                else:
                    T *= cooling
                    continue
            else:
                T *= cooling
                continue
        else:
            # Replace a comparator
            if len(comps) > 0:
                idx = random.randint(0, len(comps) - 1)
                i, j = random.choice(all_pairs)
                new_comps = list(comps)
                new_comps[idx] = (i, j)
                delta = 0
            else:
                T *= cooling
                continue
        
        new_count = len(new_comps)
        # Acceptance
        if delta < 0 or random.random() < np.exp(-delta / T):
            comps = new_comps
            current_count = new_count
            if current_count < best_count:
                best_count = current_count
                best_comps = list(comps)
        
        T *= cooling
    
    return best_comps, best_count

def construct():
    """Main entry point. Returns a list of (i, j) comparators that sort all 18-input sequences."""
    start_time = time.time()
    
    # Start with seed network
    network = _make_seed_network()
    
    # Phase 1: Aggressive local search removal
    if time.time() < DEADLINE - 30:
        network = _local_search_remove(network, budget=min(30, DEADLINE - 30 - (time.time() - start_time)))
    
    # Phase 2: Simulated annealing
    remaining = DEADLINE - 10 - (time.time() - start_time)
    if remaining > 10:
        sa_net, sa_count = _simulated_annealing(network, budget=min(remaining, 50))
        if sa_count < len(network):
            network = sa_net
    
    # Phase 3: Final greedy removal
    remaining = DEADLINE - 5 - (time.time() - start_time)
    if remaining > 2:
        network = _local_search_remove(network, budget=max(0, remaining))
    
    # Verify the final network
    if not _verify_network(network):
        # Fallback to seed
        network = _make_seed_network()
    
    return network

# EVOLVE-BLOCK-END
