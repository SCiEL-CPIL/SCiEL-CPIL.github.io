"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 17
DEADLINE = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def _apply_comparators(state, comps):
    """Apply a list of comparators to a state (list of ints)."""
    for i, j in comps:
        if state[i] > state[j]:
            state[i], state[j] = state[j], state[i]
    return state

def _verify_network(comps):
    """Verify that the network sorts all 2^17 Boolean inputs."""
    n = len(comps)
    # Precompute for speed: use bit masks
    # Represent state as integer bitmask
    # For each input mask, apply comparators
    # We can vectorize somewhat, but let's do it efficiently in Python
    
    # Use a more efficient approach: track which wires are 0 and which are 1
    # Actually, the zero-one principle means we just need to check all 2^17 inputs
    # Let's use a faster method: represent state as a list and use the fact that
    # after sorting, all 0s come before all 1s
    
    for mask in range(1 << N):
        state = [(mask >> i) & 1 for i in range(N)]
        for i, j in comps:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        # Check if sorted
        for k in range(N - 1):
            if state[k] > state[k + 1]:
                return False
    return True

def _verify_network_fast(comps):
    """Faster verification using bit-level operations where possible."""
    # For Boolean inputs, a comparator (i,j) does:
    # if bit_i=1 and bit_j=0, swap them
    # This is equivalent to: new_i = bit_i AND bit_j, new_j = bit_i OR bit_j
    
    # We can represent the state as a single integer
    # For each comparator, we need to check and potentially swap bits
    
    # Let's try a different approach: use the fact that we're dealing with 0/1
    # and process all masks at once isn't easy in pure Python without numpy
    
    # Use numpy if available
    try:
        import numpy as np
        # Process in batches
        batch_size = 1024
        total = 1 << N
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            masks = np.arange(start, end, dtype=np.uint32)
            # Create state array: shape (batch_size, N)
            # state[b, i] = (masks[b] >> i) & 1
            states = np.zeros((end - start, N), dtype=np.uint8)
            for i in range(N):
                states[:, i] = ((masks >> i) & 1).astype(np.uint8)
            
            for i, j in comps:
                # Compare and swap
                swap_mask = (states[:, i] > states[:, j])
                states[:, i] = np.where(swap_mask, states[:, j], states[:, i])
                states[:, j] = np.where(swap_mask, states[:, i], states[:, j])
            
            # Check if sorted: for each row, check if non-decreasing
            for k in range(N - 1):
                if np.any(states[:, k] > states[:, k + 1]):
                    return False
        return True
    except ImportError:
        # Fallback to pure Python
        for mask in range(1 << N):
            state = [(mask >> i) & 1 for i in range(N)]
            for i, j in comps:
                if state[i] > state[j]:
                    state[i], state[j] = state[j], state[i]
            for k in range(N - 1):
                if state[k] > state[k + 1]:
                    return False
        return True

def _odd_even_transposition_network():
    """Classic odd-even transposition sorting network."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _build_seed_network():
    """Build the seed odd-even transposition network."""
    return _odd_even_transposition_network()

def _random_neighbor(comps):
    """Generate a random neighbor by mutating the comparator list."""
    comps = list(comps)
    r = random.random()
    if r < 0.3 and len(comps) > 1:
        # Remove a random comparator
        idx = random.randint(0, len(comps) - 1)
        comps.pop(idx)
    elif r < 0.6:
        # Add a random comparator
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(comps))
        comps.insert(pos, (i, j))
    elif r < 0.9:
        # Move a comparator to a different position
        if len(comps) > 1:
            idx = random.randint(0, len(comps) - 1)
            comp = comps.pop(idx)
            new_pos = random.randint(0, len(comps))
            comps.insert(new_pos, comp)
    else:
        # Swap two adjacent comparators
        if len(comps) > 1:
            idx = random.randint(0, len(comps) - 2)
            comps[idx], comps[idx + 1] = comps[idx + 1], comps[idx]
    return comps

def _local_search(seed_comps, max_iterations=50000):
    """Local search to minimize comparator count while maintaining correctness."""
    best = list(seed_comps)
    best_len = len(best)
    
    start_time = time.time()
    iter_count = 0
    
    while iter_count < max_iterations:
        if time.time() > DEADLINE - 5:  # Leave 5s safety margin
            break
        
        # Try to remove a random comparator
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            candidate = list(best)
            candidate.pop(idx)
            
            if _verify_network_fast(candidate):
                best = candidate
                best_len = len(best)
        
        # Also try moving a comparator to enable removal elsewhere
        if len(best) > 2:
            idx = random.randint(0, len(best) - 1)
            candidate = list(best)
            comp = candidate.pop(idx)
            new_pos = random.randint(0, len(candidate))
            candidate.insert(new_pos, comp)
            
            # Try removing another comparator
            if len(candidate) > 1:
                idx2 = random.randint(0, len(candidate) - 1)
                candidate2 = list(candidate)
                candidate2.pop(idx2)
                if _verify_network_fast(candidate2):
                    best = candidate2
                    best_len = len(best)
        
        iter_count += 1
    
    return best

def construct():
    """Construct a sorting network for 17 wires."""
    start_time = time.time()
    
    # Start with the seed network
    seed = _build_seed_network()
    best = list(seed)
    
    # Verify seed
    if not _verify_network_fast(best):
        # Shouldn't happen, but just in case
        return best
    
    # Local search to reduce comparator count
    best = _local_search(best, max_iterations=100000)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return best
    
    # Try multiple restarts with different random seeds
    for restart in range(5):
        if time.time() > DEADLINE - 10:
            break
        
        # Create a perturbed version of the best
        perturbed = list(best)
        for _ in range(random.randint(3, 10)):
            perturbed = _random_neighbor(perturbed)
        
        # Verify perturbed version
        if _verify_network_fast(perturbed):
            # Local search from this starting point
            candidate = _local_search(perturbed, max_iterations=20000)
            if len(candidate) < len(best):
                best = candidate
        
        # Also try adding some random comparators then optimizing
        if time.time() > DEADLINE - 10:
            break
        
        perturbed2 = list(best)
        for _ in range(random.randint(1, 3)):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(perturbed2))
            perturbed2.insert(pos, (i, j))
        
        if _verify_network_fast(perturbed2):
            candidate = _local_search(perturbed2, max_iterations=20000)
            if len(candidate) < len(best):
                best = candidate
    
    return best

# EVOLVE-BLOCK-END
