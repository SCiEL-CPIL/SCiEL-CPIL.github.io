import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _build_oet(n):
    """Build Odd-Even Transposition network for n wires."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def _verify_numpy(network):
    """Verify network using NumPy vectorization. Returns True if valid."""
    if not network:
        return True
        
    # All 2^16 masks
    masks = np.arange(1 << N, dtype=np.uint32)
    
    # Apply comparators
    for i, j in network:
        mask_i = np.uint32(1 << i)
        mask_j = np.uint32(1 << j)
        
        # Condition: bit i is 0 AND bit j is 1
        cond = ((masks & mask_i) == 0) & ((masks & mask_j) != 0)
        
        if np.any(cond):
            swap_mask = mask_i | mask_j
            masks[cond] ^= swap_mask
            
    # Check if all masks are sorted
    # Sorted mask m: m is 0 or (m | (m-1)) == 2^N - 1
    all_ones = np.uint32((1 << N) - 1)
    
    is_zero = masks == 0
    
    # For non-zero masks, check (m | (m-1)) == all_ones
    # m-1 underflow for 0 is handled by is_zero mask, but we compute it for all.
    # For m=0, m-1 is 2^32-1. 0 | (2^32-1) = 2^32-1.
    # all_ones is 2^16-1. So (m | (m-1)) != all_ones for m=0.
    # So we just OR with is_zero.
    
    m_minus_1 = masks - 1
    sorted_check = (masks | m_minus_1) == all_ones
    
    valid = is_zero | sorted_check
    
    return bool(np.all(valid))

def _remove_redundant_comparators(network, deadline):
    """Greedy removal of redundant comparators using NumPy verifier."""
    if not network:
        return network
    
    if not _verify_numpy(network):
        return network

    changed = True
    # Try removing from end to start, or random order?
    # Random order might find better reductions faster if there are multiple valid reductions.
    # But greedy from end is standard.
    # Let's try random order to escape local minima of "this specific comparator is redundant".
    indices = list(range(len(network)))
    random.shuffle(indices)
    
    for idx in indices:
        if time.time() > deadline:
            break
        if idx >= len(network):
            continue
            
        candidate = network[:idx] + network[idx+1:]
        if _verify_numpy(candidate):
            network = candidate
            # Restart? Or continue?
            # If we remove one, indices shift.
            # It's safer to restart the shuffle or just break and restart the whole function.
            # Given the small size (120 -> 60), restarting is fine.
            return _remove_redundant_comparators(network, deadline)
            
    return network

def construct():
    start_time = time.time()
    # Safety margin: 85s
    deadline = start_time + 85.0
    
    # Start with OET (120 comparators)
    network = _build_oet(N)
    
    # Prune
    network = _remove_redundant_comparators(network, deadline)
    
    # Optional: Local search to try to remove more?
    # Greedy pruning should get us to 60 (Batcher OMS size) or close to it.
    # If we are at 60, we can't remove any more (presumably).
    
    return network

# EVOLVE-BLOCK-END
