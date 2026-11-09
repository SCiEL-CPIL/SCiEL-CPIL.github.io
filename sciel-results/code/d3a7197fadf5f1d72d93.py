"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

N = 18
MAX_COMPARATORS = 4096
TIME_LIMIT = 85.0  # Safety margin before 90s kill

# EVOLVE-BLOCK-START
def _verify_sorts_fast(nets, n_wires, timeout=None):
    """Verify that a sorting network sorts all 2^n Boolean inputs using vectorized operations."""
    n = n_wires
    total = 1 << n
    
    start_time = time.time()
    
    # Create array of all 2^n inputs
    # Use uint32 to be safe for n up to 32
    inputs = np.arange(total, dtype=np.uint32)
    
    # Apply each comparator
    for (i, j) in nets:
        bi = (inputs >> i) & 1
        bj = (inputs >> j) & 1
        # If bi > bj, we need to swap: clear bit i, set bit j
        mask = (bi > bj).astype(np.uint32)
        inputs = (inputs & ~(mask << i)) | (mask << j)
        
        if timeout and time.time() - start_time > timeout:
            return None  # Timeout
    
    # Check if sorted: all 0s come before all 1s
    # x is sorted iff x = 0 or x = (1<<n)-1 or x has form 0...01...1
    # i.e., the complement y = ~x & ((1<<n)-1) should be of form (1<<k)-1
    # which means y & (y+1) == 0
    y = (~inputs) & ((1 << n) - 1)
    # Check y & (y+1) == 0 for all
    result = (y & (y + 1)) == 0
    return bool(np.all(result))

def _depth(nets, n_wires):
    """Compute parallel depth of a sorting network."""
    depth = [0] * n_wires
    for (i, j) in nets:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0

def _odd_even_transposition(n):
    """Generate odd-even transposition sorting network."""
    nets = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                nets.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                nets.append((i, i + 1))
    return nets

def _bitonic_sort(n):
    """Generate a bitonic sorting network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    # Recursively sort halves
    left = _bitonic_sort(half)
    right = [((i + half), (j + half)) for (i, j) in left]
    # Merge the two sorted halves
    merge = _bitonic_merge(half)
    return left + right + merge

def _bitonic_merge(n):
    """Bitonic merge for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    half = n // 2
    # Compare elements that are half apart
    comps = []
    for i in range(half):
        comps.append((i, i + half))
    # Recursively merge halves
    left = _bitonic_merge(half)
    right = [((i + half), (j + half)) for (i, j) in left]
    return comps + left + right

def _generate_bitonic_for_n(n):
    """Generate bitonic sort network for arbitrary n by padding to next power of 2."""
    if n <= 1:
        return []
    # Find next power of 2
    p2 = 1
    while p2 < n:
        p2 *= 2
    # Generate bitonic sort for p2 wires
    full_net = _bitonic_sort(p2)
    # Remove comparators involving wires >= n
    filtered = [(i, j) for (i, j) in full_net if i < n and j < n]
    return filtered

def _akamai_network(n):
    """
    Generate a good starting network using a combination of approaches.
    For n=18, we can use a modified bitonic or other known constructions.
    """
    # Start with bitonic for 16 and add 2 more wires
    # Or use a different approach
    
    # Let's try a hybrid: bitonic for the first 16, then handle wires 16,17
    base = _generate_bitonic_for_n(16)
    
    # Add comparators to incorporate wires 16 and 17
    # First, sort the 16-wire network
    # Then merge in wire 16
    extra = []
    for i in range(16):
        extra.append((i, 16))
    # Now we have 17 wires sorted
    # Then merge in wire 17
    for i in range(17):
        extra.append((i, 17))
    
    # This is not optimal but gives a valid network
    # The total is 57 (for 16) + 16 + 17 = 90 comparators
    # But we can do better with more sophisticated merging
    
    # Alternative: use a known good network structure
    # For now, return the simple construction
    return base + extra

def construct():
    start_time = time.time()
    
    # Generate candidate networks
    candidates = []
    
    # Odd-even transposition (baseline)
    oet = _odd_even_transposition(N)
    candidates.append(oet)
    
    # Bitonic for 18
    bitonic = _generate_bitonic_for_n(N)
    candidates.append(bitonic)
    
    # Akamai-style hybrid
    akamai = _akamai_network(N)
    candidates.append(akamai)
    
    # Verify and find the best valid network
    best_net = None
    best_count = float('inf')
    
    for net in candidates:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        is_valid = _verify_sorts_fast(net, N, timeout=5.0)
        if is_valid:
            if len(net) < best_count:
                best_count = len(net)
                best_net = list(net)
    
    if best_net is None:
        # Fallback to odd-even transposition (should always be valid)
        best_net = oet
        best_count = len(best_net)
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 15:
        return best_net
    
    # Local search: try to remove comparators
    current_net = list(best_net)
    current_count = len(current_net)
    
    # Try removing each comparator in random order
    random.seed(42)
    remove_order = list(range(len(current_net)))
    random.shuffle(remove_order)
    
    for idx in remove_order:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        if idx >= len(current_net):
            continue
        
        comp = current_net[idx]
        # Try removing this comparator
        new_net = current_net[:idx] + current_net[idx+1:]
        
        if len(new_net) < current_count - 1:
            continue
        
        is_valid = _verify_sorts_fast(new_net, N, timeout=3.0)
        if is_valid:
            current_net = new_net
            current_count -= 1
            # After removal, re-shuffle remaining indices
            remove_order = list(range(len(current_net)))
            random.shuffle(remove_order)
    
    # Try swapping adjacent comparators to enable further removals
    for _ in range(50):  # Limited iterations
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        if len(current_net) < 2:
            break
        
        # Try swapping two random adjacent comparators
        pos1 = random.randint(0, len(current_net) - 2)
        pos2 = pos1 + 1
        
        new_net = list(current_net)
        new_net[pos1], new_net[pos2] = new_net[pos2], new_net[pos1]
        
        is_valid = _verify_sorts_fast(new_net, N, timeout=2.0)
        if is_valid:
            current_net = new_net
    
    # Final removal pass after swaps
    remove_order = list(range(len(current_net)))
    random.shuffle(remove_order)
    
    for idx in remove_order:
        if time.time() - start_time > TIME_LIMIT - 2:
            break
        if idx >= len(current_net):
            continue
        
        comp = current_net[idx]
        new_net = current_net[:idx] + current_net[idx+1:]
        
        if len(new_net) < current_count - 1:
            continue
        
        is_valid = _verify_sorts_fast(new_net, N, timeout=2.0)
        if is_valid:
            current_net = new_net
            current_count -= 1
            remove_order = list(range(len(current_net)))
            random.shuffle(remove_order)
    
    if len(current_net) < best_count:
        best_net = current_net
        best_count = len(best_net)
    
    return best_net
# EVOLVE-BLOCK-END
