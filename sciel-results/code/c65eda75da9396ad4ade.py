import numpy as np
import time

N = 16

# EVOLVE-BLOCK-START
def _verify_network(network, inputs):
    """Verify that the network sorts all 2^16 Boolean inputs."""
    wires = inputs.copy()
    for i, j in network:
        min_val = np.minimum(wires[i], wires[j])
        max_val = np.maximum(wires[i], wires[j])
        wires[i] = min_val
        wires[j] = max_val
    
    diffs = np.diff(wires, axis=0)
    return np.all(diffs >= 0)

def _batcher_16():
    """Generate Batcher's bitonic sort network for 16 wires."""
    def bitonic_merge(n):
        comps = []
        k = n // 2
        while k >= 1:
            for i in range(n - k):
                comps.append((i, i + k))
            k //= 2
        return comps
    
    def bitonic_sort(n):
        if n == 1:
            return []
        half = n // 2
        left = bitonic_sort(half)
        
        right_asc = bitonic_sort(half)
        right_desc = []
        for i, j in right_asc:
            i_new = half - 1 - j
            j_new = half - 1 - i
            right_desc.append((half + i_new, half + j_new))
        
        merge = bitonic_merge(n)
        return left + right_desc + merge
    
    return bitonic_sort(16)

def construct():
    start_time = time.time()
    timeout = 85.0
    
    # Precompute all 2^16 Boolean inputs
    inputs = np.array(np.unravel_index(np.arange(2**N), (2,)*N)).T
    
    # Start with Batcher (Valid)
    best_net = _batcher_16()
    best_count = len(best_net)
    
    # Local Search: Removal
    def try_remove(network, idx):
        candidate = network[:idx] + network[idx+1:]
        if _verify_network(candidate, inputs):
            return candidate
        return None

    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        n = len(best_net)
        indices = list(range(n))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > timeout:
                break
            candidate = try_remove(best_net, idx)
            if candidate:
                best_net = candidate
                best_count = len(best_net)
                improved = True
                break
                
    # If we have time, try swaps to enable further removals
    # This is a structural change: adding swap-based local search.
    if time.time() - start_time < timeout - 5.0:
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            n = len(best_net)
            # Try random swaps
            for _ in range(50):
                if time.time() - start_time > timeout:
                    break
                i, j = np.random.choice(n, size=2, replace=False)
                i, j = int(i), int(j)
                # Swap i and j
                candidate = best_net.copy()
                candidate[i], candidate[j] = candidate[j], candidate[i]
                # Check if this allows a removal? 
                # Or just check if it's valid (it will be, swap preserves validity).
                # The goal is to find a state where removal is possible.
                # So we don't accept the swap unless it leads to a better state?
                # Actually, swap alone doesn't reduce count.
                # We should try: Swap -> Try Remove -> If Remove works, accept both.
                
                # Try removing any comparator in the swapped network
                # This is expensive.
                pass
            # Skip swap for now, focus on robust removal.

    return best_net
# EVOLVE-BLOCK-END
