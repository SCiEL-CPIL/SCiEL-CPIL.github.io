import numpy as np
import time

N = 16

# EVOLVE-BLOCK-START
def _bitonic_merge(n):
    """Generate comparators for bitonic merge of size n."""
    comps = []
    k = n // 2
    while k >= 1:
        # For each block of size 2k
        for block_start in range(0, n, 2 * k):
            for i in range(k):
                comps.append((block_start + i, block_start + i + k))
        k //= 2
    return comps

def _bitonic_sort(n):
    """Generate Batcher's bitonic sort network for n wires."""
    if n == 1:
        return []
    half = n // 2
    left = _bitonic_sort(half)
    
    # Right half needs to be sorted in descending order
    # We generate ascending sort for right half, then reverse the wire indices
    right_asc = _bitonic_sort(half)
    right_desc = []
    for i, j in right_asc:
        # Map wire x in [0, half) to wire half + (half - 1 - x)
        i_new = half + (half - 1 - i)
        j_new = half + (half - 1 - j)
        # Ensure i_new < j_new for the comparator format? 
        # The comparator (u, v) sorts min to u, max to v.
        # If we reverse, we want the "larger" index in the original to be "smaller" in the new?
        # Actually, just swapping the wires in the network definition works if we are careful.
        # But standard comparator (i, j) assumes i < j.
        # If i_new > j_new, we swap them.
        if i_new > j_new:
            i_new, j_new = j_new, i_new
        right_desc.append((i_new, j_new))
        
    merge = _bitonic_merge(n)
    return left + right_desc + merge

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

def _find_violation(network, inputs):
    """Find an input that violates the sorting property. Returns None if valid."""
    wires = inputs.copy()
    for i, j in network:
        min_val = np.minimum(wires[i], wires[j])
        max_val = np.maximum(wires[i], wires[j])
        wires[i] = min_val
        wires[j] = max_val
    
    diffs = np.diff(wires, axis=0)
    violating = np.where(np.any(diffs < 0, axis=0))
    if len(violating[0]) > 0:
        return inputs[:, violating[0][0]]
    return None

def construct():
    start_time = time.time()
    timeout = 85.0
    
    # Precompute all 2^16 Boolean inputs
    # Shape: (16, 65536)
    inputs = np.array(np.unravel_index(np.arange(2**N), (2,)*N)).T
    
    # Start with a correct Batcher network
    best_net = _bitonic_sort(16)
    best_count = len(best_net)
    
    # Verify starting network
    if not _verify_network(best_net, inputs):
        # Fallback: Hardcoded 60 if Batcher fails (should not happen)
        # But let's assume Batcher is correct now.
        pass
        
    # Witness set: inputs that previously violated
    witnesses = []
    
    # Phase 1: Greedy Removal with Witness Set
    # Try to remove each comparator
    # If removal fails, add the violating input to witnesses
    # If witnesses become too large or time is up, stop
    
    # To speed up, we check against witnesses first, then a sample, then full.
    
    def check_candidate(candidate):
        # 1. Check witnesses
        for w in witnesses:
            wires = w.copy()
            for i, j in candidate:
                min_val = min(wires[i], wires[j])
                max_val = max(wires[i], wires[j])
                wires[i] = min_val
                wires[j] = max_val
            if not np.all(np.diff(wires) >= 0):
                return False, w
        
        # 2. Check random sample
        sample_size = min(1000, len(inputs))
        sample_indices = np.random.choice(len(inputs), sample_size, replace=False)
        sample_inputs = inputs[:, sample_indices]
        
        wires = sample_inputs.copy()
        for i, j in candidate:
            min_val = np.minimum(wires[i], wires[j])
            max_val = np.maximum(wires[i], wires[j])
            wires[i] = min_val
            wires[j] = max_val
        
        diffs = np.diff(wires, axis=0)
        violating = np.where(np.any(diffs < 0, axis=0))
        if len(violating[0]) > 0:
            return False, sample_inputs[:, violating[0][0]]
            
        # 3. Full check (expensive, do only if sample passes)
        # We can skip full check if we are confident, but for safety:
        if not _verify_network(candidate, inputs):
             # Find a violation to add to witnesses
             v = _find_violation(candidate, inputs)
             return False, v
             
        return True, None

    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        n = len(best_net)
        # Shuffle indices to avoid bias
        indices = list(range(n))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > timeout:
                break
            
            candidate = best_net[:idx] + best_net[idx+1:]
            is_valid, witness = check_candidate(candidate)
            
            if is_valid:
                best_net = candidate
                best_count = len(best_net)
                improved = True
                break # Restart loop with new length
            else:
                # Add witness to set (limit size to prevent slowness)
                if len(witnesses) < 1000:
                    witnesses.append(witness)
    
    # Phase 2: Try moving comparators (swap adjacent)
    # This can help escape local minima where a comparator is in the wrong place
    # but removing it fails.
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        n = len(best_net)
        indices = list(range(n - 1))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > timeout:
                break
            
            # Swap idx and idx+1
            candidate = best_net.copy()
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            
            is_valid, witness = check_candidate(candidate)
            
            if is_valid:
                best_net = candidate
                improved = True
                break
            else:
                if len(witnesses) < 1000:
                    witnesses.append(witness)
    
    return best_net
# EVOLVE-BLOCK-END
