import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 55  # Safety margin before 60s kill

def _depth(net):
    """Compute the parallel depth of a comparator network."""
    if not net:
        return 0
    layer = [0] * N
    for i, j in net:
        layer[i] = max(layer[i], layer[j]) + 1
    return max(layer) if layer else 0

def _is_sorted_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Uses the 0-1 principle with vectorized NumPy operations.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return True
    
    # Generate all 2^N boolean inputs as a matrix
    all_masks = np.arange(1 << N, dtype=np.int32)
    wires = np.zeros((1 << N, N), dtype=np.int8)
    for i in range(N):
        wires[:, i] = (all_masks >> i) & 1
    
    # Apply comparators using vectorized min/max
    for i, j in net:
        wi = wires[:, i]
        wj = wires[:, j]
        min_val = np.minimum(wi, wj)
        max_val = np.maximum(wi, wj)
        wires[:, i] = min_val
        wires[:, j] = max_val
    
    # Check if sorted: wires[:, k] <= wires[:, k+1] for all k
    if np.any(wires[:, :-1] > wires[:, 1:]):
        return False
    return True

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with odd-even transposition network (153 comparators for N=18)
    net = _odd_even_transposition()
    best_net = net[:]
    best_count = len(net)
    
    # Phase 1: Greedy removal - try removing comparators one at a time
    # Iterate until no more removals are possible or time runs out
    changed = True
    while changed:
        if time.time() > deadline - 3:
            break
        changed = False
        
        # Try removing from end to start (often less critical comparators are at the end)
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 3:
                return best_net
            
            new_net = net[:idx] + net[idx+1:]
            
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break  # Restart with new network
        
        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 3:
                    return best_net
                
                new_net = net[:idx] + net[idx+1:]
                
                if _is_sorted_network(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
    
    # Phase 2: Local search - try replacing comparators to enable further removal
    if time.time() < deadline - 5:
        attempts = 0
        max_attempts = 500
        while time.time() < deadline - 3 and attempts < max_attempts:
            attempts += 1
            if len(net) < 2:
                break
            
            # Pick a random comparator to replace
            idx = random.randint(0, len(net) - 1)
            i, j = net[idx]
            
            # Try a random different comparator
            ni = random.randint(0, N - 2)
            nj = random.randint(ni + 1, N - 1)
            if (ni, nj) == (i, j):
                continue
            
            new_net = net[:]
            new_net[idx] = (ni, nj)
            
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                attempts = 0  # Reset on success
                # After a successful replacement, try to remove more
                changed = True
                while changed and time.time() < deadline - 3:
                    changed = False
                    for ridx in range(len(net) - 1, -1, -1):
                        if time.time() > deadline - 3:
                            break
                        new_net2 = net[:ridx] + net[ridx+1:]
                        if _is_sorted_network(new_net2):
                            net = new_net2
                            best_net = net[:]
                            best_count = len(net)
                            changed = True
                            break
    
    # Final validation of best_net
    if not _is_sorted_network(best_net):
        # Fallback to original odd-even transposition
        best_net = _odd_even_transposition()
    
    return best_net
