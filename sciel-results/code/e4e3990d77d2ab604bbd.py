import numpy as np
import time

N = 17
DEADLINE = 85.0 # 5s safety margin

# Precompute shifts for bit unpacking
SHIFTS = np.arange(N - 1, -1, -1, dtype=np.uint32)
# Precompute indices
INDICES = np.arange(1 << N, dtype=np.uint32)

def _verify(net):
    """
    Verifies if the given network sorts all 2^N inputs.
    Returns True if valid, False otherwise.
    """
    # Unpack bits into a (2^N, N) array of uint8
    # Using broadcasting: INDICES[:, None] >> SHIFTS
    # This creates a (131072, 17) array
    wires = (INDICES[:, None] >> SHIFTS) & 1
    
    # Apply comparators
    for i, j in net:
        # min to i, max to j
        min_v = np.minimum(wires[:, i], wires[:, j])
        max_v = np.maximum(wires[:, i], wires[:, j])
        wires[:, i] = min_v
        wires[:, j] = max_v
        
    # Check if sorted: w[k] <= w[k+1] for all k
    # left = wires[:, :-1]
    # right = wires[:, 1:]
    # return not np.any(left > right)
    
    # Even faster:
    # Check if any 1 is followed by 0
    # diff = np.diff(wires, axis=1)
    # return not np.any(diff < 0)
    
    # Using subtraction to avoid diff overhead?
    # left > right
    return not np.any(wires[:, :-1] > wires[:, 1:])

def _generate_seed():
    """
    Generates an Odd-Even Transposition network for N wires.
    """
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def construct():
    """
    Constructs a size-efficient sorting network for N=17 wires.
    Uses local search (remove redundant comparators) starting from
    an Odd-Even Transposition network.
    """
    start_time = time.time()
    
    # 1. Generate Seed
    net = _generate_seed()
    
    # 2. Verify Seed (should be valid)
    if not _verify(net):
        # Fallback: return seed if it's invalid (should not happen)
        return net

    best_net = net[:]
    best_size = len(net)
    
    # 3. Local Search Loop
    # We try to remove each comparator. If valid, we keep it removed.
    # We restart the scan from the beginning after each removal.
    
    while time.time() < DEADLINE:
        improved = False
        for i in range(len(net)):
            if time.time() > DEADLINE:
                break
            
            # Try removing net[i]
            saved_comp = net[i]
            net.pop(i)
            
            # Verify
            if _verify(net):
                # Valid! Keep removed.
                improved = True
                # Update best if smaller (it is smaller by 1)
                best_net = net[:]
                best_size = len(net)
                
                # Restart scan from 0 because indices shifted
                break
            else:
                # Invalid, restore
                net.insert(i, saved_comp)
        
        if not improved:
            # No more removals possible in this pass
            break
            
    return best_net
