import time
import numpy as np

N = 18
_DEADLINE = 85.0  # Safety margin under 90s kill
_start_time = None

def _check_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_valid_network(comparators, n=N):
    """
    Fast vectorized check: does this network sort all 2^n boolean inputs?
    Returns True if valid.
    """
    if len(comparators) == 0:
        return n <= 1
    
    # Basic bounds check
    for i, j in comparators:
        if not (0 <= i < j < n):
            return False

    # Generate all 2^n inputs as a matrix of shape (2^n, n)
    # Use numpy for speed
    masks = np.arange(1 << n, dtype=np.uint8)
    wires = np.zeros((1 << n, n), dtype=np.uint8)
    for k in range(n):
        wires[:, k] = (masks >> k) & 1

    # Apply comparators
    for i, j in comparators:
        wi = wires[:, i]
        wj = wires[:, j]
        # If wi > wj, swap. Since they are 0/1, this means wi=1, wj=0.
        # swap: wi becomes 0, wj becomes 1
        # Equivalent: 
        # new_wi = min(wi, wj)
        # new_wj = max(wi, wj)
        # For 0/1: min is AND, max is OR
        new_wi = wi & wj
        new_wj = wi | wj
        wires[:, i] = new_wi
        wires[:, j] = new_wj

    # Check if all rows are sorted (non-decreasing)
    # A row is sorted if for all k, wires[k] <= wires[k+1]
    # This is equivalent to checking if the row equals its sorted version
    # Or simply: np.all(wires[:, :-1] <= wires[:, 1:])
    
    # Vectorized check: 
    # left = wires[:, :-1]
    # right = wires[:, 1:]
    # if any(left > right), invalid.
    
    if np.any(wires[:, :-1] > wires[:, 1:]):
        return False
        
    return True

def _get_known_networks():
    """
    Return a dictionary of known small sorting networks.
    Key: n, Value: list of (i, j)
    """
    networks = {}
    
    # n=1
    networks[1] = []
    
    # n=2
    networks[2] = [(0, 1)]
    
    # n=3 (Akers)
    networks[3] = [(0, 1), (1, 2), (0, 1)]
    
    # n=4 (Optimal: 5 comparators)
    networks[4] = [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)]
    
    # n=5 (Optimal: 9 comparators)
    # Reference: 0-1 principle verified
    networks[5] = [
        (0, 2), (1, 3), (2, 4),
        (0, 1), (2, 3),
        (1, 2), (3, 4),
        (1, 3), (2, 3)
    ]
    # Let's verify n=5 quickly in head or rely on standard knowns.
    # Actually, let's use a safer construction for n=5 if unsure, but 9 is known optimal.
    
    # n=6 (Optimal: 12 comparators)
    # A known 12-comparator network for 6 wires
    networks[6] = [
        (0, 2), (1, 3), (2, 4), (3, 5),
        (0, 1), (2, 3), (4, 5),
        (1, 2), (3, 4),
        (0, 1), (2, 3), (4, 5),
        (1, 2)
    ]
    # Wait, I should verify these are actually valid before using them as seeds.
    # If they are invalid, the merge will fail or produce invalid results.
    # To be safe, I will verify them at runtime and discard if invalid.
    
    return networks

def _merge_networks(net1, net2, offset1=0, offset2=0):
    """
    Merge two sorting networks into a larger one.
    net1 sorts wires [offset1, offset1+len1)
    net2 sorts wires [offset2, offset2+len2)
    This simple merge just concatenates. It does NOT guarantee the merged network
    sorts the combined set unless we add cross-comparisons.
    
    For a valid sorting network on the union, we need to ensure the two sorted
    sequences are merged.
    """
    comps = []
    for i, j in net1:
        comps.append((i + offset1, j + offset1))
    for i, j in net2:
        comps.append((i + offset2, j + offset2))
    return comps

def _construct_bitonic(n, offset=0):
    """
    Construct a bitonic sort network for n wires (n must be power of 2).
    This is a reliable, dense starting point.
    """
    comps = []
    if n <= 1:
        return comps
    
    def bitonic_merge(k, off, d):
        if k <= 1:
            return
        h = k // 2
        for i in range(h):
            a = off + i
            b = off + i + h
            if d == 1:
                comps.append((a, b))
            else:
                comps.append((b, a))
        bitonic_merge(h, off, d)
        bitonic_merge(h, off + h, d)
    
    def bitonic_sort(k, off, d):
        if k <= 1:
            return
        h = k // 2
        bitonic_sort(h, off, 1)
        bitonic_sort(h, off + h, -1)
        bitonic_merge(k, off, d)
    
    bitonic_sort(n, offset, 1)
    return comps

def _greedy_prune(comparators, time_limit=80.0):
    """
    Greedy removal of comparators.
    Uses the vectorized validator for speed.
    """
    comps = list(comparators)
    deadline = _start_time + time_limit
    
    # Try removing from the end first
    i = len(comps) - 1
    while i >= 0:
        if time.time() > deadline:
            break
        # Try removing comps[i]
        new_comps = comps[:i] + comps[i+1:]
        if _is_valid_network(new_comps):
            comps = new_comps
        i -= 1
        
    # Second pass: try removing from the beginning
    i = 0
    while i < len(comps):
        if time.time() > deadline:
            break
        new_comps = comps[:i] + comps[i+1:]
        if _is_valid_network(new_comps):
            comps = new_comps
        else:
            i += 1
            
    return comps

def construct():
    global _start_time
    _start_time = time.time()
    
    # Strategy: 
    # 1. Start with a dense, known-valid network (Bitonic on 32 wires, take first 18? No, Bitonic on 16 + merge 2 is better).
    # Actually, let's use a dense network on 18 wires.
    # A simple dense network: Odd-Even Transposition (153 comps).
    # But we can do better. 
    # Let's try constructing a network by merging known optimal small networks.
    
    # Step 1: Get known small networks and verify them.
    known = _get_known_networks()
    verified_small = {}
    for n, net in known.items():
        if _is_valid_network(net, n=n):
            verified_small[n] = net
        else:
            # If my hardcoded network is invalid, generate a bitonic one (dense but valid)
            if n <= 16:
                # Bitonic requires power of 2. If n is not power of 2, pad?
                # For n=3,4,5,6, bitonic is not directly applicable unless we treat it as part of a larger power of 2.
                # Let's just use Odd-Even Transposition for invalid ones as a fallback seed.
                pass
            
    # Let's create a dense seed network for 18 wires.
    # Option A: Odd-Even Transposition (153 comps)
    comps_oe = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps_oe.append((i, i + 1))
            
    # Option B: Bitonic on 16 wires + simple merge for 16-17
    # Bitonic 16 is 80 comps.
    comps_bitonic_16 = _construct_bitonic(16, 0)
    comps_bitonic_16.append((16, 17))
    # Merge 16 and 17 into 0-15.
    # A simple correct merge: compare 16 and 17 with all of 0-15 in a structured way?
    # Actually, just adding (i, 16) and (i, 17) for all i in 0-15 is 32 comps. Total 80+1+32=113.
    # Let's try this.
    for i in range(16):
        comps_bitonic_16.append((i, 16))
    for i in range(16):
        comps_bitonic_16.append((i, 17))
        
    # Verify Option B
    if not _is_valid_network(comps_bitonic_16):
        # Fallback to Option A if B fails
        comps_bitonic_16 = comps_oe
        
    # Start with the smaller valid one
    best_comps = comps_bitonic_16 if len(comps_bitonic_16) < len(comps_oe) else comps_oe
    best_comps = list(best_comps)
    
    # Try to improve using local search / greedy pruning.
    # The vectorized validator is much faster than the loop-based one.
    
    # Phase 1: Aggressive Greedy Pruning
    best_comps = _greedy_prune(best_comps, time_limit=_DEADLINE - _check_time() - 5)
    
    # Phase 2: Try to add/remove swaps? 
    # For now, just rely on the fast pruning.
    
    # Final check
    if not _is_valid_network(best_comps):
        return comps_oe
        
    return best_comps
