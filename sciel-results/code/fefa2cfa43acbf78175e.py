import time
import numpy as np
import random

N = 18
TIME_LIMIT = 85  # Safety margin from 90s hard limit

# Pre-computed bitsets for fast verification
# Each bitset is an array of 4096 uint64s representing 2^18 inputs
_INITIAL_BITSETS = None

def _init_bitsets():
    """
    Initialize bitsets for fast Boolean verification.
    bitset[w] is a numpy array of uint64s.
    bit b of bitset[w][k] is 1 if the (k*64 + b)-th input has a 1 on wire w.
    """
    global _INITIAL_BITSETS
    if _INITIAL_BITSETS is not None:
        return _INITIAL_BITSETS
    
    num_blocks = 2**N // 64
    bitsets = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    for w in range(N):
        for b in range(num_blocks):
            base = b * 64
            mask = 0
            if w < 6:
                # Pattern repeats within the block
                # Bit w is 1 for indices in [base + half, base + period) within the 64-block?
                # No, the pattern is global.
                # For a specific block b, the indices are base...base+63.
                # We need to check (idx >> w) & 1 for each bit position.
                # This is O(64) per block. 18 * 4096 * 64 ~ 4.7M ops. Fast enough.
                for bit in range(64):
                    idx = base + bit
                    if (idx >> w) & 1:
                        mask |= (1 << bit)
            else:
                # For w >= 6, the bit is constant within a 64-block because 2^w >= 64.
                # Wait, 2^6 = 64. So for w=6, the bit flips every 64 numbers.
                # So within a block of 64, it is constant.
                if (base >> w) & 1:
                    mask = (1 << 64) - 1
                else:
                    mask = 0
            
            bitsets[w][b] = mask
            
    _INITIAL_BITSETS = bitsets
    return _INITIAL_BITSETS

def _verify(comparators):
    """
    Verify if the network sorts all inputs using pre-computed bitsets.
    """
    # Copy initial bitsets
    # We use a workspace to avoid modifying the global initial bitsets
    # But copying 18 * 4096 uint64s is cheap.
    # To be even faster, we can use a single flat array and views, or just lists of arrays.
    # Let's use a list of arrays.
    
    # Optimization: Use a pre-allocated workspace?
    # Since verification is called many times, allocation overhead might matter.
    # But numpy array creation is fast.
    
    bitsets = [arr.copy() for arr in _INITIAL_BITSETS]
    
    for i, j in comparators:
        bi = bitsets[i]
        bj = bitsets[j]
        
        # min = bi & bj, max = bi | bj
        # We need to update bitsets[i] and bitsets[j]
        # Note: bi and bj are views/copies, so we can modify them in place if we are careful.
        # But bi & bj creates a new array.
        
        min_vals = bi & bj
        max_vals = bi | bj
        
        # Update in place
        bitsets[i][:] = min_vals
        bitsets[j][:] = max_vals
        
    # Check sortedness
    # For all k, bitsets[k] must be a subset of bitsets[k+1]
    # i.e., bitsets[k] & ~bitsets[k+1] == 0
    
    for k in range(N - 1):
        if np.any(bitsets[k] & ~bitsets[k + 1]):
            return False
            
    return True

def _greedy_reduce(comparators, deadline):
    """
    Greedy reduction: try removing each comparator, keep if valid.
    Multiple passes until no more removals or deadline hit.
    """
    current = list(comparators)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(current):
            if time.time() > deadline:
                return current
            
            saved = current[i]
            current.pop(i)
            
            if _verify(current):
                improved = True
            else:
                current.insert(i, saved)
                i += 1
                
    return current

def _odd_even_transposition():
    """Standard odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _batcher_odd_even_merge(n):
    """
    Batcher Odd-Even Merge for n wires (n must be power of 2).
    Assumes inputs are on wires 0..n-1.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    comps = []
    
    # Recursively merge even and odd parts
    # Even part: wires 0, 2, 4... -> map to 0, 1, 2...
    # Odd part: wires 1, 3, 5... -> map to 0, 1, 2...
    
    # Merge A (even indices) and B (odd indices)
    # This is tricky to map back to original indices.
    # Standard implementation:
    # 1. Split into A (even) and B (odd).
    # 2. Recursively merge A and B? No.
    
    # Correct Recursive Definition for O-E Merge (n, m) where n=m=power of 2:
    # Let k = n/2.
    # 1. Merge A[0..k-1] and B[0..k-1]? No.
    
    # Let's use the standard definition from Batcher 1968.
    # O-E Merge(n):
    #   if n == 2: return [(0,1)]
    #   k = n/2
    #   M = O-E Merge(k)  # for the first k wires? No.
    
    # Actually, it's easier to implement Batcher Mergesort for powers of 2 and then adapt.
    # Or just use the known comparator list for 16 wires and merge 2?
    
    # Given the complexity and time, and since the seed OET is 153,
    # and the target is ~77, a better seed is crucial.
    
    # Let's implement a correct Bitonic Sort for 16 wires and use a merge for 18?
    # Or just use a pre-computed good network for 18?
    # Since I can't look it up, I will implement a generic Batcher Mergesort for N=18.
    
    # Batcher Mergesort for arbitrary N:
    # 1. Split into N1, N2.
    # 2. Recursively sort N1, N2.
    # 3. Merge N1, N2 using O-E Merge.
    
    pass # Placeholder, see _batcher_mergesort below

def _batcher_mergesort(n):
    """
    Batcher Odd-Even Mergesort for n wires.
    """
    if n <= 1:
        return []
    
    # Split
    k = n // 2
    left = _batcher_mergesort(k)
    right = _batcher_mergesort(n - k)
    
    # Shift right
    right = [(i + k, j + k) for i, j in right]
    
    # Merge
    merge = _odd_even_merge(k, n - k)
    
    return left + right + merge

def _odd_even_merge(n, m):
    """
    Batcher Odd-Even Merge for two sorted sequences of length n and m.
    Wires 0..n-1 contain sequence A.
    Wires n..n+m-1 contain sequence B.
    """
    if m == 0:
        return []
    if m == 1:
        # Merge n and 1
        # Comparators: (2*i, 2*i+1) for i in range(n)?
        # Let's check n=1, m=1: (0,1).
        # n=2, m=1: Wires 0,1 (A), 2 (B).
        # O-E Merge(2,1):
        # Split A into A0 (0), A1 (1). B is B0 (2).
        # Merge A0, B0? No.
        
        # Reference implementation for O-E Merge(n, m) with n>=m:
        # 1. Split A into A_even, A_odd.
        # 2. Split B into B_even, B_odd.
        # 3. Merge A_even and B_even.
        # 4. Merge A_odd and B_odd.
        # 5. Merge the results?
        
        # This is getting complex. Let's use a simpler, valid, dense network as seed.
        # The OET is 153.
        # A simple "Shell Sort" network might be better?
        # Or just return OET if this fails.
        
        # For the sake of having a valid seed that is better than OET,
        # I will use a pre-computed list for 18 wires if I had one.
        # Since I don't, I will fall back to OET in the construct function.
        # But wait, the inspiration 1 used a specific strategy.
        
        # Let's try to implement the merge correctly for powers of 2 first, then adapt.
        # For now, return [] to indicate "use fallback".
        return []

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Initialize bitsets
    _init_bitsets()
    
    # Seed Network: Odd-Even Transposition
    best_network = _odd_even_transposition()
    best_network = _greedy_reduce(best_network, deadline)
    best_len = len(best_network)
    
    current_network = list(best_network)
    
    # Simulated Annealing
    T = 2.0
    cooling_rate = 0.995
    min_T = 0.05
    
    while time.time() < deadline - 5 and T > min_T:
        current_network = _sa_step(current_network, T, deadline)
        
        if len(current_network) < best_len:
            best_network = list(current_network)
            best_len = len(best_network)
            
        T *= cooling_rate
        
    # Final reduction
    best_network = _greedy_reduce(best_network, deadline)
    
    return best_network

def _sa_step(current, T, deadline):
    """
    One step of Simulated Annealing.
    """
    neighbor = list(current)
    move_type = random.random()
    
    if move_type < 0.35:
        # Remove
        if len(neighbor) > 0:
            idx = random.randint(0, len(neighbor) - 1)
            neighbor.pop(idx)
    elif move_type < 0.55:
        # Slide
        if len(neighbor) > 0:
            idx = random.randint(0, len(neighbor) - 1)
            i, j = neighbor[idx]
            if random.random() < 0.5:
                if i + 1 < N - 1 and j + 1 < N:
                    ni, nj = i + 1, j + 1
                    if ni < nj:
                        neighbor[idx] = (ni, nj)
            else:
                if i - 1 >= 0 and j - 1 > 0:
                    ni, nj = i - 1, j - 1
                    if ni < nj:
                        neighbor[idx] = (ni, nj)
    elif move_type < 0.70:
        # Replace
        if len(neighbor) > 0:
            idx = random.randint(0, len(neighbor) - 1)
            dist = random.randint(1, 3) if random.random() < 0.6 else random.randint(1, N - 1)
            a = random.randint(0, N - 1 - dist)
            b = a + dist
            neighbor[idx] = (a, b)
    elif move_type < 0.80:
        # Move
        if len(neighbor) > 1:
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    else:
        # Add
        a, b = random.sample(range(N), 2)
        if a > b: a, b = b, a
        idx = random.randint(0, len(neighbor))
        neighbor.insert(idx, (a, b))
        
    # Verify
    if _verify(neighbor):
        # Accept if better or with probability
        delta = len(neighbor) - len(current)
        if delta <= 0:
            return neighbor
        else:
            prob = np.exp(-delta / T)
            if random.random() < prob:
                return neighbor
    else:
        # Invalid network, reject
        return current

    return current
