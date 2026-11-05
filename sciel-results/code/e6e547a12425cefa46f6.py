import numpy as np
import time

N = 17
NUM_INPUTS = 1 << N
DEADLINE = 85  # seconds, safety margin

def _generate_batcher_odd_merge(n):
    """
    Generate Batcher's Odd-Merge sorting network for n wires.
    Returns a list of (i, j) comparators.
    """
    if n == 1:
        return []
    
    # Recursive step for Odd-Merge
    # 1. Split into two halves
    if n % 2 == 0:
        m1 = n // 2
        m2 = n // 2
    else:
        m1 = (n + 1) // 2
        m2 = (n - 1) // 2
        
    # 2. Recursively sort the two halves
    comps1 = _generate_batcher_odd_merge(m1)
    comps2 = _generate_batcher_odd_merge(m2)
    
    # Offset the second half's comparators by m1
    comps2_offset = [(i + m1, j + m1) for i, j in comps2]
    
    # 3. Odd-Even Merge the two sorted halves
    # The merge pattern depends on the sizes.
    # For simplicity and correctness, we can use a standard merge network generator.
    # However, implementing Odd-Even Merge correctly is tricky.
    
    # Alternative: Use a simpler, known-good small network.
    # Let's use a "Shell Sort" network or just stick to the transposition but start smaller?
    # Actually, let's just use the standard Odd-Even Transposition but cut it down?
    # No, Batcher is better.
    
    # Let's use a simpler approach: Generate a network using a known algorithm.
    # For N=17, a simple Batcher network is:
    # B(17) = Merge(B(9), B(8))
    # We need a helper to generate the merge comparators.
    
    # To avoid complex recursive merge logic errors, let's use a hardcoded small network 
    # or a very simple recursive construction that is guaranteed correct.
    
    # Actually, the simplest correct small network is the "Binary Tree" sort network?
    # No, that's large.
    
    # Let's use the following strategy:
    # 1. Generate a valid network using Odd-Even Transposition (136 comps).
    # 2. Aggressively prune it using the fast packed verifier.
    # The key is the FAST VERIFIER.
    
    # But starting from 136 is still large.
    # Let's try to generate a Batcher network correctly.
    
    # Here is a correct implementation of Batcher's Odd-Merge:
    
    def _merge(comps_a, comps_b, n_a, n_b, offset_a=0, offset_b=0):
        """
        Merge two sorted networks.
        comps_a sorts wires [offset_a, offset_a+n_a)
        comps_b sorts wires [offset_b, offset_b+n_b)
        """
        # This is complex to implement generically without bugs.
        # Let's fall back to a robust seed: 
        # Use a "Layered" network that is known to be small.
        # For N=17, the lower bound is 63.
        # A simple heuristic: Use the first 64 comparators of a larger network? No.
        
        # Let's just use the Odd-Even Transposition but optimize the search heavily.
        # The bottleneck is verification. If we make verification 10x faster, 
        # we can do 10x more removals.
        
        pass

    # Given the complexity of implementing a bug-free Batcher generator from scratch in this context,
    # and the fact that the previous program scored 1.0 (meaning it returned the seed or close to it),
    # the main issue is likely that it couldn't remove many comparators due to time.
    
    # Let's focus on the Packed Verification speedup.
    
    # For the seed, let's use a slightly optimized Odd-Even Transposition 
    # or just the standard one, but rely on the faster verifier to strip it down.
    
    # Actually, I can hardcode a small valid network for N=17?
    # No, that's brittle.
    
    # Let's use a simple recursive Batcher implementation that is known to work.
    
    def _batcher(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        if n % 2 == 0:
            m = n // 2
            left = _batcher(m)
            right = [(i + m, j + m) for i, j in _batcher(m)]
        else:
            m1 = (n + 1) // 2
            m2 = (n - 1) // 2
            left = _batcher(m1)
            right = [(i + m1, j + m1) for i, j in _batcher(m2)]
            
        # Now we need to merge left (size m1) and right (size m2)
        # Odd-Even Merge Network
        # The merge network for two sorted sequences of lengths n1 and n2
        # consists of:
        # 1. Comparators between corresponding elements? No.
        
        # Standard Odd-Even Merge:
        # M(n1, n2)
        # If n1 == 0 or n2 == 0: return []
        # Split n1 into n1a (even) and n1b (odd)
        # Split n2 into n2a (even) and n2b (odd)
        
        # This is getting too complex to implement safely in a short block.
        
        # STRATEGY SHIFT:
        # Use the current seed (Odd-Even Transposition) but with a MUCH faster verifier.
        # The packed verifier will be significantly faster than the matrix one.
        
        # Let's return the standard OET for now, the verifier will prune it.
        return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

    return _batcher(n)

def _make_all_inputs_packed():
    """Create array of all 2^N inputs as uint64 array."""
    return np.arange(NUM_INPUTS, dtype=np.uint64)

def _is_sorting_network_packed(comparators, inputs):
    """Check if network sorts all inputs using packed representation."""
    # inputs is a uint64 array of length NUM_INPUTS
    for i, j in comparators:
        mask_i = np.uint64(1) << i
        mask_j = np.uint64(1) << j
        
        bit_i = (inputs & mask_i) != 0
        bit_j = (inputs & mask_j) != 0
        
        new_i = bit_i & bit_j
        new_j = bit_i | bit_j
        
        # Clear bits i and j
        inputs = inputs & ~(mask_i | mask_j)
        # Set new bits
        inputs = np.where(new_i, inputs | mask_i, inputs)
        inputs = np.where(new_j, inputs | mask_j, inputs)
        
    # Check if sorted
    # For each k in 0..N-2, if bit k is 1, bit k+1 must be 1.
    # Violation: bit k == 1 and bit k+1 == 0
    for k in range(N - 1):
        mask_k = np.uint64(1) << k
        mask_k1 = np.uint64(1) << (k + 1)
        
        bit_k = (inputs & mask_k) != 0
        bit_k1 = (inputs & mask_k1) != 0
        
        if np.any(bit_k & ~bit_k1):
            return False
            
    return True

def construct():
    start_time = time.time()
    
    # Base: Odd-even transposition
    comps = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    inputs = _make_all_inputs_packed()
    
    # Verify base is correct
    if not _is_sorting_network_packed(comps, inputs):
        return comps
    
    # Iteratively remove redundant comparators
    changed = True
    while changed and (time.time() - start_time) < DEADLINE - 2:
        changed = False
        for idx in range(len(comps)):
            if (time.time() - start_time) > DEADLINE - 2:
                break
            # Try removing idx
            new_comps = comps[:idx] + comps[idx+1:]
            if _is_sorting_network_packed(new_comps, inputs):
                comps = new_comps
                changed = True
                break  # Restart loop since indices changed
                
    return comps
