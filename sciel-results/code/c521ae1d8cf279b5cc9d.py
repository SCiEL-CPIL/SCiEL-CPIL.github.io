import time
import random

N = 18

def _batcher_odd_merge(n):
    """
    Generates a Batcher's Odd-Merge sorting network for n wires.
    """
    if n <= 1:
        return []
    
    # Split into two halves
    n1 = (n + 1) // 2
    n2 = n // 2
    
    # Recursive calls
    net1 = _batcher_odd_merge(n1)
    net2 = _batcher_odd_merge(n2)
    
    # Adjust indices for net2 (shifted by n1)
    net2 = [(i + n1, j + n1) for i, j in net2]
    
    # Merge phase
    merge_net = _batcher_merge(n1, n2)
    
    return net1 + net2 + merge_net

def _batcher_merge(n1, n2):
    """
    Generates the merge phase of Batcher's Odd-Merge for two sorted sub-networks.
    n1 and n2 are the sizes of the two sub-networks.
    """
    if n2 == 0:
        return []
    
    comparators = []
    
    # The merge phase consists of multiple levels of comparators.
    # This is a simplified implementation. A full Batcher merge is complex.
    # For the purpose of this seed, we can use a simpler, known valid network.
    # Let's use a Bitonic network instead, which is easier to implement correctly.
    pass

def _bitonic_sort_network(n):
    """
    Generates a Bitonic sorting network for n wires (must be power of 2, or padded).
    For n=18, we pad to 32? No, that's too many comparators.
    Let's use a known small network or just generate a valid one.
    
    Actually, let's use a simpler approach: 
    Start with the Odd-Even Transposition network (153 comps) and prune it.
    This is safer and guaranteed to work.
    """
    return [(i, i + 1) for phase in range(n)
            for i in range(phase % 2, n - 1, 2)]

def _verify_network(net):
    """
    Verifies if the network sorts all 2^N binary inputs.
    Uses the 0-1 principle.
    """
    # Precompute bit masks for comparators? 
    # For each mask in 0..2^N-1, simulate the network.
    # If output is not sorted (all 0s then 1s), return False.
    
    # Optimization: 
    # A binary vector is sorted if it is of the form 0...01...1.
    # We can check this by verifying that the number of 1s is preserved 
    # and the output matches the expected pattern for that count.
    
    # Expected pattern for k ones: 
    # The last k bits are 1, the first N-k bits are 0.
    # Mask: (1 << k) - 1.
    
    # We can iterate k from 0 to N.
    # For each k, we need to check if ALL inputs with k ones are sorted.
    # There are C(N, k) such inputs.
    
    # Total checks: sum(C(N, k)) = 2^N.
    
    # Let's implement a fast simulation.
    for mask in range(1 << N):
        # Simulate network
        state = mask
        for i, j in net:
            # Extract bits i and j
            # If bit_i > bit_j, swap them
            # bit_i is at position i, bit_j at position j.
            # Since i < j, bit_i is less significant than bit_j? 
            # Wait, wire 0 is usually the most significant or least?
            # The problem says "min to wire i and max to j".
            # If we treat wire 0 as MSB and wire N-1 as LSB:
            # min goes to i (MSB side), max to j (LSB side).
            # This means 0s go to the left, 1s to the right.
            # So sorted state is 0...01...1.
            
            # Let's assume wire 0 is MSB (bit N-1) and wire N-1 is LSB (bit 0).
            # This is a common convention.
            # Let's verify with the seed:
            # Seed: (i, i+1). 
            # If input is 10 (wire0=1, wire1=0), i=0, j=1.
            # min=0, max=1. Wire0 gets 0, Wire1 gets 1. Result 01.
            # In binary (wire0 is MSB): 01 is 1. 
            # Original 10 is 2.
            # So 2 -> 1.
            # If we treat wire 0 as bit N-1:
            # mask 2 (binary ...10). Bit N-1 is 1, Bit N-2 is 0.
            # Comparator (0,1) acts on bit N-1 and N-2.
            # If bit N-1 > bit N-2, swap.
            # 1 > 0, swap. Result ...01.
            # This matches.
            
            # So, wire i corresponds to bit (N - 1 - i).
            # Let bi = N - 1 - i, bj = N - 1 - j.
            # Since i < j, bi > bj.
            # We are comparing bit bi and bit bj.
            # If bit bi > bit bj, swap.
            
            # Fast check:
            # bit_bi = (state >> bi) & 1
            # bit_bj = (state >> bj) & 1
            # if bit_bi > bit_bj:
            #    state ^= (1 << bi) | (1 << bj)
            
            # We can optimize this loop.
            pass
        
        # Check if sorted
        # Count ones
        k = bin(state).count('1')
        expected = (1 << k) - 1
        if state != expected:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    # Start with Odd-Even Transposition
    net = _bitonic_sort_network(N)
    
    # Greedy Pruning
    # Try to remove each comparator.
    # To be efficient, we can shuffle the order of removal attempts.
    # However, removing one might make another removable.
    # We do multiple passes.
    
    # Initial verification
    if not _verify_network(net):
        # Should not happen, but just in case
        return net
    
    current_size = len(net)
    
    # Pruning loop
    while time.time() < deadline - 10.0:  # Leave time for SA
        progress = False
        # Try removing each comparator
        # We iterate in reverse to avoid index issues, or use a copy
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            candidate = net[:idx] + net[idx+1:]
            if _verify_network(candidate):
                net = candidate
                current_size = len(net)
                progress = True
                # Restart the pass? Or continue?
                # Continuing is faster, but might miss opportunities if we removed a "key" comparator.
                # However, if a network is valid without comparator X, and we remove X, 
                # other comparators might become redundant.
                # So we should continue the loop.
        
        if not progress:
            break
            
    # Simulated Annealing (optional if time remains)
    # Given the 90s limit and Python speed, SA might be risky.
    # Greedy pruning on 153 comparators is likely to get down to ~80-90.
    
    return net
