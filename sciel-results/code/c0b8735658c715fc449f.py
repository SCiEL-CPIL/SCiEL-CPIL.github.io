"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network for 16 wires using a BDD-based approach.
    
    We use a known optimal or near-optimal construction. Since a full BDD-based
    search within 60s for 16 wires is extremely complex, we use a well-known
    efficient construction and then attempt to prune redundant comparators
    using the zero-one principle with efficient bitset operations.
    
    The baseline odd-even transposition uses 120 comparators. We start from
    a more efficient known network (e.g., AKS-inspired or a known optimal
    network) and then prune.
    
    For 16 wires, a known good construction is the Batcher odd-even mergesort
    network which uses 57 comparators (or similar). Let's use a known efficient
    network and then prune.
    
    Actually, let's use a known optimal or near-optimal network. For 16 wires,
    the known optimal is 57 comparators (or close). Let me construct a Batcher
    odd-even mergesort network and then prune.
    """
    # Batcher odd-even mergesort network for 16 wires
    # This produces a sorting network with ~57 comparators
    
    def batcher_odd_even_mergesort(n):
        """Generate Batcher's odd-even mergesort network for n wires."""
        if n <= 1:
            return []
        
        comparators = []
        
        if n % 2 == 0:
            # Split into two halves
            half = n // 2
            # Recursively sort each half
            left = batcher_odd_even_mergesort(half)
            right = batcher_odd_even_mergesort(half)
            for (i, j) in left:
                comparators.append((i, j))
            for (i, j) in right:
                comparators.append((half + i, half + j))
            
            # Merge the two halves
            comparators.extend(merge_odd_even(half, half))
        else:
            # For odd n, handle differently
            # This case won't be hit for n=16
            pass
        
        return comparators
    
    def merge_odd_even(m, n):
        """Merge two sorted sequences of lengths m and n using odd-even merge."""
        comparators = []
        
        if m == 0:
            return []
        if n == 0:
            return []
        
        if m == 1 and n == 1:
            return [(0, 1)]
        
        if m == 1:
            # Merge 1 element with n elements
            for i in range(n - 1):
                comparators.append((i, i + 1))
            return comparators
        
        if n == 1:
            # Merge m elements with 1 element
            for i in range(m - 1, -1, -1):
                comparators.append((i, i + 1))
            return comparators
        
        # General case: odd-even merge
        # Split into even and odd positions
        even_m = m // 2
        odd_m = (m + 1) // 2
        even_n = n // 2
        odd_n = (n + 1) // 2
        
        # Recursively merge even positions and odd positions
        even_merge = merge_odd_even(even_m, even_n)
        odd_merge = merge_odd_even(odd_m, odd_n)
        
        # Place even merge at even positions
        for (i, j) in even_merge:
            comparators.append((2 * i, 2 * j))
        
        # Place odd merge at odd positions
        for (i, j) in odd_merge:
            comparators.append((2 * i + 1, 2 * j + 1))
        
        # Final phase: compare adjacent pairs that span the even/odd boundary
        # The final comparators are (1,2), (3,4), ..., etc.
        total = m + n
        for i in range(1, total - 1, 2):
            comparators.append((i, i + 1))
        
        return comparators
    
    # Generate the Batcher network
    network = batcher_odd_even_mergesort(N)
    
    # Deduplicate and ensure valid comparators
    seen = set()
    unique_network = []
    for (i, j) in network:
        if i < j and 0 <= i < N and 0 <= j < N:
            if (i, j) not in seen:
                seen.add((i, j))
                unique_network.append((i, j))
    
    # Now try to prune redundant comparators
    # Use the zero-one principle: a network is correct iff it sorts all 2^16 Boolean inputs
    # We can represent the set of inputs that are NOT sorted by the network as a bitset
    # and iteratively remove comparators that don't affect the failure set
    
    # For efficiency, we'll use a different approach: 
    # Check if removing a comparator still leaves the network correct.
    # We do this by simulating all 2^16 inputs, but we can do it efficiently
    # using bit operations.
    
    # Actually, for 16 wires, 2^16 = 65536 inputs. We can simulate each input
    # through the network and check if the output is sorted.
    # To speed up pruning, we'll try removing each comparator and checking.
    
    # But 65536 * 57 * 57 checks might be too slow in Python.
    # Let's use a smarter approach: represent the network's behavior on Boolean inputs
    # using the fact that compare-exchange on Boolean values is:
    #   min(a,b) = a AND b, max(a,b) = a OR b
    # So the network maps Boolean vectors to Boolean vectors via AND/OR operations.
    
    # We can represent each wire's value as a function of the 16 input bits.
    # Each wire's value is a monotone Boolean function of the inputs.
    # We can represent these functions as bitsets of size 2^16.
    
    # Let's do this: for each input bit pattern (0 to 65535), simulate the network.
    # Store the result as a bitset where bit k indicates whether wire k is 1.
    
    # Actually, let's just do a simple simulation for correctness checking.
    
    def is_sorted_network(network):
        """Check if the network sorts all 2^N Boolean inputs."""
        for mask in range(1 << N):
            # Initialize wires
            wires = [(mask >> i) & 1 for i in range(N)]
            # Apply network
            for (i, j) in network:
                if wires[i] > wires[j]:
                    wires[i], wires[j] = wires[j], wires[i]
            # Check if sorted
            for i in range(N - 1):
                if wires[i] > wires[i + 1]:
                    return False
        return True
    
    # Try to prune: for each comparator, check if removing it still leaves the network correct
    # We do this in a greedy fashion: try removing from the end first (later comparators
    # are more likely to be redundant)
    
    pruned = list(unique_network)
    changed = True
    while changed and len(pruned) > 50:  # Stop at 50 to avoid going below optimal
        changed = False
        for idx in range(len(pruned) - 1, -1, -1):
            # Try removing comparator at idx
            candidate = pruned[:idx] + pruned[idx + 1:]
            if is_sorted_network(candidate):
                pruned = candidate
                changed = True
                break
    
    return pruned
# EVOLVE-BLOCK-END
