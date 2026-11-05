"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a size-efficient sorting network for 16 wires.
    Uses a heuristic search (simulated annealing / greedy) to minimize comparator count
    while verifying correctness via the zero-one principle.
    
    The strategy:
    1. Start with a known correct but potentially non-minimal network (e.g., Batcher odd-even merge or a dense random network).
    2. Iteratively remove or swap comparators if the network remains correct.
    3. Use the zero-one principle: a network sorts all inputs iff it sorts all 2^16 binary inputs.
    4. To speed up verification, we can simulate the network on all 65536 binary inputs.
    
    Since 65536 * 120 comparators * 16 bits is manageable in Python if optimized, or we can use numpy.
    However, pure Python loop over 65536 inputs might be slow. Let's use a bit-parallel approach.
    
    Actually, let's use a simpler heuristic:
    - Start with the odd-even transposition network (120 comparators).
    - Try to remove each comparator one by one. If the network is still correct, keep it removed.
    - This is a greedy reduction. It may not find the global minimum but should reduce from 120.
    
    For better results, we can also try swapping comparators or using a known good network as a seed.
    A known 60-comparator network exists. Let's try to find one via local search.
    
    Verification:
    def is_sorting_network(comparators):
        for mask in range(1 << N):
            # Simulate
            # ...
    
    To make it fast, we can represent the state of 16 wires as a 16-bit integer.
    A comparator (i, j) on a 16-bit int x:
        bit_i = (x >> i) & 1
        bit_j = (x >> j) & 1
        if bit_i > bit_j:
            x ^= (1 << i) | (1 << j)
    
    We need to check if the output is sorted, i.e., all 1s are on the left (higher indices? No, min to i, max to j).
    Wait, "min to wire i and max to j". Since i < j, wire i gets the smaller value.
    So for binary inputs, 0 is smaller than 1. So wire i gets 0, wire j gets 1 if they were 1,0.
    A sorted binary array has all 0s followed by all 1s.
    So the output must be of the form 0...01...1.
    This means the output integer x must satisfy:
    There exists k such that bits 0..k-1 are 0 and bits k..15 are 1.
    Equivalently, x must be in the set {0, 1, 3, 7, ..., 2^16-1} (i.e., 2^k - 1).
    
    Let's implement a fast verifier.
    """
    
    def verify(comparators):
        # Check all 2^16 binary inputs
        # For each input, simulate the network and check if output is sorted.
        # Sorted binary output: 000...0111...1
        # This corresponds to integers of the form 2^k - 1 for k in 0..16.
        valid_outputs = set(2**k - 1 for k in range(N + 1))
        
        for inp in range(1 << N):
            x = inp
            for i, j in comparators:
                # Compare and exchange
                # bit i should be <= bit j
                # if bit i > bit j, swap
                # Since i < j, we want 0 at i and 1 at j.
                # If x has 1 at i and 0 at j, we need to swap.
                # Condition: ((x >> i) & 1) > ((x >> j) & 1)
                # This is equivalent to: (x & (1 << i)) and not (x & (1 << j))
                # If true, flip both bits.
                if (x & (1 << i)) and not (x & (1 << j)):
                    x ^= (1 << i) | (1 << j)
            if x not in valid_outputs:
                return False
        return True

    # Start with a known network. Let's use the odd-even transposition network.
    # Or better, a Batcher network which is smaller.
    # Batcher odd-even merge sort for 16 elements has 120 comparators? No, Batcher is more efficient.
    # Let's just start with the 120 comparator network and try to prune.
    
    # Actually, let's start with a random network that is likely to be correct, or just the 120 one.
    # The 120 one is guaranteed correct.
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Greedy reduction
    # We will iterate and try to remove each comparator.
    # To be more effective, we can shuffle the order of removal attempts.
    
    # Let's do multiple passes.
    for _ in range(10):  # 10 passes of greedy removal
        removed_any = False
        # Shuffle the indices to avoid bias
        indices = list(range(len(network)))
        random.shuffle(indices)
        for idx in indices:
            if idx >= len(network):
                continue
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            if verify(candidate):
                network = candidate
                removed_any = True
        if not removed_any:
            break
    
    # Optional: Try to swap comparators to enable further removals?
    # This is complex. Let's stick to removal for now.
    
    # If we want to be more aggressive, we could try to replace a comparator with a different one.
    # But removal is the primary driver.
    
    return network
# EVOLVE-BLOCK-END
