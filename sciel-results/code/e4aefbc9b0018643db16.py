"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s kill
    
    # Pre-compute bit masks for initialization
    NUM_WORDS = (1 << N) // 64
    words_per_wire = NUM_WORDS
    
    # Generate initial wires using NumPy
    total_inputs = 1 << N
    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)
    
    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        Returns True if the network sorts all Boolean inputs.
        """
        # Copy initial wires
        # Using list comprehension with copy is fast enough
        wires = [w.copy() for w in initial_wires]
        
        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            # min to i, max to j
            # min(a, b) = a & b
            # max(a, b) = a | b
            wires[i] = wi & wj
            wires[j] = wi | wj
        
        # Check sorted: for all k, wires[k] & ~wires[k+1] == 0
        # This means if wire k has a 1, wire k+1 must have a 1.
        # So wires[k] is a subset of wires[k+1].
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    def greedy_remove(net):
        """
        Attempts to remove redundant comparators.
        Returns a new network with as few comparators as possible 
        by iteratively removing any removable comparator.
        """
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to keep indices stable if we were to 
            # modify in place, but we build a new list or break.
            # We iterate in random order to avoid bias.
            indices = list(range(len(net)))
            random.shuffle(indices)
            for idx in indices:
                if len(net) == 0:
                    return net
                # Create test network
                test = net[:idx] + net[idx + 1:]
                if verify_network(test):
                    net = test
                    changed = True
                    break # Restart loop with new length
        return net

    # Start with odd-even transposition
    current = odd_even_transposition()
    current = greedy_remove(current)
    
    best_network = list(current)
    best_count = len(current)
    
    random.seed(42)
    
    # Local Search
    while time.time() < deadline - 1.0:
        if not current:
            break
            
        # Choose a move type
        r = random.random()
        
        # Work on a copy to allow easy reversion
        candidate = list(current)
        
        if r < 0.4:
            # Move: Pick a comparator and move it to a new position
            if len(candidate) > 1:
                i = random.randint(0, len(candidate) - 1)
                j = random.randint(0, len(candidate) - 1)
                # Extract
                comp = candidate.pop(i)
                # Adjust j if i < j
                if i < j:
                    j -= 1
                # Insert
                candidate.insert(j, comp)
        elif r < 0.7:
            # Swap: Swap two comparators
            if len(candidate) > 1:
                i, j = random.sample(range(len(candidate)), 2)
                candidate[i], candidate[j] = candidate[j], candidate[i]
        else:
            # Replace: Replace a comparator with a random one
            i = random.randint(0, len(candidate) - 1)
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                # Retry or skip
                continue
            if u > v:
                u, v = v, u
            candidate[i] = (u, v)
        
        # Verify candidate
        if verify_network(candidate):
            # Accept and try to reduce
            candidate = greedy_remove(candidate)
            
            if len(candidate) < best_count:
                best_count = len(candidate)
                best_network = list(candidate)
                current = candidate
            else:
                # Even if not better than best, accept if it's the same size?
                # Or reject to keep hill climbing strict?
                # Accepting same-size moves can help escape local minima.
                # Let's accept if size is <= current size.
                if len(candidate) <= len(current):
                    current = candidate
                else:
                    # Revert (candidate was a copy, so just ignore)
                    pass
        else:
            # Revert (candidate was a copy, so just ignore)
            pass

    # Final verification
    if not verify_network(best_network):
        best_network = odd_even_transposition()
    
    return best_network
# EVOLVE-BLOCK-END
