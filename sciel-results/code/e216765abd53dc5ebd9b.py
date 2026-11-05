"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators
    
    def verify_network(comparators):
        """Check if the network sorts all 2^N Boolean inputs using numpy."""
        total = 1 << N
        bits = np.arange(total, dtype=np.int32)
        wires = np.zeros((total, N), dtype=np.int8)
        for i in range(N):
            wires[:, i] = (bits >> i) & 1
        
        for (i, j) in comparators:
            wi = wires[:, i]
            wj = wires[:, j]
            # min to i, max to j
            wires[:, i] = np.where(wi < wj, wi, wj)
            wires[:, j] = np.where(wi < wj, wj, wi)
        
        diffs = np.diff(wires, axis=1)
        return bool(np.all(diffs >= 0))
    
    def verify_network_bit(comparators):
        """Faster verification using bit-level parallelism."""
        # Each wire holds a 2^N-bit value represented as a list of uint64
        # But for N=18, 2^18 = 262144 bits = 4096 uint64 values per wire
        # This is complex; stick with numpy for clarity but optimize the inner loop
        
        total = 1 << N
        # Pre-compute the initial wires as a 2D array
        bits = np.arange(total, dtype=np.int32)
        wires = np.empty((total, N), dtype=np.int8)
        for i in range(N):
            wires[:, i] = (bits >> i) & 1
        
        for (i, j) in comparators:
            wi = wires[:, i]
            wj = wires[:, j]
            # Use minimum/maximum which are fast numpy ops
            wires[:, i] = np.minimum(wi, wj)
            wires[:, j] = np.maximum(wi, wj)
        
        # Check sorted: all adjacent differences >= 0
        return bool(np.all(np.diff(wires, axis=1) >= 0))
    
    # Start with odd-even transposition
    best_network = odd_even_transposition()
    best_count = len(best_network)
    
    # Phase 1: Greedy removal - try removing each comparator from last to first
    current = list(best_network)
    
    # Pre-compute all possible comparator pairs for reference
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    all_pairs_set = set(all_pairs)
    
    # Greedy removal pass
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            test = current[:idx] + current[idx + 1:]
            if verify_network(test):
                current = test
                best_network = list(current)
                best_count = len(current)
                changed = True
                break  # Restart since indices shifted
    
    # Phase 2: Local search with 2-opt moves
    # Try to replace one comparator with another to enable further reductions
    random.seed(42)
    
    # Score function: prefer moves that keep the network valid
    # We'll do random restarts of local search
    
    for iteration in range(2000):
        if time.time() > deadline - 5.0:
            break
        if not current:
            break
        
        # Pick a random position to modify
        idx = random.randint(0, len(current) - 1)
        old_i, old_j = current[idx]
        
        # Try a few random alternative comparators
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            
            # Random new comparator
            new_i = random.randint(0, N - 1)
            new_j = random.randint(0, N - 1)
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            if new_i == new_j:
                continue
            if (new_i, new_j) == (old_i, old_j):
                continue
            
            test = list(current)
            test[idx] = (new_i, new_j)
            
            if verify_network(test):
                current = test
                best_network = list(current)
                best_count = len(current)
                # After a successful swap, try to remove more
                # Quick removal pass
                removed = True
                while removed and time.time() < deadline - 5.0:
                    removed = False
                    for ridx in range(len(current) - 1, -1, -1):
                        if time.time() > deadline - 5.0:
                            break
                        test2 = current[:ridx] + current[ridx + 1:]
                        if verify_network(test2):
                            current = test2
                            best_network = list(current)
                            best_count = len(current)
                            removed = True
                            break
                break
    
    # Final verification
    if not verify_network(best_network):
        best_network = odd_even_transposition()
    
    return best_network
# EVOLVE-BLOCK-END
