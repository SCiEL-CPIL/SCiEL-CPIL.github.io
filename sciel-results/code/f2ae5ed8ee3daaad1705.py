import numpy as np
import time
import random
import sys
from functools import lru_cache

N = 16

# EVOLVE-BLOCK-START
def construct():
    # Initial odd-even transposition network: 120 comparators
    initial_net = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    # Pre-compute inputs and targets using vectorized operations
    inputs = np.arange(2**N, dtype=np.uint32)
    # Target: sorted 0s then 1s.
    # If k ones, target is ((1 << k) - 1) << (N - k)
    popcounts = np.unpackbits(inputs.view(np.uint8).reshape(-1, 4), axis=1, count=4).sum(axis=1).astype(np.uint8)
    targets = np.zeros(2**N, dtype=np.uint32)
    for k in range(1, N+1):
        mask = (popcounts == k)
        targets[mask] = ((1 << k) - 1) << (N - k)
    # k=0 gives 0, which is already set

    # Precompute bit masks for each wire position
    bit_masks = [1 << i for i in range(N)]
    
    def verify(net):
        """Fast verification using in-place operations on a copy."""
        masks = inputs.copy()
        for i, j in net:
            bit_i = bit_masks[i]
            bit_j = bit_masks[j]
            # Condition: bit i is 1, bit j is 0
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks[cond] ^= bit_i | bit_j
        return np.array_equal(masks, targets)

    def verify_fast(net):
        """Even faster verification with early termination and caching."""
        masks = inputs.copy()
        for i, j in net:
            bit_i = bit_masks[i]
            bit_j = bit_masks[j]
            # Condition: bit i is 1, bit j is 0
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks[cond] ^= bit_i | bit_j
            # Early check: if all sorted, we can stop
            if np.array_equal(masks, targets):
                return True
        return np.array_equal(masks, targets)

    # Verify initial
    if not verify(initial_net):
        return initial_net

    current_net = list(initial_net)
    best_net = list(initial_net)
    best_len = len(current_net)
    
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    random.seed(42)
    
    # Phase 1: Systematic greedy removal (multiple passes)
    for pass_num in range(10):
        if time.time() - start_time > deadline:
            break
        i = 0
        while i < len(current_net):
            if time.time() - start_time > deadline:
                break
            candidate = current_net[:i] + current_net[i+1:]
            if verify_fast(candidate):
                current_net = candidate
                if len(current_net) < best_len:
                    best_net = current_net
                    best_len = len(current_net)
            else:
                i += 1
    
    # Phase 2: Random local search with multiple strategies
    removal_count = 0
    while time.time() - start_time < deadline:
        strategy = random.random()
        
        if strategy < 0.4 and len(current_net) > 1:
            # Random removal
            idx = random.randint(0, len(current_net) - 1)
            candidate = current_net[:idx] + current_net[idx+1:]
            if verify_fast(candidate):
                current_net = candidate
                removal_count += 1
                if len(current_net) < best_len:
                    best_net = current_net
                    best_len = len(current_net)
        
        elif strategy < 0.6 and len(current_net) > 1:
            # Swap two comparators
            i1 = random.randint(0, len(current_net) - 1)
            i2 = random.randint(0, len(current_net) - 1)
            if i1 != i2:
                candidate = current_net.copy()
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                if verify_fast(candidate):
                    current_net = candidate
        
        elif strategy < 0.75 and len(current_net) > 2:
            # Move a comparator to a random position
            i1 = random.randint(0, len(current_net) - 1)
            i2 = random.randint(0, len(current_net) - 1)
            if i1 != i2:
                candidate = current_net.copy()
                comp = candidate.pop(i1)
                candidate.insert(i2, comp)
                if verify_fast(candidate):
                    current_net = candidate
        
        elif strategy < 0.85:
            # Insert a useful comparator then try removals
            if len(current_net) > 1:
                # Find a random pair that might help
                new_i = random.randint(0, N - 2)
                new_j = random.randint(new_i + 1, N - 1)
                insert_pos = random.randint(0, len(current_net))
                candidate = current_net.copy()
                candidate.insert(insert_pos, (new_i, new_j))
                if verify_fast(candidate):
                    current_net = candidate
                    # Try to remove other comparators now
                    for _ in range(3):
                        if time.time() - start_time > deadline:
                            break
                        if len(current_net) > 1:
                            idx = random.randint(0, len(current_net) - 1)
                            candidate2 = current_net[:idx] + current_net[idx+1:]
                            if verify_fast(candidate2):
                                current_net = candidate2
                                if len(current_net) < best_len:
                                    best_net = current_net
                                    best_len = len(current_net)
        
        else:
            # Restart from best if stuck
            if removal_count > 50 and random.random() < 0.3:
                current_net = best_net.copy()
                removal_count = 0
    
    # Final phase: Try one more systematic pass
    if time.time() - start_time < deadline - 2:
        i = 0
        while i < len(current_net):
            if time.time() - start_time > deadline:
                break
            candidate = current_net[:i] + current_net[i+1:]
            if verify_fast(candidate):
                current_net = candidate
                if len(current_net) < best_len:
                    best_net = current_net
                    best_len = len(current_net)
            else:
                i += 1
    
    return best_net
# EVOLVE-BLOCK-END
