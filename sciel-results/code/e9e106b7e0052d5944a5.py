import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    # Batcher's odd-even mergesort network for N=16 (60 comparators)
    def batcher_odd_even_merge_sort(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        k = n // 2
        even_net = batcher_odd_even_merge_sort(k)
        odd_net = batcher_odd_even_merge_sort(k)
        even_mapped = [(2*i, 2*j) for i, j in even_net]
        odd_mapped = [(2*i+1, 2*j+1) for i, j in odd_net]
        merge_net = batcher_odd_even_merge(k)
        return even_mapped + odd_mapped + merge_net

    def batcher_odd_even_merge(k):
        if k == 1:
            return [(0, 1)]
        first = [(2*i, 2*i+1) for i in range(k)]
        k2 = k // 2
        even_merge = batcher_odd_even_merge(k2)
        odd_merge = batcher_odd_even_merge(k2)
        even_mapped = [(2*i, 2*j) for i, j in even_merge]
        odd_mapped = [(2*i+1, 2*j+1) for i, j in odd_merge]
        return first + even_mapped + odd_mapped

    batcher_net = batcher_odd_even_merge_sort(N)
    
    # Pre-compute inputs and targets
    inputs = np.arange(2**N, dtype=np.uint32)
    popcounts = np.array([bin(x).count('1') for x in range(2**N)], dtype=np.uint8)
    targets = np.array([((1 << k) - 1) << (N - k) if k > 0 else 0 for k in popcounts], dtype=np.uint32)

    def verify(net):
        masks = inputs.copy()
        for i, j in net:
            bit_i = 1 << i
            bit_j = 1 << j
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return np.array_equal(masks, targets)

    def verify_with_firing(net):
        """Returns (is_valid, fired_mask) where fired_mask[c] is a boolean array 
        indicating which inputs have comparator c fire."""
        masks = inputs.copy()
        fired = [None] * len(net)
        for c, (i, j) in enumerate(net):
            bit_i = 1 << i
            bit_j = 1 << j
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            fired[c] = cond
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        valid = np.array_equal(masks, targets)
        return valid, fired

    # Verify Batcher network
    if not verify(batcher_net):
        initial_net = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
        return initial_net

    current_net = list(batcher_net)
    best_net = list(batcher_net)
    best_len = len(current_net)
    
    start_time = time.time()
    deadline = 55.0
    random.seed(42)
    
    # Main optimization loop
    while time.time() - start_time < deadline:
        net_len = len(current_net)
        if net_len <= 40:  # Don't go below a reasonable bound
            break
        
        # Get firing information for current network
        valid, fired = verify_with_firing(current_net)
        if not valid:
            # Shouldn't happen, but recover
            current_net = list(best_net)
            continue
        
        # Find non-firing comparators (trivially removable)
        non_firing = [c for c in range(net_len) if not np.any(fired[c])]
        
        if non_firing:
            # Remove a non-firing comparator
            idx = random.choice(non_firing)
            current_net = current_net[:idx] + current_net[idx+1:]
            if len(current_net) < best_len:
                best_net = current_net
                best_len = len(current_net)
            continue
        
        # All comparators fire for some input. Try random removals.
        # Heuristic: try comparators that fire for fewer inputs first
        fire_counts = [np.sum(fired[c]) for c in range(net_len)]
        
        # Sort by fire count (ascending) - try low-impact comparators first
        order = sorted(range(net_len), key=lambda c: fire_counts[c])
        random.shuffle(order)
        # But keep some bias toward low fire count
        order = sorted(order, key=lambda c: (fire_counts[c] > 1000, random.random()))
        
        found_removal = False
        for idx in order[:min(10, net_len)]:
            candidate = current_net[:idx] + current_net[idx+1:]
            if verify(candidate):
                current_net = candidate
                if len(current_net) < best_len:
                    best_net = current_net
                    best_len = len(current_net)
                found_removal = True
                break
        
        if not found_removal:
            # Try swaps to escape local minima
            if net_len > 2:
                for _ in range(5):
                    i1 = random.randint(0, net_len - 1)
                    i2 = random.randint(0, net_len - 1)
                    if i1 != i2:
                        candidate = current_net.copy()
                        candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                        if verify(candidate):
                            current_net = candidate
                            break
    
    return best_net
# EVOLVE-BLOCK-END
