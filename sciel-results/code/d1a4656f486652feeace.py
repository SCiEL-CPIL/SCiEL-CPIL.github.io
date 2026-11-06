import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on 16 wires.
    Strategy: Start from Batcher's odd-even mergesort (60 comparators),
    then use a fast verification + aggressive local search to remove comparators.
    """
    
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
    
    # Precompute targets for all 2^16 inputs
    all_inputs = np.arange(2**N, dtype=np.uint32)
    popcounts = np.array([bin(x).count('1') for x in range(2**N)], dtype=np.uint8)
    targets = np.array([((1 << k) - 1) << (N - k) if k > 0 else 0 for k in popcounts], dtype=np.uint32)
    
    def verify(net):
        """Verify network sorts all 2^N inputs. Returns True if correct."""
        masks = all_inputs.copy()
        for i, j in net:
            bit_i = 1 << i
            bit_j = 1 << j
            # cond: bit i is 1 and bit j is 0 -> need to swap
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return np.array_equal(masks, targets)
    
    def verify_subset(net, subset):
        """Verify on a subset of inputs for quick rejection."""
        masks = subset.copy()
        for i, j in net:
            bit_i = 1 << i
            bit_j = 1 << j
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        # Check if sorted: for each input, the result should have all 1s in high positions
        pc = np.array([bin(x).count('1') for x in subset], dtype=np.uint8)
        tgt = np.array([((1 << k) - 1) << (N - k) if k > 0 else 0 for k in pc], dtype=np.uint32)
        return np.array_equal(masks, tgt)
    
    # Quick check that Batcher is valid
    if not verify(batcher_net):
        # Fallback: odd-even transposition
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    current_net = list(batcher_net)
    best_net = list(batcher_net)
    best_len = len(current_net)
    
    start_time = time.time()
    deadline = 50.0  # Leave margin for 90s total
    
    random.seed(42)
    
    # Compute "criticality" for each comparator:
    # For each comparator, count how many inputs it actually swaps.
    # Comparators with few swaps are more likely removable.
    def compute_criticality(net):
        """Return array of swap counts for each comparator in the network."""
        counts = np.zeros(len(net), dtype=np.int32)
        masks = all_inputs.copy()
        for idx, (i, j) in enumerate(net):
            bit_i = 1 << i
            bit_j = 1 << j
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            counts[idx] = np.sum(cond)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return counts
    
    # Use a smaller random subset for quick checks during search
    # to speed up the inner loop
    subset_size = 4096
    rng = np.random.default_rng(123)
    subset_idx = rng.choice(2**N, size=subset_size, replace=False)
    subset = all_inputs[subset_idx].copy()
    subset_pc = np.array([bin(x).count('1') for x in subset], dtype=np.uint8)
    subset_tgt = np.array([((1 << k) - 1) << (N - k) if k > 0 else 0 for k in subset_pc], dtype=np.uint32)
    
    def verify_subset_fast(net):
        """Fast verification on subset."""
        masks = subset.copy()
        for i, j in net:
            bit_i = 1 << i
            bit_j = 1 << j
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return np.array_equal(masks, subset_tgt)
    
    no_improve = 0
    
    while time.time() - start_time < deadline:
        net_len = len(current_net)
        
        # Compute criticality to prioritize removal candidates
        crit = compute_criticality(current_net)
        
        # Sort indices by criticality (low first = most likely removable)
        order = np.argsort(crit)
        
        found = False
        # Try removing comparators in order of low criticality
        for idx in order:
            if time.time() - start_time > deadline:
                break
            idx = int(idx)
            # Quick subset check first
            candidate = current_net[:idx] + current_net[idx+1:]
            if verify_subset_fast(candidate):
                # Full verification
                if verify(candidate):
                    current_net = candidate
                    found = True
                    if len(current_net) < best_len:
                        best_net = list(current_net)
                        best_len = len(current_net)
                    no_improve = 0
                    break
        
        if not found:
            no_improve += 1
            
            if no_improve >= 30:
                # Try swapping adjacent comparators
                if net_len > 1:
                    i1 = random.randint(0, net_len - 2)
                    candidate = current_net.copy()
                    candidate[i1], candidate[i1+1] = candidate[i1+1], candidate[i1]
                    if verify_subset_fast(candidate):
                        if verify(candidate):
                            current_net = candidate
                            no_improve = 0
            
            if no_improve >= 60:
                # Try replacing a low-criticality comparator with a random one
                if net_len > 1:
                    idx = int(order[0])
                    new_i = random.randint(0, N - 1)
                    new_j = random.randint(0, N - 1)
                    if new_i < new_j:
                        candidate = current_net.copy()
                        candidate[idx] = (new_i, new_j)
                        if verify_subset_fast(candidate):
                            if verify(candidate):
                                current_net = candidate
                                no_improve = 0
            
            if no_improve >= 100:
                # Restart from best
                current_net = list(best_net)
                no_improve = 0
    
    return best_net
# EVOLVE-BLOCK-END
