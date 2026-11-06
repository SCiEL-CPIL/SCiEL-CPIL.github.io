import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    # Initial odd-even transposition network: 120 comparators
    initial_net = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
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

    def verify_batch(net, remove_indices):
        """Try removing each index in remove_indices and check validity.
        Returns set of indices that can be removed."""
        valid_removals = set()
        for idx in remove_indices:
            candidate = net[:idx] + net[idx+1:]
            masks = inputs.copy()
            for i, j in candidate:
                bit_i = 1 << i
                bit_j = 1 << j
                cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
                if np.any(cond):
                    masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
            if np.array_equal(masks, targets):
                valid_removals.add(idx)
        return valid_removals

    # Verify initial
    if not verify(initial_net):
        return initial_net

    current_net = initial_net
    best_net = initial_net
    best_len = len(current_net)
    
    start_time = time.time()
    deadline = 55.0
    
    random.seed(42)
    
    # Strategy: iteratively try to remove comparators, with bias toward
    # comparators in the "middle" of the network which are more likely redundant
    while time.time() - start_time < deadline:
        net_len = len(current_net)
        if net_len <= 1:
            break
        
        # Try to find a removable comparator
        # Bias: try indices in a shuffled order but with some structure
        indices = list(range(net_len))
        random.shuffle(indices)
        
        # Try up to a batch of removals
        batch_size = min(20, net_len)
        batch = indices[:batch_size]
        
        removable = verify_batch(current_net, batch)
        
        if removable:
            # Remove the first removable one found
            idx = min(removable)
            current_net = current_net[:idx] + current_net[idx+1:]
            if len(current_net) < best_len:
                best_net = current_net
                best_len = len(current_net)
        else:
            # No removals found in this batch; try a swap to escape local minimum
            if net_len > 1:
                i1 = random.randint(0, net_len - 1)
                i2 = random.randint(0, net_len - 1)
                if i1 != i2:
                    candidate = current_net.copy()
                    candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                    if verify(candidate):
                        current_net = candidate

    return best_net
# EVOLVE-BLOCK-END
