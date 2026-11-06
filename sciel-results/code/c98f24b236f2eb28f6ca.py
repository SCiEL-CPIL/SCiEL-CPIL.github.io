import time
import numpy as np

N = 16
MAX_COMPARATORS = 4096
TIME_LIMIT = 60

def _batcher_oes(n):
    """Generate Batcher's Odd-Even Mergesort network for n wires (power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    k = n // 2
    left = _batcher_oes(k)
    right = _batcher_oes(k)
    right_shifted = [(i + k, j + k) for i, j in right]
    merge = _batcher_merge(n)
    return left + right_shifted + merge

def _batcher_merge(n):
    """Generate Batcher's odd-even merge for n wires (two sorted n/2 sequences)."""
    if n == 2:
        return [(0, 1)]
    k = n // 2
    result = []
    depth = int(np.log2(n))
    for d in range(depth):
        step = 1 << (depth - 1 - d)
        for i in range(0, n, 2 * step):
            for j in range(step):
                a = i + j
                b = i + j + step
                result.append((a, b))
    return result

def _verify_network(network):
    """
    Verify the network sorts all 2^16 Boolean inputs using zero-one principle.
    Uses vectorized NumPy for speed.
    Returns True if valid, False otherwise.
    """
    if not network:
        return N <= 1
    
    # Generate all 2^16 input vectors as a 2D array: shape (65536, 16)
    # Each row is one input mask, bits represent wire values
    all_masks = np.arange(1 << N, dtype=np.uint16)
    # Extract bits: for each wire position, get that bit from all masks
    # vec[m, w] = (m >> w) & 1
    wires = np.arange(N, dtype=np.uint8)
    # Create input matrix: shape (65536, 16)
    # Using broadcasting: masks[:, None] >> wires[None, :] & 1
    vec = (all_masks[:, None] >> wires[None, :]) & 1  # shape (65536, 16), uint8
    
    # Apply each comparator
    for i, j in network:
        # For each input, if vec[i] > vec[j], swap
        # vi = vec[:, i], vj = vec[:, j]
        # swap_mask = vi > vj  (True where we need to swap)
        swap_mask = vec[:, i] > vec[:, j]
        if np.any(swap_mask):
            # Where swap_mask is True, set vec[i]=0, vec[j]=1
            # Where False, leave as is
            vec[swap_mask, i] = 0
            vec[swap_mask, j] = 1
    
    # Check if all outputs are sorted: for each row, should be non-decreasing
    # i.e., vec[:, 0] <= vec[:, 1] <= ... <= vec[:, 15]
    # For binary, sorted means: all 0s come before all 1s
    # Check: for each adjacent pair, vec[:, k] <= vec[:, k+1]
    diffs = np.diff(vec, axis=1)  # shape (65536, 15)
    # If any diff is -1 (i.e., 1 followed by 0), it's not sorted
    if np.any(diffs < 0):
        return False
    return True

def _verify_sample(network, sample_size=5000):
    """Fast sample-based verification for quick trials."""
    if not network:
        return N <= 1
    
    np.random.seed(42)
    masks = np.random.randint(0, 1 << N, size=sample_size, dtype=np.uint16)
    wires = np.arange(N, dtype=np.uint8)
    vec = (masks[:, None] >> wires[None, :]) & 1  # shape (sample_size, 16)
    
    for i, j in network:
        swap_mask = vec[:, i] > vec[:, j]
        if np.any(swap_mask):
            vec[swap_mask, i] = 0
            vec[swap_mask, j] = 1
    
    diffs = np.diff(vec, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def construct():
    """
    Return a fixed compare-exchange network that sorts all 16-input sequences.
    Each comparator is a tuple (i, j) with 0 <= i < j < N.
    """
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Start with Batcher's OES for 16 wires
    network = _batcher_oes(N)
    best_network = network[:]
    best_count = len(best_network)
    
    # First, verify the Batcher network is correct
    if not _verify_network(best_network):
        # Fallback to odd-even transposition if Batcher has a bug
        best_network = [(i, i + 1) for phase in range(N)
                        for i in range(phase % 2, N - 1, 2)]
        best_count = len(best_network)
    
    # Greedy removal: try to remove each comparator, check if still valid
    import random
    rng = np.random.default_rng(seed=123)
    
    while time.time() < deadline:
        removed_any = False
        n_comp = len(best_network)
        
        # Try removals in random order for diversity
        indices = list(range(n_comp))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Skip if we've already improved significantly
            # Try removing comparator at position idx
            temp_network = best_network[:idx] + best_network[idx + 1:]
            
            # Quick sample check first
            if not _verify_sample(temp_network, sample_size=3000):
                continue
            
            # Full verification
            if _verify_network(temp_network):
                best_network = temp_network
                best_count -= 1
                removed_any = True
                # After a removal, restart the scan (indices are now shifted)
                break
        
        if not removed_any:
            # No more single removals work in this pass
            # Try a few more random shuffles to escape local minima
            for attempt in range(3):
                if time.time() > deadline:
                    break
                removed_any = False
                n_comp = len(best_network)
                indices = list(range(n_comp))
                rng.shuffle(indices)
                for idx in indices:
                    if time.time() > deadline:
                        break
                    temp_network = best_network[:idx] + best_network[idx + 1:]
                    if not _verify_sample(temp_network, sample_size=3000):
                        continue
                    if _verify_network(temp_network):
                        best_network = temp_network
                        best_count -= 1
                        removed_any = True
                        break
                if not removed_any:
                    break
            if not removed_any:
                break
    
    # Final safety check
    if not _verify_network(best_network):
        # Shouldn't happen, but fallback
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return best_network
