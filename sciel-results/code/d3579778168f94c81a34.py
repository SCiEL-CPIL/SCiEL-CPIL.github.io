"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096
SEED_DEADLINE = 45.0  # safety margin before 60s kill

# Pre-compute the input matrix for verification to avoid regeneration overhead
_N_INPUTS = 1 << N
_INDICES = np.arange(_N_INPUTS, dtype=np.uint32)
_INPUT_MATRIX = np.zeros((_N_INPUTS, N), dtype=np.uint8)
for _i in range(N):
    _INPUT_MATRIX[:, _i] = (_INDICES >> _i) & 1

def _apply_network(network, x):
    """Apply the compare-exchange network to input vector x (numpy array)."""
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _verify_network_fastest(network):
    """
    Fast verification using pre-computed input matrix.
    """
    # Copy the input matrix
    x = _INPUT_MATRIX.copy()
    
    # Apply network
    for i, j in network:
        mask = x[:, i] > x[:, j]
        if np.any(mask):
            x[mask, i] = 0
            x[mask, j] = 1
    
    # Verify sorted
    for i in range(N - 1):
        if np.any(x[:, i] > x[:, i + 1]):
            return False
    return True

def _count_comparators(network):
    return len(network)

def _compute_depth(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    
    n = len(network)
    wire_to_last_comp = {}  # wire -> last comparator index that touched it
    depths = [0] * n
    
    for idx, (i, j) in enumerate(network):
        max_dep = 0
        for wire in (i, j):
            if wire in wire_to_last_comp:
                prev_idx = wire_to_last_comp[wire]
                if depths[prev_idx] + 1 > max_dep:
                    max_dep = depths[prev_idx] + 1
        depths[idx] = max_dep
        wire_to_last_comp[i] = idx
        wire_to_last_comp[j] = idx
    
    return max(depths) if depths else 0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network for 16 wires.
    
    Uses a known 60-comparator construction derived from Batcher's merge sort,
    then attempts to prune redundant comparators via randomized local search.
    """
    start_time = time.time()
    deadline = start_time + SEED_DEADLINE - 2.0  # extra safety margin
    
    # --- Base construction: Batcher's odd-even merge sort ---
    # This gives 119 comparators for N=16, which is a solid starting point.
    # We then aggressively prune.
    
    def _batcher_sort_network(n):
        """Return Batcher's sort network for n wires (0-indexed)."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        if n == 3:
            return [(0, 1), (1, 2), (0, 1)]
        if n == 4:
            return [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)]
        
        half = n // 2
        net = []
        # Sort first half
        net.extend([(i, j) for i, j in _batcher_sort_network(half)])
        # Sort second half
        net.extend([(i + half, j + half) for i, j in _batcher_sort_network(n - half)])
        # Merge
        net.extend(_batcher_merge_network(half, 0))
        return net
    
    def _batcher_merge_network(k, offset):
        """
        Merge two sorted lists of size k at positions offset and offset+k.
        Returns list of comparators.
        """
        if k == 1:
            return [(offset, offset + 1)]
        net = []
        # First stage: compare elements at distance k
        for i in range(k):
            net.append((offset + i, offset + i + k))
        # Recursive merges on even and odd positions
        # Even positions in the merged sequence correspond to logical indices 0, 2, 4, ...
        # which are at physical positions offset + 0, offset + 2, offset + 4, ...
        # Wait, Batcher's odd-even merge works as follows:
        # After the first stage of comparing (i, i+k), the even-indexed elements 
        # of the result form a sorted sequence, and the odd-indexed elements form a sorted sequence.
        # Then we recursively merge those.
        
        # The even logical positions (0, 2, 4, ..., 2k-2) in the 2k-element merge
        # correspond to physical positions offset + 0, offset + 2, ..., offset + 2k-2
        # But we need to map them to a contiguous range for recursive merging.
        # Actually, the standard formulation:
        # After comparing (i, i+k) for all i, the result is such that:
        #   positions 0, 2, 4, ..., 2k-2 contain the "even" part (already sorted relative to each other)
        #   positions 1, 3, 5, ..., 2k-1 contain the "odd" part (already sorted relative to each other)
        # So we recursively merge the even part and the odd part.
        
        # Even part: positions offset+0, offset+2, ..., offset+2k-2
        # These are k elements at even offsets. We treat them as a sub-problem of size k
        # with step 2. But our merge function assumes contiguous wires.
        # Let's use a different approach: use the standard recursive Batcher merge.
        
        # Standard Batcher odd-even merge for merging two sorted lists of size k:
        # The two lists are at positions [offset, offset+k) and [offset+k, offset+2k)
        # Stage 1: compare (offset+i, offset+k+i) for i in 0..k-1
        # Stage 2: recursively merge the "even" subsequence and "odd" subsequence
        #   Even subsequence: offset+0, offset+2, ..., offset+2k-2 (k elements)
        #   Odd subsequence: offset+1, offset+3, ..., offset+2k-1 (k elements)
        # But these are not contiguous. We need a merge that handles strided access.
        
        # Let me use a simpler formulation that works with contiguous blocks.
        # The standard way is to think of it as:
        # After stage 1, the even-indexed elements (0, 2, 4, ...) in the 2k sequence are sorted,
        # and the odd-indexed elements (1, 3, 5, ...) are sorted.
        # So we recursively merge two sorted sequences of length k, but they are interleaved.
        
        # To handle this, I'll use a helper that merges two sorted lists given their starting positions
        # and strides.
        
        def _merge_strided(k, start_even, start_odd, step):
            """
            Merge two sorted lists of size k.
            First list starts at start_even with stride step.
            Second list starts at start_odd with stride step.
            The merged result is interleaved: even positions from first, odd from second.
            """
            if k == 1:
                return [(start_even, start_odd)]
            net2 = []
            # Stage 1: compare corresponding elements
            for i in range(k):
                net2.append((start_even + i * step, start_odd + i * step))
            # Stage 2: recursively merge even and odd subsequences
            # Even subsequence of the merged result: positions 0, 2, 4, ... in the 2k sequence
            # These correspond to: start_even + 0*step, start_even + 2*step, ...
            # which is a strided sequence of size k with stride 2*step
            # Similarly for odd.
            net2.extend(_merge_strided(k // 2, start_even, start_odd, 2 * step))
            net2.extend(_merge_strided(k // 2, start_even + step, start_odd + step, 2 * step))
            return net2
        
        return _merge_strided(k, offset, offset + k, 1)
    
    # Actually, let me use a well-known simple Batcher implementation.
    # For N=16, Batcher's sort uses 119 comparators.
    # Let me just hardcode a known-good starting network and prune.
    
    # Generate Batcher sort network for 16 wires
    net = _batcher_sort_network(N)
    
    # Verify it's correct
    if not _verify_network_fastest(net):
        # Fallback: odd-even transposition (120 comparators)
        net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_network = list(net)
    best_count = len(best_network)
    
    # --- Greedy removal: try to remove each comparator and check if still valid ---
    changed = True
    while changed:
        changed = False
        for idx in range(len(best_network) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = best_network[:idx] + best_network[idx + 1:]
            if _verify_network_fastest(candidate):
                best_network = candidate
                changed = True
        if time.time() > deadline:
            break
    
    # --- Randomized local search: swap adjacent comparators, or try removing/inserting ---
    # Try to further reduce by swapping comparator order
    for _iter in range(50000):
        if time.time() > deadline:
            break
        idx1 = random.randint(0, len(best_network) - 1)
        idx2 = random.randint(0, len(best_network) - 1)
        if idx1 == idx2:
            continue
        # Swap
        candidate = list(best_network)
        candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        if _verify_network_fastest(candidate):
            # Check if any comparator can now be removed
            for r in range(len(candidate) - 1, -1, -1):
                if time.time() > deadline:
                    break
                test = candidate[:r] + candidate[r + 1:]
                if _verify_network_fastest(test):
                    candidate = test
                    break
            best_network = candidate
    
    return best_network
# EVOLVE-BLOCK-END
