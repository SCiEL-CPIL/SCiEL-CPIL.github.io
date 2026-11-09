"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def _verify_network(comps):
    """Verify sorting network using zero-one principle (all 2^N binary inputs)."""
    if not comps:
        return True
    # Use bit-parallel: represent all 65536 inputs as 16 bit-vectors
    # Each wire is a 65536-bit vector (as array of uint64)
    # Actually, let's use the numpy approach but optimized
    idx = np.arange(65536, dtype=np.int32)
    x = np.zeros((16, 65536), dtype=np.int8)  # transposed for better cache
    for i in range(16):
        x[i] = (idx >> i) & 1
    
    for (i, j) in comps:
        a = x[i]
        b = x[j]
        # min: a & b, max: a | b for binary
        x[i] = a & b
        x[j] = a | b
    
    # Check sorted: x[0] <= x[1] <= ... <= x[15]
    # For binary: x[i] & x[i+1] should equal x[i] (i.e., x[i] subset of x[i+1])
    for i in range(15):
        if np.any(x[i] > x[i + 1]):
            return False
    return True

def _verify_network_fast(comps):
    """Faster verification using integer bit-parallelism."""
    if not comps:
        return True
    # Represent each wire as a Python int with 65536 bits
    # Wire i has bit k set iff input k has a 1 on wire i
    wires = []
    for i in range(16):
        mask = 0
        # For input value v (0..65535), wire i gets (v >> i) & 1
        # We need: for all v, wire_i[v] = (v >> i) & 1
        # This is a pattern: bits where (v >> i) & 1 == 1
        # v ranges 0 to 65535, so we need a 65536-bit integer
        # Pattern for wire i: blocks of 2^i ones followed by 2^i zeros, repeated
        # Actually, let's build it differently
        # For wire i, bit position v is 1 iff (v >> i) & 1 == 1
        # This means v mod 2^(i+1) >= 2^i
        block = 1 << (i + 1)
        half = 1 << i
        # Build the mask: for each block of size `block`, the upper half bits are 1
        mask = 0
        for start in range(0, 65536, block):
            mask |= ((1 << half) - 1) << (start + half)
        wires.append(mask)
    
    for (i, j) in comps:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b
        wires[j] = a | b
    
    # Check sorted: for all consecutive wires, wires[k] subset of wires[k+1]
    for k in range(15):
        if wires[k] & ~wires[k + 1]:
            return False
    return True

def _batcher_oe_merge_sort(n):
    """Build Batcher's odd-even merge sort network."""
    def oe_merge_network(n):
        """Odd-even merge network for 2n inputs (n from top half, n from bottom half)."""
        if n == 1:
            return [(0, 1)]
        comps = [(i, n + i) for i in range(n)]
        even_merge = oe_merge_network(n // 2)
        odd_merge = oe_merge_network(n // 2)
        for (a, b) in even_merge:
            comps.append((2 * a, 2 * b))
        for (a, b) in odd_merge:
            comps.append((2 * a + 1, 2 * b + 1))
        return comps
    
    def sort_network(n):
        """Sorting network using odd-even merge sort."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        if n % 2 == 1:
            base = sort_network(n - 1)
            extra = [(i, n - 1) for i in range(n - 1)]
            return base + extra
        half = n // 2
        left = sort_network(half)
        right = [(i + half, j + half) for (i, j) in sort_network(half)]
        merge = oe_merge_network(half)
        return left + right + merge
    
    return sort_network(n)

def _batcher_bitonic_sort(n):
    """Build Batcher's bitonic sort network."""
    def bitonic_merge(n, direction=1):
        """Bitonic merge network for n inputs."""
        if n == 1:
            return []
        comps = []
        half = n // 2
        if n % 2 == 1:
            # Handle odd: compare (i, n-1-i) for i in 0..n//2-1, then merge
            for i in range(n // 2):
                if direction == 1:
                    comps.append((i, n - 1 - i))
                else:
                    comps.append((n - 1 - i, i))
            # Recursively merge the two halves (they are now bitonic)
            # Actually for odd n, this gets complex. Let's just use even case.
            # For n=16, we only need even cases in the recursion
            pass
        for i in range(half):
            if direction == 1:
                comps.append((i, n - 1 - i))
            else:
                comps.append((n - 1 - i, i))
        # Recursively merge even and odd indexed elements
        even_merge = bitonic_merge(half, direction)
        odd_merge = bitonic_merge(half, direction)
        for (a, b) in even_merge:
            comps.append((2 * a, 2 * b))
        for (a, b) in odd_merge:
            comps.append((2 * a + 1, 2 * b + 1))
        return comps
    
    def bitonic_sort(n):
        """Bitonic sort network for n inputs."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        half = n // 2
        # Sort first half ascending, second half descending
        left = bitonic_sort(half)
        right_desc = bitonic_sort(half)
        # Reverse the right half comparators to make it descending
        right = [(i + half, j + half) for (i, j) in right_desc]
        # To make descending, we need to reverse the output
        # Actually, for bitonic sort: sort left ascending, right descending, then merge
        # The right half needs to be sorted in opposite direction
        # We can achieve this by applying the sort network but with reversed comparisons
        # Simpler: just build the standard bitonic sort
        comps = left + right
        # Now merge the two halves
        merge = bitonic_merge(n, 1)
        comps.extend(merge)
        return comps
    
    return bitonic_sort(n)

def _local_search_remove(comps, max_iters=50000, deadline=None, use_fast_verify=True):
    """Local search to minimize comparator count by removing redundant comparators."""
    best = list(comps)
    start_time = time.time()
    iters = 0
    no_improve = 0
    verify_fn = _verify_network_fast if use_fast_verify else _verify_network
    
    import random
    random.seed(42)
    
    while iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if verify_fn(trial):
                best = trial
                no_improve = 0
            else:
                no_improve += 1
        
        iters += 1
        if iters % 100 == 0 and no_improve > 3000:
            break
    
    return best

def _local_search_swap(comps, max_iters=20000, deadline=None):
    """Local search that tries swapping adjacent comparators."""
    best = list(comps)
    start_time = time.time()
    iters = 0
    no_improve = 0
    
    import random
    random.seed(123)
    
    while iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        
        if len(best) > 1:
            idx = random.randint(0, len(best) - 2)
            trial = best[:]
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _verify_network_fast(trial):
                best = trial
                no_improve = 0
            else:
                no_improve += 1
        
        iters += 1
        if iters % 100 == 0 and no_improve > 5000:
            break
    
    return best

def _try_insert_remove(comps, deadline=None):
    """Try to find a better network by inserting a comparator and then removing others."""
    best = list(comps)
    best_count = len(best)
    
    import random
    random.seed(77)
    
    for _ in range(20):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try inserting a random comparator
        i = random.randint(0, 15)
        j = random.randint(i + 1, 15)
        pos = random.randint(0, len(best))
        trial = best[:pos] + [(i, j)] + best[pos:]
        
        if _verify_network_fast(trial):
            # Now try to remove comparators from this larger network
            reduced = _local_search_remove(trial, max_iters=2000, deadline=deadline)
            if len(reduced) < best_count:
                best = reduced
                best_count = len(best)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Start with Batcher's odd-even merge sort (72 comparators for N=16)
    comps = _batcher_oe_merge_sort(N)
    
    # Verify initial network
    if not _verify_network_fast(comps):
        # Fallback: odd-even transposition
        comps = [(i, i + 1) for phase in range(N)
                 for i in range(phase % 2, N - 1, 2)]
    
    # Phase 1: Aggressive removal
    best = _local_search_remove(comps, max_iters=200000, deadline=deadline)
    
    # Phase 2: Try swap-based local search
    if time.time() < deadline - 10:
        swapped = _local_search_swap(best, max_iters=50000, deadline=deadline)
        if len(swapped) < len(best):
            best = swapped
    
    # Phase 3: Try insert-then-remove
    if time.time() < deadline - 5:
        inserted = _try_insert_remove(best, deadline=deadline)
        if len(inserted) < len(best):
            best = inserted
    
    # Final verification
    if not _verify_network_fast(best):
        # Fall back to the initial verified network
        if _verify_network_fast(comps):
            return comps
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return best
# EVOLVE-BLOCK-END
