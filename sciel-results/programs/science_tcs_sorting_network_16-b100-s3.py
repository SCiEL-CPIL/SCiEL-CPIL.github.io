"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time

def _batcher_odd_even_merge_sort(n):
    """Batcher's odd-even merge sort network for n wires (n must be power of 2)."""
    if n == 1:
        return []
    half = n // 2
    left = _batcher_odd_even_merge_sort(half)
    right = [(i + half, j + half) for (i, j) in left]
    # Odd-even merge of two sorted halves
    merge = _odd_even_merge(half)
    # Apply merge: pairs (i, i+half) for the merge network
    # The merge network operates on 2*half wires where wire i (0..half-1) is from left
    # and wire i+half is from right. The merge produces sorted 2*half output.
    # We need to translate the merge comparators to the full wire indices.
    # The merge network for size 2*half uses wires 0..2*half-1.
    # Left half occupies wires 0..half-1, right half occupies wires half..2*half-1.
    # So the merge comparators are already in the right coordinate system.
    return left + right + merge

def _odd_even_merge(n):
    """Odd-even merge network for 2n wires (n from each half).
    Returns comparators for wires 0..2n-1 where 0..n-1 are 'even' (left) and n..2n-1 are 'odd' (right)."""
    if n == 1:
        return [(0, 1)]
    # Split into even and odd positions
    # Even positions: 0, 2, 4, ... 2n-2  -> these come from left[0], right[0], left[1], right[1], ...
    # Actually for odd-even merge:
    #   Merge even-indexed inputs: positions 0,2,4,...,2n-2 (n elements)
    #   Merge odd-indexed inputs: positions 1,3,5,...,2n-1 (n elements)
    #   Then merge the two resulting sorted sequences
    # The even positions get values from left[0],right[0],left[1],right[1],...
    # The odd positions get values from left[1],right[1],left[2],right[2],...
    
    # Recursive: merge even subsequence and odd subsequence
    even_merge = _odd_even_merge(n // 2)
    odd_merge = _odd_even_merge(n // 2)
    
    # Translate even_merge from wires 0..n-1 to wires 0,2,4,...,2n-2
    even_translated = [(2*i, 2*j) for (i, j) in even_merge]
    # Translate odd_merge from wires 0..n-1 to wires 1,3,5,...,2n-1
    odd_translated = [(2*i + 1, 2*j + 1) for (i, j) in odd_merge]
    
    # Final merge: compare (2i+1, 2i+2) for i in 0..n-2
    final_merge = [(2*i + 1, 2*i + 2) for i in range(n - 1)]
    
    return even_translated + odd_translated + final_merge

def _apply_comparator(wires, i, j):
    """Apply compare-exchange: min to wire i, max to wire j."""
    if wires[i] > wires[j]:
        wires[i], wires[j] = wires[j], wires[i]

def _apply_network(wires, comparators):
    """Apply a network of comparators to a wire vector."""
    for (i, j) in comparators:
        _apply_comparator(wires, i, j)

def _is_sorted(wires):
    """Check if wires is sorted in non-decreasing order."""
    for i in range(len(wires) - 1):
        if wires[i] > wires[i + 1]:
            return False
    return True

def _verify_network(comparators, n):
    """Verify a sorting network using the zero-one principle."""
    for mask in range(1 << n):
        wires = [(mask >> i) & 1 for i in range(n)]
        _apply_network(wires, comparators)
        if not _is_sorted(wires):
            return False
    return True

def _remove_rendundant(comparators, n, deadline):
    """Remove redundant comparators from a valid sorting network."""
    comparators = list(comparators)
    i = 0
    while i < len(comparators) and time.time() < deadline:
        candidate = comparators[:i] + comparators[i+1:]
        if _verify_network(candidate, n):
            comparators = candidate
            # Don't increment i, check same position again
        else:
            i += 1
    return comparators

def construct():
    start_time = time.time()
    deadline = start_time + 70.0  # Safety margin before 90s kill
    
    # Build Batcher's odd-even merge sort network
    network = _batcher_odd_even_merge_sort(N)
    
    # Verify it's correct
    if not _verify_network(network, N):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Remove redundant comparators
    pruned = _remove_rendundant(network, N, deadline)
    
    # Final verification
    if not _verify_network(pruned, N):
        return network
    
    return pruned
# EVOLVE-BLOCK-END
