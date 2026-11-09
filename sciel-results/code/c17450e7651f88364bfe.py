"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

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

def _verify_network_fast(comparators, n):
    """
    Verify a sorting network using the zero-one principle with vectorized operations.
    Returns True if the network sorts all 2^n boolean inputs.
    """
    num_inputs = 1 << n
    # Create input matrix: shape (num_inputs, n)
    # Input i is the binary representation of i
    inputs = np.array([[i >> j & 1 for j in range(n)] for i in range(num_inputs)], dtype=np.int8)
    
    # Apply comparators
    # Note: We must be careful with in-place operations if we want to preserve inputs for multiple checks,
    # but here we just want to verify the final state.
    for i, j in comparators:
        # Get values
        a = inputs[:, i]
        b = inputs[:, j]
        
        # Calculate min and max
        # min_val = where(a < b, a, b)
        # max_val = where(a < b, b, a)
        
        # np.minimum and np.maximum handle this
        min_val = np.minimum(a, b)
        max_val = np.maximum(a, b)
        
        # Update wires
        inputs[:, i] = min_val
        inputs[:, j] = max_val
        
    # Check if sorted
    # A vector is sorted if inputs[:, k] <= inputs[:, k+1] for all k
    # We can check if any adjacent pair is inverted: inputs[:, k] > inputs[:, k+1]
    # If any such pair exists for any input, it's not sorted.
    
    # Create a view of adjacent pairs
    # diff = inputs[:, 1:] - inputs[:, :-1]
    # If any diff < 0, not sorted.
    # Or simply:
    # sorted_mask = np.all(inputs[:, 1:] >= inputs[:, :-1], axis=1)
    # return np.all(sorted_mask)
    
    # More efficient:
    # Check if there exists any input and any k such that inputs[input, k] > inputs[input, k+1]
    # This is equivalent to checking if np.any(inputs[:, :-1] > inputs[:, 1:]) is True
    
    return not np.any(inputs[:, :-1] > inputs[:, 1:])

def _remove_redundant(comparators, n, deadline):
    """
    Remove redundant comparators from a valid sorting network.
    Uses vectorized verification for speed.
    """
    comparators = list(comparators)
    best_comparators = comparators
    
    # Pre-compute some info? No, network structure changes.
    
    while time.time() < deadline:
        removed_any = False
        indices = list(range(len(comparators)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Try removing comparator at idx
            candidate = comparators[:idx] + comparators[idx+1:]
            if _verify_network_fast(candidate, n):
                comparators = candidate
                removed_any = True
                if len(comparators) < len(best_comparators):
                    best_comparators = comparators
        
        if not removed_any:
            # Try multi-removal of parallel comparators?
            # If no single removal works, try removing pairs (i, j) where comparators[i] and comparators[j] are disjoint.
            # This is more expensive (O(M^2) checks) but might escape local minima.
            # Given time constraints, we might skip this or do it sparsely.
            # Let's try a few random pairs of disjoint comparators.
            
            # Build a list of disjoint pairs
            disjoint_pairs = []
            for i in range(len(comparators)):
                for j in range(i + 1, len(comparators)):
                    c1 = comparators[i]
                    c2 = comparators[j]
                    # Check if disjoint
                    if len(set([c1[0], c1[1]]) & set([c2[0], c2[1]])) == 0:
                        disjoint_pairs.append((i, j))
            
            if disjoint_pairs:
                random.shuffle(disjoint_pairs)
                # Try a limited number of these
                for i, j in disjoint_pairs[:100]: # Limit to avoid O(N^2) explosion if M is large
                    if time.time() > deadline:
                        break
                    # Remove both i and j. Note: if i < j, removing i shifts j.
                    # So we should remove the larger index first or adjust indices.
                    # Let's construct candidate carefully.
                    # To remove indices i and j (i < j):
                    # candidate = comparators[:i] + comparators[i+1:j] + comparators[j+1:]
                    candidate = comparators[:i] + comparators[i+1:j] + comparators[j+1:]
                    if _verify_network_fast(candidate, n):
                        comparators = candidate
                        removed_any = True
                        if len(comparators) < len(best_comparators):
                            best_comparators = comparators
            
            if not removed_any:
                break
                
    return best_comparators

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Build Batcher's odd-even merge sort network
    network = _batcher_odd_even_merge_sort(N)
    
    # Verify it's correct
    if not _verify_network_fast(network, N):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Remove redundant comparators
    pruned = _remove_redundant(network, N, deadline)
    
    # Final verification
    if not _verify_network_fast(pruned, N):
        return network
    
    return pruned
# EVOLVE-BLOCK-END
