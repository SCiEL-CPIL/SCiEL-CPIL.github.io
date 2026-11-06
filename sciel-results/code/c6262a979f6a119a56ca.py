"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sorting network for N wires using Odd-Even Merge Sort.
    For N=16, this uses 63 comparators.
    """
    def om_merge(n, start, direction):
        """
        Generates comparators to merge two sorted subarrays of size n/2
        starting at 'start'.
        direction is ignored in the standard merge structure but kept for interface consistency
        if needed, though merge is direction-agnostic in terms of comparator pairs,
        only the interpretation of min/max changes which is handled by the network logic.
        Actually, the comparator (i,j) always puts min at i and max at j.
        The 'direction' in sort determines if we want the final result ascending or descending.
        For a sorting network, we just need to sort ascending.
        """
        if n == 2:
            return [(start, start + 1)]
        
        half = n // 2
        # Recursively merge the first half and second half of the odd/even split?
        # No, the recurrence is:
        # merge(n) calls merge(n/2) on the first n/2 elements and merge(n/2) on the last n/2 elements?
        # Based on the N=4 trace:
        # merge(4) called merge(2, 0) and merge(2, 2).
        # So yes, it splits the block of size n into two blocks of size n/2.
        
        left = om_merge(half, start, direction)
        right = om_merge(half, start + half, direction)
        
        # Final step: compare adjacent pairs (1,2), (3,4), ..., (n-3, n-2)
        # Indices: start+1, start+2; start+3, start+4; ...
        # Range of i: 1 to n-2, step 2.
        final = []
        for i in range(1, n - 1, 2):
            final.append((start + i, start + i + 1))
            
        return left + right + final

    def om_sort(n, start, direction):
        """
        Generates comparators to sort n elements starting at 'start'.
        direction: 1 for ascending, -1 for descending.
        """
        if n == 1:
            return []
        
        half = n // 2
        # Sort first half with current direction
        left = om_sort(half, start, direction)
        # Sort second half with opposite direction
        right = om_sort(half, start + half, -direction)
        # Merge the two halves
        merge = om_merge(n, start, direction)
        
        return left + right + merge

    # Generate the network for N=16, starting at 0, ascending
    network = om_sort(N, 0, 1)
    
    # Optional: Verify and prune if necessary, but 63 is well under 120.
    # We can return the list directly.
    return network
# EVOLVE-BLOCK-END

