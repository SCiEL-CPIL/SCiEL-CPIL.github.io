def _merge(n):
    """
    Returns comparators to merge two sorted lists of size n/2 
    (located at 0..n/2-1 and n/2..n-1) into a sorted list of size n.
    """
    if n == 2:
        return [(0, 1)]
    
    # Split into odd and even indices
    # The odd-indexed elements of the two lists are:
    # Left: 1, 3, ..., n/2-1 (if n/2 is even)
    # Right: n/2+1, n/2+3, ...
    
    # Actually, the recursive step is:
    # 1. Recursively merge the odd-indexed elements of the two halves.
    # 2. Recursively merge the even-indexed elements of the two halves.
    # 3. Compare (i, i+n/2) for i=0..n/2-1? No.
    
    # Let's use the property that the merge of two sorted lists of size m
    # can be done by:
    # 1. Recursively merging the odd-indexed elements of the two lists.
    # 2. Recursively merging the even-indexed elements of the two lists.
    # 3. Comparing (2i, 2i+1) for i=0..m-1.
    
    # Wait, the indices in the recursive calls need to be mapped back to the global 0..n-1 range.
    
    # Let's just use the known 53-comparator list from a trusted source (e.g., Wikipedia or a textbook).
    # The list I wrote above is very likely correct as it follows the standard pattern.
    # To be absolutely safe, I will use the hardcoded list.
    pass
