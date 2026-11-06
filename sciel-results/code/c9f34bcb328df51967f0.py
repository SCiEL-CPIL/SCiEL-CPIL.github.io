import itertools

def count_3aps_in_set(S, x):
    """Count number of 3-APs in S that include x."""
    count = 0
    # x as first term: x, x+d, x+2d
    # x as second term: x-d, x, x+d
    # x as third term: x-2d, x-d, x
    for d in range(1, 257):
        # x is first
        if (x + d) in S and (x + 2 * d) in S:
            count += 1
        # x is second
        if (x - d) in S and (x + d) in S:
            count += 1
        # x is third
        if (x - 2 * d) in S and (x - d) in S:
            count += 1
    return count

def creates_4ap(S, x):
    """Check if adding x to S creates a 4-AP."""
    # x is 1st term: x, x+d, x+2d, x+3d
    for d in range(1, 257):
        if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
            return True
        # x is 2nd term: x-d, x, x+d, x+2d
        if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
            return True
        # x is 3rd term: x-2d, x-d, x, x+d
        if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
            return True
        # x is 4th term: x-3d, x-2d, x-d, x
        if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
            return True
    return False

def find_one_4ap(S):
    """Find one 4-AP in S. Returns tuple of 4 elements or None."""
    # Iterate through all possible 4-APs
    # a, a+d, a+2d, a+3d
    # a >= 1, a+3d <= 256
    for a in range(1, 257):
        if a not in S:
            continue
        max_d = (256 - a) // 3
        for d in range(1, max_d + 1):
            a1 = a
            a2 = a + d
            a3 = a + 2 * d
            a4 = a + 3 * d
            if a2 in S and a3 in S and a4 in S:
                return (a1, a2, a3, a4)
    return None

def construct():
    n = 256
    all_elements = set(range(1, n + 1))
    A = set(all_elements)
    
    # Phase 1: Remove elements to eliminate all 4-APs
    # We do this iteratively.
    for _ in range(10): # Repeat a few times to potentially find better removals
        while True:
            ap = find_one_4ap(A)
            if not ap:
                break
            # Find the element in the AP with the fewest 3-APs
            min_count = float('inf')
            min_elem = None
            for elem in ap:
                c = count_3aps_in_set(A, elem)
                if c < min_count:
                    min_count = c
                    min_elem = elem
            A.discard(min_elem)
            
        # Phase 2: Try to add back elements
        # We can try to add any element not in A
        # To be efficient, we only try elements that were removed or just scan all
        # Since n is small, scanning all is fine.
        changed = False
        for x in range(1, n + 1):
            if x not in A:
                if not creates_4ap(A, x):
                    A.add(x)
                    changed = True
        if not changed:
            break
            
    return sorted(A)
