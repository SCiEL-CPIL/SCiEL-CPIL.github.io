"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import bisect

def construct():
    n = 256
    
    def is_valid_addition(S, x):
        """
        Check if x can be added to sorted list S without violating the property.
        Property: For all a < b <= c in S U {x}, a does not divide b+c.
        S is assumed to be sorted and not contain x.
        """
        # Find indices to limit loops
        # idx_ge_x: first index where S[i] >= x
        idx_ge_x = bisect.bisect_left(S, x)
        # idx_gt_x: first index where S[i] > x
        idx_gt_x = bisect.bisect_right(S, x)
        
        # Case 1: x is 'a' (smallest in the triple)
        # Constraint: For all b, c in S with x < b <= c, x does not divide b+c
        # b must be > x, so start from idx_gt_x
        for i in range(idx_gt_x, len(S)):
            b = S[i]
            for j in range(i, len(S)):
                c = S[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b' (middle in the triple)
        # Constraint: For all a, c in S with a < x <= c, a does not divide x+c
        # a must be < x, so a is in S[0:idx_ge_x]
        # c must be >= x, so c is in S[idx_ge_x:]
        for i in range(idx_ge_x):
            a = S[i]
            for j in range(idx_ge_x, len(S)):
                c = S[j]
                if (x + c) % a == 0:
                    return False

        # Case 3: x is 'c' (largest in the triple)
        # Constraint: For all a, b in S with a < b <= x, a does not divide b+x
        
        # Subcase 3a: b = x
        # a must be < x, so a is in S[0:idx_ge_x]
        for i in range(idx_ge_x):
            a = S[i]
            if (x + x) % a == 0:
                return False

        # Subcase 3b: b < x
        # b is in S[0:idx_ge_x] (since b < x)
        # a is in S[0:i] where i is index of b
        for i in range(idx_ge_x):
            b = S[i]
            for j in range(i):
                a = S[j]
                if (b + x) % a == 0:
                    return False

        return True

    # Strategy: Greedy from largest to smallest
    # Large numbers are less likely to be divisors of sums of other numbers,
    # so adding them first is less restrictive for future additions.
    selected = []
    for num in range(n, 0, -1):
        if is_valid_addition(selected, num):
            selected.append(num)
    
    selected.sort()
    
    # Iterative improvement: try to add any skipped number
    # This can be done in multiple passes until no more numbers can be added
    improved = True
    while improved:
        improved = False
        # Create a set for O(1) lookup
        selected_set = set(selected)
        for num in range(1, n+1):
            if num not in selected_set:
                if is_valid_addition(selected, num):
                    selected.append(num)
                    selected.sort()
                    selected_set.add(num)
                    improved = True
    
    return selected
# EVOLVE-BLOCK-END
