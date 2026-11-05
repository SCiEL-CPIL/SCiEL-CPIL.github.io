"""Improved program: extend the upper tail and greedily add lower elements."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    deadline = time.time() + 85.0  # safety margin before 90s kill

    # The known good solution: [171, 256]
    # For any a >= 171 in the set, and any b, c in the set with a < b <= c:
    #   b + c <= 256 + 256 = 512
    #   3 * a >= 3 * 171 = 513 > 512
    # So a cannot divide b + c (since b+c < a would mean a doesn't divide it,
    # and b+c >= a would require b+c >= 2a >= 342, but we need a | (b+c),
    # and the smallest multiple of a that's >= b is 2a, but b+c < 3a means
    # b+c can only be 2a if b+c == 2a. Let's verify: if a >= 171, 2a >= 342,
    # and b+c <= 512. So b+c could equal 2a. We need to check that no two
    # elements b,c in the set satisfy b+c = 2a for some a in the set with a < b.
    #
    # Actually, let's be more careful. The condition is: for every a < b <= c in
    # the set, a does NOT divide b+c.
    #
    # For the interval [L, n] where L >= 171:
    # - a is in [L, n], b and c are in [L, n] with a < b <= c
    # - b + c is in [2L+1, 2n] (since b >= L+1, c >= b >= L+1, so b+c >= 2L+2... wait)
    #   Actually b > a >= L, so b >= L+1, c >= b >= L+1, so b+c >= 2(L+1)
    #   And b+c <= 2n
    # - a divides b+c means b+c = k*a for some integer k >= 1
    # - Since b+c >= 2(L+1) and a >= L, k*a >= 2(L+1), so k >= 2(L+1)/L >= 2 + 2/L > 2
    #   Wait, k*a = b+c. Since a < b, b+c > 2a... no, b+c > a + a = 2a? No:
    #   b > a, c >= b > a, so b+c > 2a. Thus k*a > 2a, so k >= 3.
    # - k*a >= 3a >= 3L
    # - b+c <= 2n
    # - So we need 3L > 2n for the interval to be valid.
    # - 3*171 = 513 > 512 = 2*256. Yes!
    #
    # So [171, 256] is valid. Can we extend further?
    # 3L > 512 => L > 170.67, so L >= 171. That's tight.
    #
    # But we can try to add elements below 171 if they don't create conflicts.
    
    # Start with the base set
    base_set = set(range(171, 257))
    
    # Try to add elements from 1 to 170
    # For a candidate element x < 171, we need to check:
    # 1. For all a, b, c in current set with a < b <= c, x doesn't create new violations
    #    - If x is the smallest: for all b < c in set, x does not divide b+c
    #    - If x is in the middle: for all a < x in set and b, c in set with x < b <= c, 
    #      a does not divide x+b (when c=x... no, we need a < b <= c, so if x is b or c)
    #    - More precisely, adding x to set S creates new triples where x is a, b, or c:
    #      * x as a: for all b, c in S with x < b <= c, x does not divide b+c
    #      * x as b: for all a in S with a < x, and c in S with x <= c, a does not divide x+c
    #      * x as c: for all a, b in S with a < b <= x, a does not divide b+x
    #      * x as b and c (b=c=x): for all a in S with a < x, a does not divide 2x
    
    # Build a conflict graph approach: for each pair (b,c) with b <= c, determine which
    # a values would conflict (a divides b+c and a < b). Then an element x can be added
    # if it doesn't conflict with any existing pair.
    
    # Precompute: for the base set, which pairs (b,c) exist and what a's conflict?
    # Actually, let's just do a greedy approach: try adding each element from 1 to 170
    # in some order, checking validity.
    
    def is_valid_set(s):
        """Check if set s satisfies the property."""
        elems = sorted(s)
        for i, a in enumerate(elems):
            for j in range(i + 1, len(elems)):
                b = elems[j]
                for k in range(j, len(elems)):
                    c = elems[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Greedy: try to add elements from 1 to 170, in order of "least conflict"
    # For each candidate x, count how many conflicts it would create
    def count_conflicts(x, s):
        """Count conflicts if x is added to set s."""
        count = 0
        for a in s:
            if a >= x:
                continue
            # x as b or c
            for b in s:
                if b <= x:
                    for c in s:
                        if c >= b and c >= x:
                            if (b + c) % a == 0 and a < b:
                                # This conflict exists only if it's a NEW conflict
                                # involving x
                                pass
        # Simpler: just check the new triples involving x
        for a in s:
            if a >= x:
                continue
            # x as b: a < x, need c >= x in s, check a | (x + c)
            for c in s:
                if c >= x:
                    if (x + c) % a == 0:
                        count += 1
            # x as c: a < b <= x, b in s with b <= x, check a | (b + x)
            for b in s:
                if b <= x:
                    if a < b and (b + x) % a == 0:
                        count += 1
            # x as b=c: a < x, check a | (2x)
            if (2 * x) % a == 0:
                count += 1
        # x as a: for all b, c in s with x < b <= c, check x | (b+c)
        for b in s:
            if b <= x:
                continue
            for c in s:
                if c >= b:
                    if (b + c) % x == 0:
                        count += 1
        return count
    
    current_set = set(base_set)
    
    # Try adding elements from 1 to 170, prioritizing those with fewer conflicts
    candidates = list(range(1, 171))
    
    # Sort by number of conflicts (ascending)
    def sort_key(x):
        return count_conflicts(x, current_set)
    
    # Do a few passes, adding elements that have zero conflicts
    for _pass in range(10):
        if time.time() > deadline:
            break
        # Sort candidates by conflict count
        candidates.sort(key=lambda x: count_conflicts(x, current_set))
        added_any = False
        for x in candidates[:]:
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            if count_conflicts(x, current_set) == 0:
                # Verify by actually checking
                test_set = current_set | {x}
                if is_valid_set(test_set):
                    current_set = test_set
                    candidates.remove(x)
                    added_any = True
        if not added_any:
            break
    
    # Try a different ordering: from 170 down to 1
    if time.time() < deadline - 5:
        for x in range(170, 0, -1):
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            test_set = current_set | {x}
            if is_valid_set(test_set):
                current_set = test_set
    
    # Try from 1 up to 170
    if time.time() < deadline - 5:
        for x in range(1, 171):
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            test_set = current_set | {x}
            if is_valid_set(test_set):
                current_set = test_set
    
    # Final verification
    if not is_valid_set(current_set):
        # Fallback to known good set
        return list(range(171, 257))
    
    return sorted(current_set)
# EVOLVE-BLOCK-END
