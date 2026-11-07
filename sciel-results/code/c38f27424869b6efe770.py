import time

def construct():
    start_time = time.time()
    deadline = start_time + 50

    # Structural change: greedy construction from small to large.
    # Key insight: if we include all numbers from some threshold T to 128,
    # then pair sums of two elements >= T are >= 2T. If 2T > 128, no sums
    # from the large part are in the set. But we can also include small numbers
    # carefully.
    
    # Strategy: Start with a high threshold, then greedily add smaller numbers
    # if they don't create a forbidden triple with existing elements.
    
    # First, establish a safe baseline: {T, T+1, ..., 128} where T is chosen
    # so that 2*T > 128, i.e., T > 64. So T=65 gives size 64.
    # Actually T=64: 64+65=129>128, so {64,...,128} size 65 is valid.
    # T=63: 63+64=127, 63+65=128. Sums in set: {127,128}. Only 2 sums, need 3. Valid.
    # T=62: 62+63=125, 62+64=126, 62+65=127, 62+66=128. 
    #   63+64=127, 63+65=128. 
    #   Check triple (62,63,64): sums 125,126,127. Are all in {62..128}? Yes! Violation.
    # So we need to be more careful.
    
    # Greedy approach: build set S starting from empty, add numbers 1..128 in order,
    # but skip a number if adding it would create a forbidden triple.
    # A forbidden triple (a,b,c) with a<b<c requires a+b, a+c, b+c all in S.
    # When we add x, we need to check if there exist a,b already in S with a<b<x
    # such that a+b, a+x, b+x are all in S (after adding x).
    # Actually simpler: when adding x, check all pairs (a,b) in S with a<b and a+b < x
    # (since we process in order, sums that would be created by x are x+a and x+b,
    # and a+b must already be in S). So violation if there exist a<b in S, a+b in S,
    # and a+x in S and b+x in S... wait, a+x and b+x aren't in S yet unless they were
    # already there from previous additions.
    
    # Let me reconsider. S is built incrementally. When we consider adding x:
    # A triple (a,b,c) is forbidden if a+b, a+c, b+c all in S.
    # If we add x, new triples involving x are (a,b,x) where a,b in S, a<b<x.
    # For (a,b,x): need a+b in S, a+x in S, b+x in S.
    # a+b is already in S (or not). a+x and b+x: these are > x (since a,b >= 1),
    # so they haven't been added yet (we're going in increasing order). 
    # Wait, that's not right either. a+x > x, so if we're adding in order 1,2,3,...,
    # then a+x hasn't been considered yet. So a+x is NOT in S at the time we add x.
    # Therefore, adding x can NEVER create a new forbidden triple where x is the largest!
    
    # What if x is not the largest? That can't happen if we add in increasing order.
    
    # So the key insight: if we add numbers in increasing order, adding x only creates
    # triples where x is the largest element. For such a triple (a,b,x) with a<b<x,
    # we need a+x in S and b+x in S. But a+x > x and b+x > x, so they haven't been
    # added yet. Hence, NO new forbidden triple is created!
    
    # Wait, that means ANY set is valid if we don't care about the order? No, that's
    # wrong. The issue is that a+x might already be in S if a+x < x, which is impossible
    # since a >= 1. So indeed, when building in increasing order, adding x never creates
    # a forbidden triple because the sums involving x are all larger than x and thus not yet in S.
    
    # This means we can include ALL numbers 1..128! But that's clearly wrong because
    # (1,2,3): 1+2=3 in S, 1+3=4 in S, 2+3=5 in S. All in {1..128}. Violation!
    
    # I made an error. When we add 3, a=1, b=2 are in S. a+b=3. But 3 is x itself, not a+x.
    # For triple (a,b,x) = (1,2,3): need a+b=3 in S (yes, it's x), a+x=4 in S (no, 4 not added yet), b+x=5 in S (no).
    # So actually (1,2,3) is NOT a violation when we add 3, because 4 and 5 aren't in S yet.
    # But later when we add 4 and 5, do we create the triple? No, because the triple (1,2,3)
    # was already "formed" when 3 was added, but at that time 4 and 5 weren't in S.
    
    # The violation is a property of the final set, not the construction order. So my reasoning
    # about construction order doesn't help directly.
    
    # Let me think differently. The constraint is: no three distinct a,b,c in S such that
    # a+b, a+c, b+c are ALL in S.
    
    # Key structural insight: if S contains no three elements where all three pair sums are also in S.
    # 
    # Consider the set of all numbers > 64: {65,...,128}, size 64. Any pair sum >= 65+66=131 > 128,
    # so no pair sums are in S. Valid.
    #
    # Can we add some numbers <= 64? If we add x <= 64, then for any a,b in {65,...,128},
    # x+a and x+b are in [66, 192]. Those > 128 are fine. Those <= 128 could be in S.
    # Also a+b > 128 for a,b > 64.
    # 
    # A forbidden triple could be (x, a, b) with x <= 64, a,b > 64, x < a < b.
    # Need x+a in S, x+b in S, a+b in S. But a+b > 128, so a+b not in S. No violation.
    #
    # What about (x, y, a) with x,y <= 64, a > 64, x < y < a?
    # Need x+y in S, x+a in S, y+a in S.
    # x+y <= 128, could be in S. x+a in [66,192], if <=128 could be in S. Similarly y+a.
    #
    # What about (x, y, z) all <= 64?
    # Need x+y, x+z, y+z all in S.
    
    # Greedy: start with {65,...,128}, then try to add numbers 1..64 in some order,
    # checking if adding creates a forbidden triple.
    
    S = set(range(65, 129))
    
    def would_violate(S, x):
        """Check if adding x to S creates a forbidden triple."""
        S_new = S | {x}
        S_list = sorted(S_new)
        n = len(S_list)
        # Only need to check triples involving x
        for i in range(n):
            for j in range(i+1, n):
                a, b = S_list[i], S_list[j]
                if a == x and b == x:
                    continue
                # Find c
                # Triple is (a,b,c) with c > b, or x is one of them
                # Simpler: check all pairs in S (not including x) and see if (pair, x) is bad
                pass
        
        # More efficient: for each pair (a,b) in S (original, not including x),
        # check if (a,b,x) is a forbidden triple, assuming x > b or x is between them.
        # Actually, just check all triples involving x.
        S_orig = list(S)
        for i in range(len(S_orig)):
            for j in range(i+1, len(S_orig)):
                a, b = S_orig[i], S_orig[j]
                # Triple (a, b, x) - need all three pair sums in S_new
                s1 = a + b
                s2 = a + x
                s3 = b + x
                if s1 in S_new and s2 in S_new and s3 in S_new:
                    return True
        return False
    
    # Try adding numbers from 64 down to 1
    for x in range(64, 0, -1):
        if time.time() > deadline - 5:
            break
        if not would_violate(S, x):
            S.add(x)
    
    result = sorted(S)
    return result

def is_valid(S):
    S_set = set(S)
    S_list = sorted(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i+1, n):
            for l in range(j+1, n):
                a, b, c = S_list[i], S_list[j], S_list[l]
                if a+b in S_set and a+c in S_set and b+c in S_set:
                    return False
    return True
