import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 256
    
    def is_valid_interval(L):
        """
        Checks if the interval [L, n] is a valid set.
        Condition: for all a in [L, n], b, c in [a+1, n] with b<=c, a does not divide b+c.
        """
        # We can check this by iterating a from L to n.
        # For a given a, we need to ensure no multiple of a is a sum of two numbers in [a+1, n].
        # Let S = {a+1, ..., n}.
        # Possible sums range from 2(a+1) to 2n.
        # If there is a multiple k*a in [2(a+1), 2n], is it always representable?
        # Yes, any sum S in [2(a+1), 2n] can be represented as b+c with a+1 <= b <= c <= n.
        # Proof: Let S be the sum. We need b+c=S, a+1 <= b <= c <= n.
        # Min b is max(a+1, S-n). Max b is min(n, S//2).
        # We need max(a+1, S-n) <= min(n, S//2).
        # Since S <= 2n, S-n <= n.
        # Since S >= 2(a+1), S//2 >= a+1.
        # The critical case is when the interval [S-n, S//2] does not intersect [a+1, n].
        # However, for intervals, it's known that all sums in the range are representable 
        # if the set is contiguous.
        # So, the interval is invalid if there exists a multiple of a in [2(a+1), 2n].
        
        for a in range(L, n + 1):
            min_sum = 2 * (a + 1)
            max_sum = 2 * n
            # Check if there is a multiple of a in [min_sum, max_sum]
            # First multiple >= min_sum
            k = (min_sum + a - 1) // a
            if k * a <= max_sum:
                return False
        return True

    def can_add(x, S):
        """
        Checks if x can be added to set S, assuming x < min(S).
        Condition: for all b, c in S with b <= c, x does not divide b+c.
        """
        S_list = sorted(S)
        # Optimization: if x is large, fewer multiples.
        # We need to check if any b+c is divisible by x.
        # b, c in S.
        # We can just iterate. S size is at most 256.
        # Pairs (b,c) is O(|S|^2) ~ 65k.
        # This is fast enough.
        for i in range(len(S_list)):
            b = S_list[i]
            for j in range(i, len(S_list)):
                c = S_list[j]
                if (b + c) % x == 0:
                    return False
        return True

    best_set = set()
    best_size = 0
    
    # Strategy 1: Find best interval
    # We expect L to be around 120-180.
    # Let's scan L from 100 to 256.
    for L in range(100, n + 1):
        if time.time() > deadline:
            break
            
        if is_valid_interval(L):
            current_set = set(range(L, n + 1))
            
            # Try to add smaller elements
            # Greedy: try adding x from L-1 down to 1
            # Or try adding x from 1 up to L-1?
            # Adding smaller x is harder (more restrictions).
            # Adding larger x (closer to L) is easier.
            # So iterate x from L-1 down to 1.
            for x in range(L - 1, 0, -1):
                if time.time() > deadline:
                    break
                if can_add(x, current_set):
                    current_set.add(x)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)

    # Strategy 2: Local Search / Swaps?
    # The greedy might get stuck.
    # Let's try a different order of addition for the best interval found.
    # Actually, the choice of interval L is the main factor.
    # Let's also try to start with a slightly larger set and remove elements?
    # No, the interval approach is strong.
    
    # Let's try to improve the best_set found by trying to add elements 
    # that were skipped in the greedy pass, but this time checking against the 
    # updated set? No, the greedy pass already did that.
    
    # What if we try multiple L values and for each, try different greedy orders?
    # The greedy order "L-1 down to 1" is standard.
    # Let's try "1 up to L-1" as well?
    # Adding small numbers first might block larger ones?
    # If x is small, it restricts many sums. If we add it, we might block y > x.
    # If we add y first, y restricts sums involving y.
    # Generally, adding larger numbers first is safer because they are less restrictive 
    # as 'a' (larger a means fewer multiples in the sum range).
    # Wait, if x is 'a', larger x means larger multiples.
    # If x is small, x divides many sums.
    # So small x are "expensive" to add.
    # Therefore, we should add "cheap" elements first?
    # No, if we add an expensive element, it might block others.
    # If we add a cheap element, it might not block as many.
    # But actually, the condition is symmetric in b,c but asymmetric in a.
    # The "cost" of adding x is the number of pairs (b,c) it forbids.
    # We want to maximize the set size.
    # This is a Max Independent Set problem on a conflict graph.
    # Nodes: 1..256.
    # Edge (x, y) if x cannot be in the set with y?
    # No, the constraint is 3-wise.
    # It's a 3-uniform hypergraph independent set.
    
    # Given the time, the interval + greedy is likely optimal or near-optimal.
    # Let's ensure we check all L from 1 to 256, but optimize the check.
    
    # Re-run with full L range if time permits
    if time.time() < deadline - 20.0:
        for L in range(1, 100):
            if time.time() > deadline:
                break
            if is_valid_interval(L):
                current_set = set(range(L, n + 1))
                for x in range(L - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    if can_add(x, current_set):
                        current_set.add(x)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)

    if not best_set:
        return list(range(192, 257))

    return list(best_set)
