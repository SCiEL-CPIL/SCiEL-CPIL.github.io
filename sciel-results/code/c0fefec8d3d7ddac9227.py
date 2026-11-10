import time
import random
import math

def construct():
    """
    Maximize cardinality of a subset S of [1, 256] such that for all a < b <= c in S,
    a does not divide b + c.
    
    Strategy:
    1. Start with a strong initial set (upper tail + greedy additions).
    2. Perform local search: try adding elements, and if stuck, try swaps (remove one, add another).
    3. Use multiple random restarts with different initial configurations.
    4. Deadline guard to ensure we return before 90s.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    best_set = set()
    best_size = 0
    
    def is_valid_addition(S, x):
        """Check if x can be added to S while maintaining the property."""
        if x in S:
            return False
        # For all a < b <= c in S ∪ {x}, a must not divide b+c.
        # Since x is new, we only need to check constraints involving x.
        # Case 1: x is the smallest (a=x). Then for all b < c in S with x < b <= c, x must not divide b+c.
        # Case 2: x is a middle or largest element. 
        #   - If x = b: for all a < x in S and c >= x in S (c can be x if we allow b=c? No, b<=c and a<b<=c. If b=x, then a<x<=c. c can be x or >x. But c must be in S∪{x}. Since x is being added, c can be x or any element in S that is >= x.
        #   - If x = c: for all a < b in S with b <= x, a must not divide b+x.
        
        S_list = sorted(S)
        
        # Precompute for efficiency? n=256, |S| up to ~100-150. 
        # Let's just iterate.
        
        # 1. x as 'a': x < b <= c, b,c in S.
        for i in range(len(S_list)):
            b = S_list[i]
            if b <= x:
                continue
            for j in range(i, len(S_list)):
                c = S_list[j]
                # b <= c is guaranteed by sorted order if j>=i
                if (b + c) % x == 0:
                    return False
        
        # 2. x as 'b' or 'c'. 
        # Actually, it's easier to just check all triples involving x.
        # The condition is: for all a < b <= c in T (T = S ∪ {x}), a ∤ (b+c).
        # The only new triples are those containing x.
        # Since x is the largest or not, let's just iterate through pairs in S and check if adding x creates a violation.
        
        # Violation types involving x:
        # A) x = a: Need x < b <= c in S. x ∤ (b+c).
        #    (Already checked above)
        
        # B) x = b: Need a < x <= c. a in S, c in S∪{x}.
        #    Subcase B1: c = x. Then a < x <= x. a in S. x ∤ (x+x) = 2x. Always false since 2x % x == 0. 
        #    Wait, if a < b=x and c=x, then a < x. Condition: a ∤ (x+x) = 2x. 
        #    This is a violation if a divides 2x.
        #    So for any a in S with a < x, if a divides 2x, we cannot add x?
        #    Wait, the condition is a ∤ (b+c). Here b=x, c=x. So a ∤ 2x.
        #    If a divides 2x, it's a violation.
        
        #    Subcase B2: c in S, c > x. Then a < x < c. a in S. Check a ∤ (x+c).
        
        # C) x = c: Need a < b <= x. a,b in S. Check a ∤ (b+x).
        
        # Let's re-verify B1. If we add x, and we have a in S with a < x.
        # The triple (a, x, x) is in T. Condition: a ∤ (x+x).
        # So if any a in S (with a < x) divides 2x, we CANNOT add x.
        
        # Let's check B1 first, it's a quick filter.
        for a in S_list:
            if a >= x:
                break
            if (2 * x) % a == 0:
                return False
        
        # Check B2: a < x <= c, c in S, c > x.
        for i in range(len(S_list)):
            a = S_list[i]
            if a >= x:
                break
            for j in range(len(S_list)):
                c = S_list[j]
                if c <= x:
                    continue
                # a < x < c. Check a ∤ (x+c)
                if (x + c) % a == 0:
                    return False
        
        # Check C: a < b <= x, a,b in S.
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i, len(S_list)):
                b = S_list[j]
                if b > x:
                    break
                # a < b <= x. Check a ∤ (b+x)
                if (b + x) % a == 0:
                    return False
        
        return True

    def try_add_all(S):
        """Greedy add all possible elements."""
        changed = True
        while changed:
            changed = False
            # Iterate in increasing order? Or random? 
            # Increasing order might be better to keep small numbers out if they cause issues, 
            # but actually we want to ADD numbers. 
            # Let's try adding in increasing order first.
            for x in range(1, n + 1):
                if x not in S and is_valid_addition(S, x):
                    S.add(x)
                    changed = True
        return S

    def local_search(S, max_iters=1000):
        """
        Try to improve S by adding elements or swapping.
        """
        S = set(S)
        # First, fill with greedy
        try_add_all(S)
        
        improved = True
        while improved and (time.time() < deadline - 5.0):
            improved = False
            # Try adding any remaining element
            for x in range(1, n + 1):
                if x not in S and is_valid_addition(S, x):
                    S.add(x)
                    improved = True
                    break
            
            if not improved:
                # Try swaps: remove y from S, add x not in S
                # Heuristic: remove a large y that might be blocking many small x?
                # Or just random swaps.
                S_list = sorted(S)
                # Try removing each element and see if we can add more
                for y in S_list:
                    S_temp = S - {y}
                    # Try to add any x
                    for x in range(1, n + 1):
                        if x not in S_temp and x != y and is_valid_addition(S_temp, x):
                            # Check if size increases? 
                            # If we remove y and add x, size is same. 
                            # But maybe it allows further additions.
                            # Let's just do the swap if it allows adding at least one more later?
                            # Simpler: just do the swap and then greedy fill.
                            S_temp.add(x)
                            try_add_all(S_temp)
                            if len(S_temp) > len(S):
                                S = S_temp
                                improved = True
                                break
                    if improved:
                        break
        
        return S

    # Initial Solution 1: Upper Tail
    S1 = set(range(192, n + 1))
    S1 = local_search(S1)
    if len(S1) > best_size:
        best_set = S1
        best_size = len(S1)
    
    if time.time() > deadline - 10.0:
        return list(best_set)

    # Initial Solution 2: Greedy from scratch
    S2 = set()
    S2 = local_search(S2)
    if len(S2) > best_size:
        best_set = S2
        best_size = len(S2)
        
    if time.time() > deadline - 10.0:
        return list(best_set)

    # Random Restarts
    random.seed(42)
    while time.time() < deadline - 5.0:
        # Random initial set
        S_rand = set()
        # Start with some random large numbers
        for _ in range(20):
            x = random.randint(100, n)
            if is_valid_addition(S_rand, x):
                S_rand.add(x)
        
        S_rand = local_search(S_rand, max_iters=200)
        
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(S_rand)
        
        # Also try a different random start: small numbers
        S_small = set()
        for x in range(1, n + 1):
            if random.random() < 0.1 and is_valid_addition(S_small, x):
                S_small.add(x)
        S_small = local_search(S_small, max_iters=200)
        
        if len(S_small) > best_size:
            best_set = S_small
            best_size = len(S_small)

    return list(best_set)
