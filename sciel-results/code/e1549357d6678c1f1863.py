import time
from fractions import Fraction
import random

def construct():
    deadline = time.time() + 55
    random.seed(42)
    N = 256
    
    # Precompute for each x, the set of pairs (a,b) such that a,x,b form a 3-AP
    # For checking if adding x creates a 3-AP, we need:
    # - x as middle: a + b = 2x, so for each a in s, check if 2x-a in s
    # - x as endpoint: 2a - x in s for some a in s (x is the smaller endpoint)
    #   or 2b - x in s where x is the larger endpoint... actually:
    # If x is the first term: x, x+d, x+2d => need x+d and x+2d in s
    # If x is the middle: x-d, x, x+d => need x-d and x+d in s  
    # If x is the last term: x-2d, x-d, x => need x-2d and x-d in s
    
    # Better: for each x, precompute which pairs in s would conflict
    # When adding x to s, it conflicts if:
    # 1. There exist a,b in s with a < x < b and a+b=2x (x is middle)
    # 2. There exist a in s such that 2a-x is in s and 2a-x != a (x is an endpoint)
    #    Actually if a > x: a, (a+x)/2... no. Let me think again.
    # A 3-AP is (u, v, w) with u+w=2v, u<v<w.
    # If we add x, a new 3-AP is created if:
    # - x is the middle: exists a,b in s with a+x... no, a+b=2x with a,b in s, a!=b
    # - x is the first term: exists d>0 such that x+d in s and x+2d in s
    # - x is the last term: exists d>0 such that x-d in s and x-2d in s
    
    # For efficiency, let's use a different approach:
    # Precompute for each pair (i,j) with i<j, the middle m=(i+j)/2 if integer
    # Then when checking if x can be added, check all pairs involving x
    
    # Actually, let's just optimize has_3ap_with_x by precomputing conflicts
    # For each x, store which other numbers it would conflict with given the current set
    
    # Let's use a bitmask approach for speed
    # Represent the set as a 256-bit integer (using Python int)
    
    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x-1))
        return m
    
    def from_mask(m):
        s = set()
        for i in range(1, N+1):
            if m & (1 << (i-1)):
                s.add(i)
        return s
    
    # Precompute: for each x, the list of (a, b) pairs where {a,x,b} forms a 3-AP
    # a < x < b, a + b = 2x
    # Also for x as endpoint: 
    #   x as first: (x, x+d, x+2d) for d=1,...
    #   x as last: (x-2d, x-d, x) for d=1,...
    
    # For has_3ap_with_x(mask, x):
    # Check if adding x creates any 3-AP
    # Case 1: x is middle. For each a in mask with a < x, check if b=2x-a is in mask and b > x
    # Case 2: x is first term. For each d, check if x+d in mask and x+2d in mask
    # Case 3: x is last term. For each d, check if x-d in mask and x-2d in mask
    
    def has_3ap_with_x(mask, x):
        # x is middle
        for a in range(1, x):
            if mask & (1 << (a-1)):
                b = 2*x - a
                if b > x and b <= N and (mask & (1 << (b-1))):
                    return True
        # x is first term: x, x+d, x+2d
        d = 1
        while x + 2*d <= N:
            if (mask & (1 << (x+d-1))) and (mask & (1 << (x+2*d-1))):
                return True
            d += 1
        # x is last term: x-2d, x-d, x
        d = 1
        while x - 2*d >= 1:
            if (mask & (1 << (x-d-1))) and (mask & (1 << (x-2*d-1))):
                return True
            d += 1
        return False
    
    def greedy_construct_priority():
        """Greedy that prioritizes smaller numbers (higher reciprocal value)"""
        mask = 0
        for x in range(1, N+1):
            if not has_3ap_with_x(mask, x):
                mask |= (1 << (x-1))
        return from_mask(mask)

    def greedy_construct_descending():
        """Greedy that tries larger numbers first (less restrictive)"""
        mask = 0
        for x in range(N, 0, -1):
            if not has_3ap_with_x(mask, x):
                mask |= (1 << (x-1))
        return from_mask(mask)
    
    def greedy_construct_weighted():
        """Greedy that tries numbers in order of 1/x but with lookahead"""
        mask = 0
        for x in range(1, N+1):
            if not has_3ap_with_x(mask, x):
                mask |= (1 << (x-1))
        return from_mask(mask)
    
    def local_search(s, start_time, max_time=25):
        mask = to_mask(s)
        best_sum = sum(Fraction(1, x) for x in s)
        all_mask = (1 << N) - 1
        
        while time.time() < start_time + max_time:
            improved = False
            
            # 1. Add: try to add smallest available numbers first
            for x in range(1, N+1):
                if time.time() >= start_time + max_time: break
                if not (mask & (1 << (x-1))):
                    if not has_3ap_with_x(mask, x):
                        mask |= (1 << (x-1))
                        best_sum += Fraction(1, x)
                        improved = True
            if improved:
                continue
                
            # 2. Swap: remove a larger number, add a smaller one if possible
            # Try removing each element and see if we can add a better one
            removed = False
            for rem in range(N, 0, -1):
                if time.time() >= start_time + max_time: break
                if not (mask & (1 << (rem-1))):
                    continue
                test_mask = mask & ~(1 << (rem-1))
                best_add = None
                best_gain = Fraction(0)
                for x in range(1, rem):
                    if time.time() >= start_time + max_time: break
                    if test_mask & (1 << (x-1)):
                        continue
                    if not has_3ap_with_x(test_mask, x):
                        gain = Fraction(1, x) - Fraction(1, rem)
                        if gain > best_gain:
                            best_gain = gain
                            best_add = x
                if best_add is not None:
                    mask = test_mask | (1 << (best_add-1))
                    best_sum += best_gain
                    improved = True
                    removed = True
                    break
            
            # 3. Double swap: remove two, add two
            if not improved:
                s_list = from_mask(mask)
                if len(s_list) >= 2:
                    to_remove = sorted(s_list, reverse=True)
                    found = False
                    for i in range(len(to_remove)):
                        if time.time() >= start_time + max_time: break
                        for j in range(i+1, len(to_remove)):
                            if time.time() >= start_time + max_time: break
                            rem1, rem2 = to_remove[i], to_remove[j]
                            test_mask = mask & ~(1 << (rem1-1)) & ~(1 << (rem2-1))
                            best_add1 = None
                            best_add2 = None
                            best_gain = Fraction(0)
                            for x1 in range(1, min(rem1, rem2)):
                                if time.time() >= start_time + max_time: break
                                if test_mask & (1 << (x1-1)):
                                    continue
                                if not has_3ap_with_x(test_mask, x1):
                                    test_mask2 = test_mask | (1 << (x1-1))
                                    for x2 in range(1, min(rem1, rem2)):
                                        if x2 == x1:
                                            continue
                                        if test_mask2 & (1 << (x2-1)):
                                            continue
                                        if not has_3ap_with_x(test_mask2, x2):
                                            gain = Fraction(1, x1) + Fraction(1, x2) - Fraction(1, rem1) - Fraction(1, rem2)
                                            if gain > best_gain:
                                                best_gain = gain
                                                best_add1 = x1
                                                best_add2 = x2
                            if best_add1 is not None and best_add2 is not None:
                                mask = test_mask | (1 << (best_add1-1)) | (1 << (best_add2-1))
                                best_sum += best_gain
                                improved = True
                                found = True
                                break
                        if found:
                            break
                
            if not improved:
                break
                
        return from_mask(mask), best_sum

    best_overall_set = set()
    best_overall_sum = Fraction(0)
    
    # Try multiple construction strategies
    strategies = [
        greedy_construct_priority,
        greedy_construct_descending,
    ]
    
    for strategy in strategies:
        if time.time() >= deadline - 10:
            break
        s0 = strategy()
        s1, sum1 = local_search(s0, time.time(), max_time=15)
        if sum1 > best_overall_sum:
            best_overall_set = s1
            best_overall_sum = sum1
    
    # If time remains, try random restarts with kicks
    while time.time() < deadline - 8:
        s_rand = greedy_construct_priority()
        if len(s_rand) > 3:
            items = list(s_rand)
            num_remove = random.randint(2, 5)
            for _ in range(num_remove):
                if items:
                    idx = random.randint(0, len(items) - 1)
                    s_rand.remove(items[idx])
                    items.pop(idx)
        
        s3, sum3 = local_search(s_rand, time.time(), max_time=12)
        if sum3 > best_overall_sum:
            best_overall_set = s3
            best_overall_sum = sum3
            
    return sorted(best_overall_set)
