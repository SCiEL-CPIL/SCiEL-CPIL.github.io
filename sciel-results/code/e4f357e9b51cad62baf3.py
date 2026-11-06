import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    
    # Helper to check if x can be added to S
    # S is a list of elements. We assume S is valid.
    # We need to check if adding x keeps it valid.
    # x is the new element.
    # Condition: For all a < b <= c in S U {x}, a does not divide b+c.
    # Since we add x, the new triples involving x are:
    # 1. x is a: a=x, b,c in S. Need x < b <= c.
    # 2. x is b: a in S, b=x, c in S. Need a < x <= c.
    # 3. x is c: a in S, b in S, c=x. Need a < b <= x.
    
    # Note: The original greedy added in descending order, so x was always the smallest (a).
    # In random order, x can be anywhere.
    
    def can_add(x, S):
        # S is a list of ints.
        # Convert to set for fast lookup? No, we need pairs.
        # k = len(S)
        # Case 1: x is a.
        # Check if x divides b+c for any b,c in S with x < b <= c.
        # Case 2: x is b or c.
        # Check if a divides x+c or b+x for any a in S with a < x.
        
        k = len(S)
        # Sort S? S is not necessarily sorted in random greedy.
        # For efficiency, we can just iterate.
        
        # Case 1: x is a
        for i in range(k):
            b = S[i]
            if b <= x: continue
            for j in range(i, k):
                c = S[j]
                if c < b: continue # Ensure b <= c? No, S is unsorted.
                # We just need any pair {b,c} from S such that min(b,c) > x.
                # And x divides sum.
                # Simpler:
                # Iterate all pairs (u, v) in S.
                # If min(u,v) > x and x divides u+v, return False.
                pass
        
        # This O(k^2) is fine.
        # Let's implement it cleanly.
        
        # Pre-sort S?
        # S is small (< 256).
        
        # Check 1: x is a
        # Find all b, c in S such that b > x and c > x.
        # Check if x | b+c.
        # We can just iterate all pairs in S.
        for i in range(k):
            for j in range(i, k):
                b = S[i]
                c = S[j]
                if b > x and c > x:
                    if (b + c) % x == 0:
                        return False
        
        # Check 2: x is b or c
        # Find all a in S such that a < x.
        # For each such a, check if a divides x+c (where c in S, c >= x)
        # or a divides b+x (where b in S, b <= x, b > a).
        
        # Actually, it's easier to just check all triples in S U {x}?
        # No, S is valid. Only triples involving x need checking.
        
        # Triples involving x:
        # (x, b, c) with b, c in S, x < b <= c.
        # (a, x, c) with a, c in S, a < x <= c.
        # (a, b, x) with a, b in S, a < b <= x.
        
        # Case 1: (x, b, c)
        # Covered above.
        
        # Case 2: (a, x, c) where a < x <= c
        # a in S, a < x.
        # c in S, c >= x.
        # Check if a | x + c.
        for a in S:
            if a >= x: continue
            for c in S:
                if c < x: continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: (a, b, x) where a < b <= x
        # a in S, a < x.
        # b in S, a < b <= x.
        # Check if a | b + x.
        for a in S:
            if a >= x: continue
            for b in S:
                if b <= a or b > x: continue
                if (b + x) % a == 0:
                    return False
        
        return True

    def greedy(order):
        S = []
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.append(x)
        return S

    best_set = []
    
    # 1. Descending order
    order_desc = list(range(n, 0, -1))
    S1 = greedy(order_desc)
    if len(S1) > len(best_set):
        best_set = S1
    
    # 2. Random orders
    # We can try to bias towards larger numbers?
    # Or just shuffle.
    while time.time() < deadline - 5.0: # Keep 5s margin
        order_rand = list(range(1, n + 1))
        random.shuffle(order_rand)
        # Maybe filter out 1?
        # If 1 is in the set, size is small.
        # We can try to exclude 1 in random tries.
        if random.random() < 0.5:
            order_rand.remove(1)
        
        S2 = greedy(order_rand)
        if len(S2) > len(best_set):
            best_set = S2
            
    return sorted(best_set)
