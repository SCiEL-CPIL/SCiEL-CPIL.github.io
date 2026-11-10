import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55  # Safety margin before 60s
    
    # Precompute all forbidden pairs for each 'a'
    # 1/a = 1/b + 1/c => (b-a)(c-a) = a^2
    # b > a, c > a
    pairs_for_a = {a: [] for a in range(1, 181)}
    for a in range(1, 181):
        a2 = a * a
        # d is a divisor of a^2
        # b = a + d, c = a + a^2/d
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                if 1 <= b <= 180 and 1 <= c <= 180 and b != c and b != a and c != a:
                    pairs_for_a[a].append((b, c))
                # Note: d and e are distinct unless d=e=a, in which case b=c=2a, which is excluded by b!=c
                # Also check if d != e to avoid duplicates? 
                # If d < e, we get one pair. If d > e, we get the same pair swapped.
                # Since we iterate d up to sqrt, we get d <= e.
                # If d == e, b == c, which is excluded.
                # So no duplicates in the list for a given a.

    # Helper: Check if x can be added to set S without creating a forbidden triple
    # x can be 'a', or 'b', or 'c'
    def can_add(x, S):
        # Case 1: x is 'a'
        # Need to check if there exist b, c in S such that (b,c) is in pairs_for_a[x]
        # pairs_for_a[x] is a list of (b,c).
        # We can check if b in S and c in S.
        for b, c in pairs_for_a[x]:
            if b in S and c in S:
                return False
        
        # Case 2: x is 'b' or 'c'
        # There exists a < x and z in S such that 1/a = 1/x + 1/z
        # => z = a*x / (x-a)
        # We iterate over all a in S such that a < x
        for a in S:
            if a >= x:
                continue
            den = x - a
            num = a * x
            if num % den == 0:
                z = num // den
                if z in S and z != x and z != a:
                    return False
        return True

    # Greedy construction
    def greedy(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    # Local Search (Addition only)
    def local_search(S, max_steps=200):
        S = set(S)
        for _ in range(max_steps):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try to add elements
            added = False
            # Try a few random candidates from [1, 180] not in S
            candidates = [i for i in range(1, 181) if i not in S]
            if candidates:
                random.shuffle(candidates)
                for y in candidates[:20]: # Check up to 20 random candidates
                    if can_add(y, S):
                        S.add(y)
                        added = True
                        break
            
            if not added:
                break
                
        return S

    best_set = set()
    best_size = 0
    
    random.seed(42)
    
    # Try different orders
    orders = []
    orders.append(list(range(1, 181)))
    orders.append(list(range(180, 0, -1)))
    
    # Sort by number of pairs (as 'a')
    degree_order = sorted(range(1, 181), key=lambda x: len(pairs_for_a[x]))
    orders.append(degree_order)
    orders.append(list(reversed(degree_order)))
    
    # Add some random orders
    for _ in range(10):
        r_order = list(range(1, 181))
        random.shuffle(r_order)
        orders.append(r_order)
        
    for order in orders:
        if time.time() - start_time > DEADLINE:
            break
            
        S = greedy(order)
        
        # Local search to try to expand
        S_expanded = local_search(S, max_steps=200)
        
        if len(S_expanded) > best_size:
            best_size = len(S_expanded)
            best_set = S_expanded
            
    # Final check and return
    return list(best_set)
