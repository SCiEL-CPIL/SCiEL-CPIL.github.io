import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 70  # safety margin

    # Start with the "safe" upper set
    K = 171
    S = set(range(K, n + 1))

    def can_add(x, S_set):
        """Check if x can be added to S_set."""
        # x must not be in S_set
        if x in S_set:
            return False
        
        # 1. x is smallest: x < b <= c
        # Check x !| (b+c) for all b, c in S_set with x < b <= c
        # Note: b, c are distinct elements in S_set.
        # But condition allows b=c? "Repeated larger summands b=c are included."
        # In a set, we pick elements. If b=c, we pick the same element twice?
        # "For every a<b<=c in the set".
        # If b=c, then b and c are the same value.
        # Since S is a set, we only have one copy of b.
        # Does "in the set" mean we can reuse elements?
        # Standard interpretation: a, b, c are elements of S.
        # If b=c, then b and c are the same element.
        # So we check a !| (b+b).
        
        # Let's assume b and c can be the same element value.
        # So we check pairs (b, c) from S_set such that x < b <= c.
        # If b and c are the same element, b=c.
        
        lst = sorted(S_set)
        m = len(lst)
        
        # Check x as smallest
        for i in range(m):
            b = lst[i]
            if b <= x:
                continue
            # b > x
            # Check b+c for c >= b
            # c can be b (if we allow reuse) or c > b
            # The condition is a < b <= c.
            # If we pick b and c from the set, and b=c, we use the same element.
            # So we check:
            # 1. c = b. Check x !| 2b.
            if (2 * b) % x == 0:
                return False
            # 2. c > b.
            for j in range(i + 1, m):
                c = lst[j]
                if (b + c) % x == 0:
                    return False
        
        # 2. x is middle: a < x <= c
        # Check a !| (x+c) for all a, c in S_set with a < x <= c
        for i in range(m):
            a = lst[i]
            if a >= x:
                break
            # a < x
            # c must be >= x
            # c can be x? No, x is not in S_set yet.
            # c is in S_set.
            # So c > x (since x not in S_set and a < x, c in S_set implies c != x).
            # Wait, if c = x, x must be in S_set. It is not.
            # So c > x.
            # Condition a < x <= c.
            # Since x not in S_set, c cannot be x.
            # So we check a < x < c.
            # Check a !| (x+c).
            for j in range(m):
                c = lst[j]
                if c < x:
                    continue
                # c > x
                if (x + c) % a == 0:
                    return False
        
        # 3. x is largest: a < b < x
        # Check a !| (b+x) for all a, b in S_set with a < b < x
        for i in range(m):
            a = lst[i]
            if a >= x:
                break
            # a < x
            for j in range(i + 1, m):
                b = lst[j]
                if b >= x:
                    break
                # a < b < x
                if (b + x) % a == 0:
                    return False

        return True

    # Greedy add smaller numbers
    for x in range(K - 1, 0, -1):
        if time.time() > deadline - 10:
            break
        if can_add(x, S):
            S.add(x)

    # Local Search: Restart from S-u
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_size = len(S)
        
        # Try removing each element u and greedily filling
        for u in list(S):
            if time.time() > deadline - 5:
                break
            
            S_try = S - {u}
            # Greedy add to S_try
            # Sort candidates? Smallest first? Or largest?
            # Smallest first might block more. Largest first is safer?
            # The greedy phase added smallest first.
            # Let's try adding in increasing order.
            for x in range(1, n + 1):
                if x in S_try:
                    continue
                if can_add(x, S_try):
                    S_try.add(x)
            
            if len(S_try) > current_size:
                S = S_try
                improved = True
                break

    return list(S)
