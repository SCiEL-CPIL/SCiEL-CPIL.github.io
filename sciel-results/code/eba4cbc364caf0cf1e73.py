def construct():
    import time
    import random
    from math import gcd
    
    deadline = time.time() + 50  # safety margin before 60s kill
    
    n = 128
    
    # Precompute gcd table
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # For each element x, and each pair (a,b) with a < b, determine if {a,b,x} is a bad triple.
    # A triple (a,b,c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c) > 1 (actually even if =1? 
    # No, the problem says "three pairwise greatest common divisors are equal". 
    # If gcd is 1, that's fine? Wait, "avoiding equal-gcd triples". 
    # If gcd(a,b)=gcd(a,c)=gcd(b,c)=1, they are pairwise coprime. Is that a "bad" triple?
    # The task says: "no three distinct elements whose three pairwise greatest common divisors are equal."
    # So yes, if all three pairwise gcds are 1, that IS a bad triple (equal gcds).
    # So we must avoid any three elements that are pairwise coprime too!
    # This changes everything. The set must not contain 3 pairwise coprime elements, AND
    # must not contain 3 elements with common gcd d > 1.
    
    # Actually, let's re-read: "no three distinct elements whose three pairwise greatest common divisors are equal."
    # This includes the case where the common gcd is 1.
    
    # So the constraint is: for any distinct a, b, c in S, it is NOT the case that
    # gcd(a,b) == gcd(a,c) == gcd(b,c).
    
    # Let's precompute for each x, the set of pairs (a,b) in S such that adding x creates a bad triple.
    # For a fixed x, a pair (a,b) conflicts with x if gcd(a,b) == gcd(a,x) == gcd(b,x).
    
    # Precompute conflict masks:
    # conflict[x] = bitmask of pairs? No, pairs are too many.
    # Instead: for each x and each a, conflict[x][a] = bitmask of b such that gcd(a,b)==gcd(a,x)==gcd(b,x)
    
    conflict = [[0]*(n+1) for _ in range(n+1)]
    for x in range(1, n+1):
        for a in range(1, n+1):
            if a == x:
                continue
            g_ax = gcd_table[a][x]
            mask = 0
            for b in range(1, n+1):
                if b == a or b == x:
                    continue
                if gcd_table[a][b] == g_ax and gcd_table[b][x] == g_ax:
                    mask |= (1 << b)
            conflict[x][a] = mask
    
    def can_add(x, S):
        """Check if x can be added to set S (bitmask)."""
        temp = S
        while temp:
            lb = temp & (-temp)
            a = lb.bit_length() - 1
            if conflict[x][a] & S:
                return False
            temp ^= lb
        return True
    
    def greedy_build(order):
        S = 0
        for x in order:
            if can_add(x, S):
                S |= (1 << x)
        return S
    
    def local_search(S, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add any element
            for x in range(1, n+1):
                if (S >> x) & 1:
                    continue
                if can_add(x, S):
                    S |= (1 << x)
                    improved = True
                    break
            if improved:
                continue
            # Try 1-for-1 swap
            for a in range(1, n+1):
                if not ((S >> a) & 1):
                    continue
                S2 = S & ~(1 << a)
                for x in range(1, n+1):
                    if (S2 >> x) & 1:
                        continue
                    if can_add(x, S2):
                        S = S2 | (1 << x)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Try 1-for-2 swap
            for a in range(1, n+1):
                if not ((S >> a) & 1):
                    continue
                S2 = S & ~(1 << a)
                added = 0
                for x in range(1, n+1):
                    if (S2 >> x) & 1:
                        continue
                    if can_add(x, S2):
                        S2 |= (1 << x)
                        added += 1
                        if added == 2:
                            break
                if added >= 2:
                    S = S2
                    improved = True
                    break
            if improved:
                continue
            # Try 2-for-1 swap (remove two, add one) - this can only decrease size, skip
            # Try 2-for-2 swap
            if time.time() < time_limit - 1:
                elements = [i for i in range(1, n+1) if (S >> i) & 1]
                random.shuffle(elements)
                found = False
                for i in range(len(elements)):
                    if time.time() > time_limit - 1:
                        break
                    for j in range(i+1, len(elements)):
                        if time.time() > time_limit - 1:
                            break
                        a, b = elements[i], elements[j]
                        S2 = S & ~(1 << a) & ~(1 << b)
                        added = 0
                        for x in range(1, n+1):
                            if (S2 >> x) & 1:
                                continue
                            if can_add(x, S2):
                                S2 |= (1 << x)
                                added += 1
                                if added == 2:
                                    break
                        if added >= 2:
                            S = S2
                            improved = True
                            found = True
                            break
                    if found:
                        break
        return S
    
    # Strategy: Use the structural insight.
    # A bad triple has all pairwise gcds equal to some d.
    # If d=1: three pairwise coprime elements.
    # If d>1: three elements all divisible by d, and after dividing by d, pairwise coprime.
    
    # Key insight: To maximize size, we should include many numbers.
    # Numbers with many prime factors create more conflicts.
    # Numbers that are prime or prime powers might be safer?
    
    # Let's try a different greedy: sort by number of divisors or by "conflict potential".
    # Elements that conflict with fewer others should be added first.
    
    # Compute conflict degree for each x: how many pairs (a,b) conflict with x.
    # Actually, simpler: for each x, count how many a have conflict[x][a] != 0.
    # Or better: count the total number of b across all a that conflict with x.
    
    conflict_degree = [0]*(n+1)
    for x in range(1, n+1):
        total = 0
        for a in range(1, n+1):
            if a == x:
                continue
            total += conflict[x][a].bit_count()
        conflict_degree[x] = total
    
    # Greedy by low conflict degree
    def greedy_by_degree():
        order = list(range(1, n+1))
        order.sort(key=lambda x: conflict_degree[x])
        return greedy_build(order)
    
    # Also try: prioritize numbers with few prime factors
    def num_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    def greedy_by_pf():
        order = list(range(1, n+1))
        order.sort(key=lambda x: num_prime_factors(x))
        return greedy_build(order)
    
    best_S = 0
    best_size = 0
    
    # First, try structured greedies
    S1 = greedy_by_degree()
    if S1.bit_count() > best_size:
        best_S = S1
        best_size = S1.bit_count()
    
    S2 = greedy_by_pf()
    if S2.bit_count() > best_size:
        best_S = S2
        best_size = S2.bit_count()
    
    # Random restarts
    random.seed(42)
    for trial in range(300):
        if time.time() > deadline - 10:
            break
        order = list(range(1, n+1))
        random.shuffle(order)
        S = greedy_build(order)
        if S.bit_count() > best_size:
            best_S = S
            best_size = S.bit_count()
    
    # Local search on best
    if best_S and time.time() < deadline - 5:
        best_S = local_search(best_S, deadline - 2)
    
    # Additional: try to improve by adding elements in a specific order
    if time.time() < deadline - 2:
        S = best_S
        # Try adding in order of increasing conflict degree
        for x in range(1, n+1):
            if (S >> x) & 1:
                continue
            if can_add(x, S):
                S |= (1 << x)
        if S.bit_count() > best_S.bit_count():
            best_S = S
    
    result = [i for i in range(1, n+1) if (best_S >> i) & 1]
    
    if not result:
        result = [1]
    
    return result
