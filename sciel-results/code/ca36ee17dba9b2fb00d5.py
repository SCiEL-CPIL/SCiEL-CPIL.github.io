def construct():
    from math import gcd

    # Precompute all pairs (a, b) with a < b in [1, 128]
    n = 128
    elements = list(range(1, n + 1))
    
    # For each pair (a, b), compute lcm(a, b)
    # lcm(a, b) = a * b // gcd(a, b)
    
    # We need to find a subset S such that no three distinct elements a, b, c in S
    # have lcm(a,b) == lcm(a,c) == lcm(b,c)
    
    # Key insight: For three distinct numbers a, b, c, let L = lcm(a,b) = lcm(a,c) = lcm(b,c).
    # This means L is a multiple of all three, and for any prime p, the max of the p-adic
    # valuations of any two of {a,b,c} equals the p-adic valuation of L.
    # This implies that for each prime p, at least two of {v_p(a), v_p(b), v_p(c)} must be equal
    # to v_p(L), and the third must be <= v_p(L). Actually, more precisely:
    # lcm(a,b) = L means max(v_p(a), v_p(b)) = v_p(L) for all p.
    # So for each p, max(v_p(a), v_p(b)) = max(v_p(a), v_p(c)) = max(v_p(b), v_p(c)) = v_p(L).
    # This means: for each p, at least two of the three valuations must equal v_p(L),
    # and the remaining one is <= v_p(L).
    # Equivalently, the two largest valuations among the three are equal.
    
    # Approach: Build a graph where vertices are numbers 1..128.
    # We want to find a large subset with no "bad triple".
    # This is a 3-uniform hypergraph independent set problem, which is NP-hard in general.
    # But for n=128, we can try a greedy approach with a good ordering, or use local search.
    
    # Let's use a greedy approach: sort elements and add them one by one if they don't
    # form a bad triple with any two already selected elements.
    
    # Better: use a heuristic that tries to maximize the set.
    # We'll try multiple orderings and pick the best.
    
    def is_bad_triple(a, b, c):
        """Check if three distinct elements form a bad triple."""
        la = a * b // gcd(a, b)
        lc = a * c // gcd(a, c)
        lb = b * c // gcd(b, c)
        return la == lc == lb
    
    def greedy_with_order(order):
        """Greedy: process elements in given order, add if no bad triple formed."""
        selected = []
        for x in order:
            ok = True
            # Check all pairs in selected
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    if is_bad_triple(selected[i], selected[j], x):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.append(x)
        return selected
    
    # Try several different orderings
    # 1. Natural order
    # 2. Reverse order
    # 3. Order by number of prime factors (fewer first)
    # 4. Order by value (ascending)
    # 5. Random orders
    
    import random
    
    best_set = []
    
    # Natural order
    s1 = greedy_with_order(elements)
    if len(s1) > len(best_set):
        best_set = s1
    
    # Reverse order
    s2 = greedy_with_order(list(reversed(elements)))
    if len(s2) > len(best_set):
        best_set = s2
    
    # Order by number of prime factors (omega function)
    def omega(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    by_omega = sorted(elements, key=lambda x: (omega(x), x))
    s3 = greedy_with_order(by_omega)
    if len(s3) > len(best_set):
        best_set = s3
    
    by_omega_rev = sorted(elements, key=lambda x: (-omega(x), x))
    s4 = greedy_with_order(by_omega_rev)
    if len(s4) > len(best_set):
        best_set = s4
    
    # Try some random orderings
    rng = random.Random(42)
    for _ in range(50):
        order = elements[:]
        rng.shuffle(order)
        s = greedy_with_order(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Also try: all odd numbers
    odds = [x for x in elements if x % 2 == 1]
    # Verify odds don't have bad triples (should be safe)
    valid_odds = True
    for i in range(len(odds)):
        for j in range(i + 1, len(odds)):
            for k in range(j + 1, len(odds)):
                if is_bad_triple(odds[i], odds[j], odds[k]):
                    valid_odds = False
                    break
            if not valid_odds:
                break
        if not valid_odds:
            break
    
    if valid_odds and len(odds) > len(best_set):
        best_set = odds
    
    # Try: numbers with no factor of 2 (same as odds) - already done
    # Try: numbers that are powers of distinct primes? No, that's small.
    
    # Try: all numbers whose prime factorization has max exponent 1 (square-free)
    # Actually, let's try a few more structured sets
    
    # Try: numbers in [1, 128] that are not divisible by 2 or 3
    no_2_3 = [x for x in elements if x % 2 != 0 and x % 3 != 0]
    valid_no23 = True
    for i in range(len(no_2_3)):
        for j in range(i + 1, len(no_2_3)):
            for k in range(j + 1, len(no_2_3)):
                if is_bad_triple(no_2_3[i], no_2_3[j], no_2_3[k]):
                    valid_no23 = False
                    break
            if not valid_no23:
                break
        if not valid_no23:
            break
    
    if valid_no23 and len(no_2_3) > len(best_set):
        best_set = no_2_3
    
    if not best_set:
        best_set = [1]
    
    return best_set
