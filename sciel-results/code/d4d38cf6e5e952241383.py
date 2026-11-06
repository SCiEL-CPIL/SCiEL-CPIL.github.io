def construct():
    import time
    from math import gcd
    from itertools import combinations
    
    deadline = time.time() + 75  # safety margin under 90s
    
    def is_valid_triple(a, b, c):
        """Check if triple (a,b,c) violates the constraint (all three pairwise GCDs equal)."""
        return gcd(a, b) == gcd(a, c) == gcd(b, c)
    
    def can_add(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        for i, a in enumerate(selected):
            for b in selected[i+1:]:
                if gcd(a, b) == gcd(a, x) == gcd(b, x):
                    return False
        return True
    
    def count_bad_triples(selected):
        """Count the number of bad triples in selected."""
        count = 0
        n = len(selected)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_valid_triple(selected[i], selected[j], selected[k]):
                        count += 1
        return count
    
    # Strategy: greedy with multiple orderings, then local search
    
    def greedy(order):
        """Greedy: add elements in given order if valid."""
        selected = []
        for x in order:
            if can_add(x, selected):
                selected.append(x)
        return selected
    
    def local_search(selected, time_limit):
        """Improve by trying swaps: remove one, try to add a better one."""
        selected = list(selected)
        in_set = set(selected)
        all_nums = set(range(1, 129))
        
        def try_swap_remove(x, selected):
            """Try to remove x and add a different element."""
            remaining = [a for a in selected if a != x]
            # Try adding elements not in set
            candidates = [y for y in all_nums if y not in in_set and y != x]
            best_gain = 0
            best_pair = None
            for y in candidates:
                if can_add(y, remaining):
                    # Check if removing x and adding y is valid
                    new_set = remaining + [y]
                    # This is valid by construction (can_add ensures no new bad triples)
                    # But we need to check that removing x doesn't create issues (it won't)
                    best_pair = (x, y)
                    break  # found one valid swap
            return best_pair
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i, x in enumerate(selected):
                # Try removing x and adding something else
                remaining = [a for a in selected if a != x]
                for y in all_nums:
                    if y in in_set or y == x:
                        continue
                    if can_add(y, remaining):
                        # Check: does removing x help? Only if x was in a bad triple
                        # Actually, if we can add y, the new set has same size
                        # But maybe we can do better by removing x and adding y
                        # For now, just accept if it maintains validity
                        new_selected = remaining + [y]
                        in_set.remove(x)
                        in_set.add(y)
                        selected = new_selected
                        improved = True
                        break
                if improved:
                    break
            # Also try: if we can't swap, try to just add if possible (shouldn't happen in local search)
        return selected
    
    # Generate different orderings
    # Ordering 1: by number of distinct prime factors (descending) - more "diverse" GCDs
    def num_distinct_primes(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    # Ordering 2: by total number of prime factors (with multiplicity)
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    # Ordering 3: by value (original)
    order1 = sorted(range(1, 129), key=lambda x: (-num_distinct_primes(x), x))
    order2 = sorted(range(1, 129), key=lambda x: (-num_prime_factors(x), x))
    order3 = list(range(1, 129))
    order4 = sorted(range(1, 129), key=lambda x: (num_distinct_primes(x), x))  # ascending distinct primes
    order5 = sorted(range(1, 129), reverse=True)
    order6 = sorted(range(1, 129), key=lambda x: (x % 2, -num_prime_factors(x), x))  # evens first
    
    best_selected = []
    
    for order in [order1, order2, order3, order4, order5, order6]:
        if time.time() > deadline - 10:
            break
        selected = greedy(order)
        if len(selected) > len(best_selected):
            best_selected = selected
    
    # Local search on the best
    if best_selected and time.time() < deadline - 5:
        best_selected = local_search(best_selected, deadline - 2)
    
    # Try a few more random orderings if time permits
    import random
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        order = list(range(1, 129))
        random.shuffle(order)
        # Bias: put numbers with more prime factors earlier
        order.sort(key=lambda x: (-num_distinct_primes(x), random.random()))
        selected = greedy(order)
        if len(selected) > len(best_selected):
            best_selected = selected
    
    # Final local search
    if best_selected and time.time() < deadline - 2:
        best_selected = local_search(best_selected, deadline - 1)
    
    # Ensure nonempty
    if not best_selected:
        best_selected = [1]
    
    return best_selected
