def construct():
    import time
    from math import gcd
    
    deadline = time.time() + 85
    
    # Precompute gcd table
    gcd_table = [[0] * 129 for _ in range(129)]
    for a in range(1, 129):
        for b in range(a, 129):
            g = gcd(a, b)
            gcd_table[a][b] = g
            gcd_table[b][a] = g
    
    all_numbers = list(range(1, 129))
    
    def num_distinct_pf(n):
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
    
    def total_pf(n):
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
    
    def num_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count
    
    def is_valid_triple(a, b, c):
        """Check if triple (a,b,c) violates the constraint (all three pairwise gcds equal)."""
        return gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]
    
    def can_add(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        for i in range(len(selected)):
            a = selected[i]
            ga = gcd_table[a][x]
            for j in range(i + 1, len(selected)):
                b = selected[j]
                gb = gcd_table[b][x]
                gab = gcd_table[a][b]
                if ga == gb == gab:
                    return False
        return True
    
    def greedy(order):
        selected = []
        for x in order:
            if can_add(x, selected):
                selected.append(x)
        return selected
    
    def backtracking(order, current, remaining, best):
        """Backtracking search to maximize set size."""
        if time.time() > deadline - 2:
            return
        
        if len(current) + len(remaining) <= len(best):
            return
        
        if not remaining:
            if len(current) > len(best[0]):
                best[0] = list(current)
            return
        
        # Try to add the first element in remaining
        x = remaining[0]
        rest = remaining[1:]
        
        # Option 1: Include x
        if can_add(x, current):
            current.append(x)
            backtracking(order, current, rest, best)
            current.pop()
        
        # Option 2: Exclude x
        backtracking(order, current, rest, best)
    
    def local_search(selected):
        """Improve a valid set via local search (add, swap, remove+add)."""
        selected = list(selected)
        selected_set = set(selected)
        unselected = [x for x in all_numbers if x not in selected_set]
        improved = True
        
        while improved and time.time() < deadline - 3:
            improved = False
            
            # Try adding each unselected number
            for x in unselected:
                if time.time() > deadline - 3:
                    break
                if can_add(x, selected):
                    selected.append(x)
                    selected_set.add(x)
                    unselected = [u for u in unselected if u != x]
                    improved = True
            
            # Try swaps: remove y, add x (net 0, but might enable further additions)
            for x in unselected:
                if time.time() > deadline - 3:
                    break
                for y in selected:
                    new_sel = [s for s in selected if s != y]
                    if can_add(x, new_sel):
                        # Check if this swap helps by trying to add more
                        new_sel.append(x)
                        new_set = set(new_sel)
                        new_unsel = [u for u in all_numbers if u not in new_set]
                        # Try to add more from new_unsel
                        extra = 0
                        for z in new_unsel:
                            if can_add(z, new_sel):
                                new_sel.append(z)
                                extra += 1
                        if extra > 0:
                            selected = new_sel
                            selected_set = set(selected)
                            unselected = [u for u in all_numbers if u not in selected_set]
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            
            # Try remove 1, add 2 (net +1)
            for y in selected:
                if time.time() > deadline - 3:
                    break
                temp_sel = [s for s in selected if s != y]
                temp_set = set(temp_sel)
                # Find two numbers that can both be added
                found = False
                for i in range(len(unselected)):
                    x1 = unselected[i]
                    if not can_add(x1, temp_sel):
                        continue
                    new_sel = temp_sel + [x1]
                    for j in range(len(unselected)):
                        if i == j:
                            continue
                        x2 = unselected[j]
                        if can_add(x2, new_sel):
                            selected = new_sel + [x2]
                            selected_set = set(selected)
                            unselected = [u for u in all_numbers if u not in selected_set]
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        return selected
    
    # Multiple orderings for greedy
    best = []
    
    # Ordering 1: by decreasing distinct prime factors
    order1 = sorted(all_numbers, key=lambda x: (-num_distinct_pf(x), -total_pf(x), -num_divisors(x), x))
    best = greedy(order1)
    
    # Ordering 2: by decreasing total prime factors
    order2 = sorted(all_numbers, key=lambda x: (-total_pf(x), -num_distinct_pf(x), x))
    r2 = greedy(order2)
    if len(r2) > len(best):
        best = r2
    
    # Ordering 3: by decreasing number of divisors
    order3 = sorted(all_numbers, key=lambda x: (-num_divisors(x), x))
    r3 = greedy(order3)
    if len(r3) > len(best):
        best = r3
    
    # Ordering 4: by value (ascending) - coprimality bias
    order4 = sorted(all_numbers, key=lambda x: x)
    r4 = greedy(order4)
    if len(r4) > len(best):
        best = r4
    
    # Ordering 5: by value (descending)
    order5 = sorted(all_numbers, key=lambda x: -x)
    r5 = greedy(order5)
    if len(r5) > len(best):
        best = r5
    
    # Ordering 6: primes first, then composites
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True
    
    order6 = sorted(all_numbers, key=lambda x: (0 if is_prime(x) else 1, -num_distinct_pf(x), x))
    r6 = greedy(order6)
    if len(r6) > len(best):
        best = r6
    
    # Random orderings
    import random
    random.seed(42)
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        order = all_numbers[:]
        random.shuffle(order)
        r = greedy(order)
        if len(r) > len(best):
            best = r
    
    # Local search on best
    best = local_search(best)
    
    # Backtracking from a smaller initial set to find larger solutions
    # Use backtracking with the best ordering
    if time.time() < deadline - 10:
        # Try backtracking with a limited depth
        for order in [order1, order2, order3]:
            if time.time() > deadline - 5:
                break
            # Start with empty set and use backtracking
            best_bt = [best]
            # Use a modified backtracking that doesn't go too deep
            def bt_search(current, remaining, best_ref):
                if time.time() > deadline - 3:
                    return
                if len(current) + len(remaining) <= len(best_ref[0]):
                    return
                if not remaining:
                    if len(current) > len(best_ref[0]):
                        best_ref[0] = list(current)
                    return
                
                x = remaining[0]
                rest = remaining[1:]
                
                # Try include x
                if can_add(x, current):
                    current.append(x)
                    bt_search(current, rest, best_ref)
                    current.pop()
                
                # Try exclude x
                bt_search(current, rest, best_ref)
            
            bt_search([], order, best_bt)
            if len(best_bt[0]) > len(best):
                best = best_bt[0]
                best = local_search(best)
    
    # More random restarts with local search
    random.seed(123)
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        order = all_numbers[:]
        random.shuffle(order)
        r = greedy(order)
        r = local_search(r)
        if len(r) > len(best):
            best = r
    
    return best
