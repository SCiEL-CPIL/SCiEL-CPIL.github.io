def construct():
    import time
    from math import gcd
    import random

    deadline = time.time() + 82.0

    # Precompute gcd table
    gcd_table = [[0] * 129 for _ in range(129)]
    for a in range(1, 129):
        for b in range(a, 129):
            g = gcd(a, b)
            gcd_table[a][b] = g
            gcd_table[b][a] = g

    def get_gcd(a, b):
        return gcd_table[a][b]

    def prime_factors(n):
        """Return set of distinct prime factors."""
        pf = set()
        d = 2
        while d * d <= n:
            if n % d == 0:
                pf.add(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            pf.add(n)
        return pf

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

    all_numbers = list(range(1, 129))

    # Precompute prime factor sets for all numbers
    pf_sets = {x: prime_factors(x) for x in all_numbers}

    def can_add(selected, x):
        """Return True if adding x to selected keeps the no-equal-gcd-triple property."""
        for i in range(len(selected)):
            a = selected[i]
            ga = get_gcd(a, x)
            for j in range(i + 1, len(selected)):
                b = selected[j]
                gb = get_gcd(b, x)
                gab = get_gcd(a, b)
                if ga == gb == gab:
                    return False
        return True

    def greedy(order):
        selected = []
        for x in order:
            if can_add(selected, x):
                selected.append(x)
        return selected

    def is_valid_set(selected):
        n = len(selected)
        for i in range(n):
            a = selected[i]
            for j in range(i + 1, n):
                b = selected[j]
                for k in range(j + 1, n):
                    c = selected[k]
                    if get_gcd(a, b) == get_gcd(a, c) == get_gcd(b, c):
                        return False
        return True

    # Strategy: prioritize numbers with many distinct prime factors
    # These create the most constraints, so placing them first limits conflicts
    # Numbers with few prime factors (primes, prime powers) are more flexible

    def distinct_pf_count(x):
        return len(pf_sets[x])

    # Ordering 1: by decreasing distinct prime factors, then total pf
    order1 = sorted(all_numbers, key=lambda x: (-distinct_pf_count(x), -total_pf(x), -num_divisors(x), x))
    best = greedy(order1)

    # Ordering 2: by decreasing total prime factors
    order2 = sorted(all_numbers, key=lambda x: (-total_pf(x), -distinct_pf_count(x), x))
    r2 = greedy(order2)
    if len(r2) > len(best):
        best = r2

    # Ordering 3: by decreasing number of divisors
    order3 = sorted(all_numbers, key=lambda x: (-num_divisors(x), x))
    r3 = greedy(order3)
    if len(r3) > len(best):
        best = r3

    # Ordering 4: primes and prime powers first (most flexible), then composites
    def flexibility_score(x):
        # Higher score = more flexible = should come later
        # Primes are most flexible (only one prime factor)
        # Numbers with many distinct primes are least flexible
        return (distinct_pf_count(x), total_pf(x), -x)

    order4 = sorted(all_numbers, key=flexibility_score, reverse=True)
    r4 = greedy(order4)
    if len(r4) > len(best):
        best = r4

    # Natural and reversed
    r5 = greedy(all_numbers)
    if len(r5) > len(best):
        best = r5
    r6 = greedy(list(reversed(all_numbers)))
    if len(r6) > len(best):
        best = r6

    # Random orderings with different seeds
    random.seed(42)
    for seed in range(40):
        if time.time() > deadline - 15:
            break
        random.seed(seed)
        order = all_numbers[:]
        random.shuffle(order)
        r = greedy(order)
        if len(r) > len(best):
            best = r

    # Weighted random: bias toward numbers with more prime factors
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        random.seed(1000 + seed)
        order = all_numbers[:]
        # Bias: numbers with more distinct PFs get higher weight
        weights = [1.0 / (1 + distinct_pf_count(x)) for x in order]
        # Invert so high-PF numbers are more likely to come first
        weights = [1.0 / w for w in weights]
        # Shuffle with bias
        for i in range(len(order) - 1, 0, -1):
            j = random.randint(0, i)
            order[i], order[j] = order[j], order[i]
        r = greedy(order)
        if len(r) > len(best):
            best = r

    def local_search(selected, time_budget):
        """Improve via add, remove-1-add-1, remove-1-add-2."""
        selected = list(selected)
        selected_set = set(selected)
        unselected = [x for x in all_numbers if x not in selected_set]

        # Pass A: add any element that fits
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for x in unselected[:]:
                if can_add(selected, x):
                    selected.append(x)
                    selected_set.add(x)
                    unselected.remove(x)
                    improved = True

        # Pass B: remove-1-add-2 (net +1)
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for y in selected[:]:
                if time.time() > time_budget:
                    break
                temp = [s for s in selected if s != y]
                for i in range(len(unselected)):
                    x1 = unselected[i]
                    if not can_add(temp, x1):
                        continue
                    new_sel = temp + [x1]
                    for j in range(len(unselected)):
                        if i == j:
                            continue
                        x2 = unselected[j]
                        if can_add(new_sel, x2):
                            selected = new_sel + [x2]
                            selected_set = set(selected)
                            unselected = [u for u in all_numbers if u not in selected_set]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        # Pass C: remove-2-add-3 (net +1) - more aggressive
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            n_sel = len(selected)
            if n_sel < 4:
                break
            for i in range(n_sel):
                if time.time() > time_budget:
                    break
                y1 = selected[i]
                for j in range(i + 1, n_sel):
                    if time.time() > time_budget:
                        break
                    y2 = selected[j]
                    temp = [s for s in selected if s != y1 and s != y2]
                    # Try to add 3 numbers
                    added = []
                    for x in unselected:
                        if time.time() > time_budget:
                            break
                        if x in added:
                            continue
                        if can_add(temp + added, x):
                            added.append(x)
                            if len(added) >= 3:
                                break
                    if len(added) >= 3:
                        selected = temp + added[:3]
                        selected_set = set(selected)
                        unselected = [u for u in all_numbers if u not in selected_set]
                        improved = True
                        break
                if improved:
                    break

        return selected

    # Local search on best
    best = local_search(best, deadline - 5)

    # Local search on other good results
    for r in (r2, r3, r4, r5, r6):
        if time.time() > deadline - 8:
            break
        cand = local_search(r, deadline - 5)
        if len(cand) > len(best):
            best = cand

    # Final fill
    selected_set = set(best)
    for x in all_numbers:
        if x not in selected_set and can_add(best, x):
            best.append(x)
            selected_set.add(x)

    if not best:
        best = [1]

    return best
