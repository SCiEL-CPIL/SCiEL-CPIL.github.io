"""Sum-free subset solver using parity-seeded hill climbing with efficient conflict tracking."""
import time
import random

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(s):
        sv_set = set(s)
        for a in s:
            for b in s:
                if a + b in sv_set:
                    return False
        return True

    def solve_one(values, seed):
        n = len(values)
        val_set = set(values)
        
        # Start with all odd numbers (guaranteed sum-free: odd+odd=even, odd+even=odd but need to check)
        # Actually odd+odd=even, so if all elements are odd, a+b is even, which is not in the set of odds.
        # Wait: odd + odd = even. If our set contains only odds, then a+b is even, and even is not in the set.
        # So all-odd is sum-free. But we might be able to add some evens.
        
        odds = [x for x in values if x % 2 != 0]
        evens = [x for x in values if x % 2 == 0]
        
        # Start with all odds
        current = set(odds)
        
        # For each even, check if it can be added
        # An even e can be added if:
        # 1. For all a in current: a + e not in current (i.e., a+e is odd, and we need a+e not in odds)
        # 2. 2*e not in current (2*e is even, not in odds, so this is automatically satisfied)
        # 3. For all a in current: a + e != a (trivially true since e != 0)
        # Actually we need: no a, b in current such that a+b = e, and no a in current such that a+e in current.
        # Since current is all odds:
        #   - a+b = e: a,b odd, a+b even. So we need to check if any two odds sum to e.
        #   - a+e in current: a odd, e even, a+e odd. So we need a+e not in odds.
        
        # Let's build a set of all sums of pairs of odds for quick lookup
        odds_set = set(odds)
        # Precompute: for each even e, can it be added?
        # e can be added iff:
        #   (a) No two odds a, b in odds_set have a+b = e
        #   (b) For no odd a in odds_set is a+e in odds_set
        
        # (a) is equivalent to: e is not in {a+b : a,b in odds}
        # (b) is equivalent to: (odds_set + e) ∩ odds_set = empty, i.e., no odd a has a+e also odd and in set
        
        # Let's precompute the set of all pair sums of odds
        # For n odds, this is O(n^2) which for n=300 (half of 600... wait n is at most 240)
        # Max n = 240, so odds up to 240, pair sums up to 240*240 = 57600 checks. Fine.
        
        pair_sums = set()
        for i in range(len(odds)):
            a = odds[i]
            for j in range(i, len(odds)):
                b = odds[j]
                pair_sums.add(a + b)
        
        # Now try to add each even
        for e in evens:
            if time.time() > deadline:
                break
            # Check condition (a): e not in pair_sums
            if e in pair_sums:
                continue
            # Check condition (b): no odd a has a+e in odds_set
            ok = True
            for a in odds:
                if (a + e) in odds_set:
                    ok = False
                    break
            if ok:
                current.add(e)
                # Update pair_sums: new pairs involving e
                # e is even, a is odd: e+a is odd, not in pair_sums (which are even)
                # e+e = 2e is even, add to pair_sums
                pair_sums.add(2 * e)
                # Also e + odd = odd, not in pair_sums
                # But we need to update: for future evens, the pair sums now include e+odd? No, pair_sums is for checking if a new even equals a sum of two existing elements.
                # Actually pair_sums should be the set of all a+b where a,b in current.
                # Let me reconsider. We need to maintain the full set of pair sums of current.
        
        # Let me redo this more carefully with a proper conflict tracking approach.
        
        # Reset and use a cleaner approach
        current = set(odds)
        
        # Precompute for each element x in values, which pairs (a,b) in current would conflict if x were added
        # Actually, let's use a simpler hill-climbing approach:
        
        # Phase 1: Start with all odds, greedily add evens if they don't create conflicts
        current = set(odds)
        odds_set = set(odds)
        
        # For efficient checking, maintain:
        # - current set
        # - For each candidate even e, track if it's "safe" to add
        
        def can_add(e, cur_set):
            """Check if e can be added to cur_set without violating sum-free property."""
            # Need: no a, b in cur_set with a + b = e
            # And: no a in cur_set with a + e in cur_set
            # And: 2*e not in cur_set (implied by a+b=e with a=b=e, but e not in cur_set yet)
            # Actually a+b=e: both a,b in cur_set
            # a+e in cur_set: a in cur_set, a+e in cur_set
            
            # Check a+b=e
            for a in cur_set:
                if (e - a) in cur_set:
                    # Found a and e-a both in cur_set, their sum is e
                    return False
            # Check a+e in cur_set
            for a in cur_set:
                if (a + e) in cur_set:
                    return False
            return True
        
        # Try adding each even
        for e in evens:
            if time.time() > deadline:
                break
            if can_add(e, current):
                current.add(e)
        
        # Phase 2: Hill climbing - try to swap elements to improve
        best_size = len(current)
        best_set = set(current)
        
        # Try removing an element and adding one or two others
        for _ in range(50):  # iterations
            if time.time() > deadline:
                break
            improved = False
            for v in list(current):
                if time.time() > deadline:
                    break
                test_set = current - {v}
                # Try adding each unselected element
                for w in values:
                    if w in test_set:
                        continue
                    if can_add(w, test_set):
                        test_set.add(w)
                        if len(test_set) > best_size:
                            current = set(test_set)
                            best_size = len(current)
                            best_set = set(current)
                            improved = True
                        break
            if not improved:
                # Try 2-for-1 swaps: remove one, add two
                for v in list(current):
                    if time.time() > deadline:
                        break
                    test_set = current - {v}
                    added = 0
                    for w in values:
                        if w in test_set or added >= 2:
                            continue
                        if can_add(w, test_set):
                            test_set.add(w)
                            added += 1
                    if len(test_set) > best_size:
                        current = set(test_set)
                        best_size = len(current)
                        best_set = set(current)
                        improved = True
            if not improved:
                break
        
        # Phase 3: Randomized restarts with different starting points
        rng = random.Random(seed * 7919 + 1)
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Random subset of odds + greedy evens
            if restart % 3 == 0:
                # Start with all odds
                init = set(odds)
            elif restart % 3 == 1:
                # Random subset of odds
                k = max(1, len(odds) // 2)
                init = set(rng.sample(odds, k))
            else:
                # Random mix
                init = set(rng.sample(values, n // 2))
            
            # Greedy add
            for v in values:
                if v in init:
                    continue
                if can_add(v, init):
                    init.add(v)
            
            # Local search
            for _ in range(20):
                if time.time() > deadline:
                    break
                improved = False
                for v in list(init):
                    if time.time() > deadline:
                        break
                    test_set = init - {v}
                    for w in values:
                        if w in test_set:
                            continue
                        if can_add(w, test_set):
                            test_set.add(w)
                            if len(test_set) > len(init):
                                init = set(test_set)
                                improved = True
                            break
                if not improved:
                    break
            
            if len(init) > best_size:
                best_size = len(init)
                best_set = set(init)
        
        return sorted(best_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        vals = instance(size, seed)
        results.append(solve_one(vals, seed))

    # Final validation
    for i, subset in enumerate(results):
        sv_set = set(subset)
        valid = True
        for a in subset:
            for b in subset:
                if a + b in sv_set:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            size, seed = ((60, 801), (120, 802), (240, 803))[i]
            vals = instance(size, seed)
            results[i] = [x for x in vals if x % 2]

    return results
