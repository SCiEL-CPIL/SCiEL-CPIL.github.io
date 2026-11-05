"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def can_add_fast(S_set, x):
        """Fast check: can x be added to sum-free set S?
        Violation if exists a in S with a+x in S, or x-a in S, or 2x in S.
        """
        # Check 2x
        if (2 * x) in S_set:
            return False
        # Check a + x in S for all a in S
        for a in S_set:
            if (a + x) in S_set:
                return False
        # Check x - a in S for all a in S
        for a in S_set:
            if (x - a) in S_set:
                return False
        return True

    def find_violations(S_set):
        """Find all triples (a,b,c) with a+b=c in S. Returns list of (a,b,c)."""
        violations = []
        S_list = list(S_set)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):  # a <= b to avoid duplicates, but allow a=b
                b = S_list[j]
                c = a + b
                if c in S_set and c >= 0:  # c must be in set
                    violations.append((a, b, c))
        return violations

    def greedy_expand_fast(S_set, V_set, order=None):
        """Add elements from V to S in the given order using fast check."""
        if order is None:
            order = sorted(V_set - S_set, key=lambda x: abs(x), reverse=True)
        for x in order:
            if x not in S_set and can_add_fast(S_set, x):
                S_set.add(x)
        return S_set

    def is_sum_free(S_set):
        """Check if S_set is sum-free."""
        S_list = list(S_set)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                b = S_list[j]
                c = a + b
                if c in S_set:
                    return False
        return True

    def sanitize(S_set):
        """Remove elements until S_set is sum-free."""
        # We need to remove elements involved in violations.
        # Simple greedy removal: find a violation, remove one of the elements, repeat.
        # To be fast, we can remove the element that is part of the most violations,
        # but for safety and simplicity, we just pick any element from a violation.
        # Since sets are small (<300), this is fast.
        while not is_sum_free(S_set):
            # Find one violation
            S_list = list(S_set)
            found = False
            for i in range(len(S_list)):
                a = S_list[i]
                for j in range(i, len(S_list)):
                    b = S_list[j]
                    c = a + b
                    if c in S_set:
                        # Remove c (or a or b). Removing c is often safe if c is large,
                        # but let's just remove a to break the cycle.
                        # Actually, removing any one breaks this specific triple.
                        S_set.discard(a)
                        found = True
                        break
                if found:
                    break
        return S_set

    def repair_and_expand(S_list, V_set):
        """Remove elements to fix violations, then greedily expand."""
        S_set = set(S_list)
        
        # Phase 1: Iteratively remove elements causing most violations
        # Use sanitize to ensure we definitely get a sum-free set
        S_set = sanitize(S_set)
        
        # Phase 2: Greedy expand
        S_set = greedy_expand_fast(S_set, V_set)
        # Ensure final result is sum-free (greedy_expand should preserve it, but just in case)
        S_set = sanitize(S_set)
        return S_set

    def local_search_enhanced(S_list, V_set, max_iters=500, time_budget=30):
        """Enhanced local search with better move operators."""
        S_set = set(S_list)
        # Ensure start is valid
        if not is_sum_free(S_set):
            S_set = sanitize(S_set)
            
        best_size = len(S_set)
        no_improve_count = 0
        t_start = time.time()
        
        for it in range(max_iters):
            # Time check
            if time.time() - t_start > time_budget:
                break
            
            improved = False
            current_size = len(S_set)
            
            # Move 1: Try to add elements (most common improvement)
            candidates = [x for x in V_set if x not in S_set]
            random.shuffle(candidates)
            added_any = False
            for x in candidates[:50]:
                if can_add_fast(S_set, x):
                    S_set.add(x)
                    added_any = True
                    if len(S_set) > best_size:
                        best_size = len(S_set)
            if added_any:
                improved = True
                no_improve_count = 0
                continue
            
            # Move 2: Remove one element and try to add multiple
            S_list = list(S_set)
            random.shuffle(S_list)
            swap_improved = False
            for a in S_list[:20]:
                S_set.discard(a)
                # Try to add as many as possible
                candidates2 = [x for x in V_set if x not in S_set]
                random.shuffle(candidates2)
                added_count = 0
                temp_set = set(S_set)
                for x in candidates2[:80]:
                    if can_add_fast(temp_set, x):
                        temp_set.add(x)
                        added_count += 1
                if len(temp_set) > current_size:
                    # Verify temp_set is sum-free (it should be, but just in case)
                    if is_sum_free(temp_set):
                        S_set = temp_set
                        if len(S_set) > best_size:
                            best_size = len(S_set)
                        swap_improved = True
                        break
                else:
                    S_set.add(a)  # restore
            
            if swap_improved:
                improved = True
                no_improve_count = 0
                continue
            
            # Move 3: Swap two elements for two new ones
            S_list = list(S_set)
            if len(S_list) >= 2:
                random.shuffle(S_list)
                two_swap_improved = False
                for i in range(min(15, len(S_list))):
                    a = S_list[i]
                    for j in range(i+1, min(i+15, len(S_list))):
                        b = S_list[j]
                        S_set.discard(a)
                        S_set.discard(b)
                        # Try to add two new elements
                        candidates3 = [x for x in V_set if x not in S_set]
                        random.shuffle(candidates3)
                        temp_set = set(S_set)
                        added_count = 0
                        for x in candidates3[:100]:
                            if can_add_fast(temp_set, x):
                                temp_set.add(x)
                                added_count += 1
                                if added_count >= 2:
                                    break
                        if len(temp_set) > current_size:
                            if is_sum_free(temp_set):
                                S_set = temp_set
                                if len(S_set) > best_size:
                                    best_size = len(S_set)
                                two_swap_improved = True
                                break
                        else:
                            S_set.add(a)
                            S_set.add(b)
                    if two_swap_improved:
                        break
                if two_swap_improved:
                    improved = True
                    no_improve_count = 0
                    continue
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 80:
                    break
            else:
                no_improve_count = 0
        
        # Final sanity check
        if not is_sum_free(S_set):
            S_set = sanitize(S_set)
            
        return list(S_set)

    def search_instance(elements, seed_offset=0, time_budget=30):
        """Search for a large sum-free subset of elements."""
        rng = random.Random(seed_offset)
        best = []
        best_size = 0
        t_start = time.time()
        elements_set = set(elements)
        
        def try_candidate(cand):
            nonlocal best, best_size
            if time.time() - t_start > time_budget - 2:
                return
            S = repair_and_expand(cand, elements_set)
            S = local_search_enhanced(S, elements_set, max_iters=300, time_budget=time_budget - (time.time() - t_start) - 1)
            if len(S) > best_size:
                best_size = len(S)
                best = list(S)

        # Generate diverse initial candidates
        candidates = []

        # 1. Odds (guaranteed sum-free)
        odd = [x for x in elements if x % 2 != 0]
        candidates.append(odd)

        # 2. All positive
        pos = [x for x in elements if x > 0]
        candidates.append(pos)

        # 3. All negative
        neg = [x for x in elements if x < 0]
        candidates.append(neg)

        # 4. Top-k by value (largest)
        sorted_desc = sorted(elements, reverse=True)
        for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            k = int(len(elements) * frac)
            candidates.append(sorted_desc[:k])

        # 5. Top-k by magnitude
        sorted_mag = sorted(elements, key=lambda x: abs(x), reverse=True)
        for frac in [0.3, 0.4, 0.5, 0.6]:
            k = int(len(elements) * frac)
            candidates.append(sorted_mag[:k])

        # 6. Interval-based: elements in [L, R]
        elements_sorted = sorted(elements)
        n = len(elements_sorted)
        for lo_frac in [0.0, 0.1, 0.2, 0.3]:
            for hi_frac in [0.5, 0.7, 0.9, 1.0]:
                lo = int(n * lo_frac)
                hi = int(n * hi_frac)
                if hi > lo and hi - lo >= 5:
                    candidates.append(elements_sorted[lo:hi])

        # 7. Random samples with varying sizes
        for _ in range(12):
            k = int(len(elements) * rng.uniform(0.2, 0.9))
            candidates.append(rng.sample(elements, k))

        # 8. Mixed: some high, some low
        for _ in range(4):
            k_high = int(len(elements) * rng.uniform(0.1, 0.3))
            k_low = int(len(elements) * rng.uniform(0.1, 0.3))
            high = sorted_desc[:k_high]
            low = sorted_desc[-k_low:]
            candidates.append(high + low)

        # Process each candidate
        for cand in candidates:
            if time.time() - t_start > time_budget - 3:
                break
            try_candidate(cand)

        # Additional random restarts if time permits
        restart_count = 0
        while time.time() - t_start < time_budget - 3:
            k = int(len(elements) * rng.uniform(0.2, 0.9))
            cand = rng.sample(elements, k)
            try_candidate(cand)
            restart_count += 1
            if restart_count > 80:
                break

        # Final sanity check on best
        if best and not is_sum_free(set(best)):
            best = sanitize(set(best))
            
        return best

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))

    results = []
    # Allocate time budget: n=60 gets 25s, n=120 gets 40s, n=240 gets 55s (total 120s, leaving 25s margin)
    time_budgets = [25, 40, 55]
    for i, elems in enumerate(instances):
        if time.time() - start_time > DEADLINE:
            # Fallback: odd numbers
            results.append([x for x in elems if x % 2 != 0])
            continue
        result = search_instance(elems, seed_offset=801 + i, time_budget=time_budgets[i])
        results.append(result)

    return results
# EVOLVE-BLOCK-END
