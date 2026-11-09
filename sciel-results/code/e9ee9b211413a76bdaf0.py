"""Rewritten program: maximize subset of [1,128] with no equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 80  # Safety margin
    
    N = 128
    # Precompute GCD table for fast access
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def can_add(selected, x):
        """
        Check if x can be added to selected without creating a bad triple.
        A bad triple (a, b, x) exists if gcd(a,b) == gcd(a,x) == gcd(b,x).
        """
        n = len(selected)
        # Iterate over pairs in selected
        for i in range(n):
            a = selected[i]
            ga_x = gcd_table[a][x]
            for j in range(i + 1, n):
                b = selected[j]
                if gcd_table[a][b] == ga_x and gcd_table[b][x] == ga_x:
                    return False
        return True

    def greedy_build(order):
        selected = []
        for x in order:
            if can_add(selected, x):
                selected.append(x)
        return selected

    def local_search_fill(selected):
        """
        Try to add any remaining element that fits.
        This is a local search with 'add' moves.
        """
        selected_set = set(selected)
        all_nums = set(range(1, N + 1))
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of the list of candidates to avoid modification issues
            # We can just check all numbers not in selected_set
            for x in all_nums - selected_set:
                if can_add(selected, x):
                    selected.append(x)
                    selected_set.add(x)
                    changed = True
        return selected

    def local_search_swap(selected):
        """
        Try to swap elements to potentially unlock more additions or increase size.
        Since we only care about size, and 'add' is always better than 'swap' if possible,
        we only do swaps if we are stuck (no adds possible).
        A swap (remove a, add b) keeps size same. It might allow future adds.
        However, if 'add' is exhaustive, and we are stuck, a swap might help.
        But if no add is possible, is a swap useful?
        If we swap a->b, we might make it possible to add c.
        So: Try all swaps. If a swap allows an add, do it.
        """
        selected_set = set(selected)
        all_nums = set(range(1, N + 1))
        candidates = list(all_nums - selected_set)
        
        improved = False
        # To keep it fast, limit swaps
        max_swaps = 100
        swaps_done = 0
        
        while swaps_done < max_swaps and time.time() < deadline - 5:
            improved_this_round = False
            for a in selected:
                if time.time() > deadline - 5:
                    break
                for b in candidates:
                    if b in selected_set:
                        continue
                    # Try removing a and adding b
                    # Check if b can be added to (selected - {a})
                    # This is expensive. 
                    # Optimization: if can_add(selected, b) is False, 
                    # it's because of some pair (u,v) in selected.
                    # If a is not in that pair, removing a doesn't help.
                    # This logic is complex. Just check directly.
                    
                    # Create a temporary list without a
                    # Since selected is small, list comprehension is fine
                    temp_selected = [x for x in selected if x != a]
                    
                    if can_add(temp_selected, b):
                        # Check if this swap actually helps?
                        # We don't know if it helps immediately.
                        # But if we do the swap, we might be able to add more.
                        # Let's just do the swap if it's valid.
                        # But we need to ensure we don't oscillate.
                        # Heuristic: Only swap if b is "better" than a?
                        # Hard to define.
                        # Let's just try to add after swap.
                        new_selected = temp_selected + [b]
                        new_selected_set = set(new_selected)
                        
                        # Try to fill again
                        new_selected = local_search_fill(new_selected)
                        
                        if len(new_selected) > len(selected):
                            selected = new_selected
                            selected_set = set(selected)
                            candidates = list(all_nums - selected_set)
                            improved = True
                            improved_this_round = True
                            break
                if improved_this_round:
                    break
            
            if not improved_this_round:
                break
            swaps_done += 1
            
        return selected

    nums = list(range(1, N + 1))
    best = []

    # Heuristic orderings
    orderings = []
    
    # 1. Ascending
    orderings.append(nums[:])
    # 2. Descending
    orderings.append(nums[::-1])
    
    # 3. By number of divisors (descending)
    def div_count(n):
        c = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                c += 1
                if i != n // i:
                    c += 1
        return c
    
    orderings.append(sorted(nums, key=lambda x: -div_count(x)))
    orderings.append(sorted(nums, key=lambda x: div_count(x)))
    
    # 4. Primes first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [n for n in nums if is_prime(n)]
    composites = [n for n in nums if not is_prime(n)]
    orderings.append(primes + composites)
    orderings.append(composites + primes)

    # 5. Random restarts
    rng = random.Random(42)
    for _ in range(30):
        o = nums[:]
        rng.shuffle(o)
        orderings.append(o)

    for order in orderings:
        if time.time() > deadline - 10:
            break
        
        # Greedy build
        cand = greedy_build(order)
        
        # Local search fill (add any possible)
        cand = local_search_fill(cand)
        
        # Local search swap (try to improve if stuck)
        # Only do swaps if we have time
        if time.time() < deadline - 5:
            cand = local_search_swap(cand)
        
        if len(cand) > len(best):
            best = cand
            # If we reach a high score, we can stop early?
            # Max possible is 128. Unlikely to reach.
            # But if we reach, say, 60, it's good.
            
    # Ensure non-empty
    if not best:
        best = [1]
        
    return best
# EVOLVE-BLOCK-END
