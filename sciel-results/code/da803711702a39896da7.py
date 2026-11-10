"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    n = 192
    max_elements = 48
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Precompute: for each candidate x, which other values y could form a violating triple?
    # We need: for selected set S, no a in S divides sum of any 3 distinct others in S\{a}.
    # Strategy: greedy with backtracking, using bitsets for speed.
    
    # Bitset representation: use Python integers as bitsets (bit i set means value i+1 is selected)
    # Values 1..192, so we need 193 bits.
    
    # Precompute for each value v (1..192): the set of values u such that v divides u
    # Actually, we need: for a selected set S, check if any a in S divides sum of 3 distinct others.
    
    # Better approach: greedy construction with incremental validity checking.
    # Use bitsets to represent the selected set.
    
    # For efficiency, precompute all triples and their sums, and for each sum s, which divisors d in [1,192] divide s.
    # Then when we have a set S, for each a in S, we check if there exist 3 distinct elements in S\{a} whose sum is divisible by a.
    
    # Let's use a smarter greedy: try to build a large set.
    # Key insight: larger numbers are less likely to divide sums of 3 others (since sums are bounded by 3*192=576).
    # So prefer larger numbers.
    
    # Precompute: for each a in 1..192, and for each possible sum s (3..576), whether a divides s.
    # Actually, let's precompute for each a, the set of sums (as a bitmask or set) that are multiples of a.
    
    # For the greedy approach:
    # Start with empty set. Try adding candidates in some order.
    # When adding candidate c to set S, check:
    #   1. c does not divide sum of any 3 distinct elements in S
    #   2. No element a in S divides sum of any 3 distinct elements in (S + {c})\{a}
    
    # This is expensive. Let's use a different strategy.
    
    # Alternative: Use the fact that if all selected numbers are > 288, then no sum of 3 others (max 3*192=576) can be divisible by any of them unless the sum is 0, which is impossible. Wait, no: if a > 576, then a cannot divide any positive sum of 3 numbers <= 192. But max value is 192, so a <= 192.
    
    # If a > 576/1 = 576, impossible since a <= 192.
    # If a > 576/3 = 192, i.e., a >= 193, then a cannot divide any sum of 3 numbers each <= 192 (max sum = 576). But a <= 192.
    
    # For a = 192: 192 divides s only if s is a multiple of 192. Possible sums: 192, 384, 576.
    # For a = 100: 100 divides s if s is multiple of 100: 100, 200, 300, 400, 500.
    
    # Key observation: If we select only numbers in range (576/3, 192] = (192, 192], that's empty.
    # If we select numbers in (576/4, 192] = (144, 192], i.e., 145..192, that's 48 numbers!
    # For any a in [145, 192], the only multiples of a in range [3, 576] are: a, 2a, 3a.
    #   - a: need 3 distinct numbers in S\{a} summing to a. But each number >= 145, so sum >= 3*145 = 435 > 192 >= a. Impossible.
    #   - 2a: need 3 distinct numbers summing to 2a. Min sum = 435. 2a <= 384. 435 > 384. Impossible.
    #   - 3a: need 3 distinct numbers summing to 3a. Min sum = 435. 3a >= 435 when a >= 145. So 3a is in [435, 576].
    #     We need 3 distinct numbers from S\{a}, each in [145,192], summing to 3a.
    #     The max sum of 3 distinct numbers from [145,192] is 192+191+190 = 573.
    #     So we need 3a <= 573, i.e., a <= 191. And 3a >= 435, i.e., a >= 145.
    #     So for a in [145, 191], we need to check if 3 distinct numbers in S\{a} sum to 3a.
    #     For a = 192: 3*192 = 576 > 573, impossible.
    
    # So if we pick all numbers 145..192 (48 numbers), we need to verify no a in this set has 3 distinct others summing to 3a.
    
    # Let's just try this set and check validity.
    
    candidate_set = list(range(145, 193))  # 48 numbers
    # Check validity
    valid = True
    for a in candidate_set:
        others = [x for x in candidate_set if x != a]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                valid = False
                break
        if not valid:
            break
    
    if valid:
        return candidate_set
    
    # If not valid, we need to remove some elements. Let's use greedy removal.
    # Start with all 48, and remove elements that cause violations.
    
    selected = set(candidate_set)
    changed = True
    while changed:
        changed = False
        for a in list(selected):
            others = [x for x in selected if x != a]
            for triple in combinations(others, 3):
                if sum(triple) % a == 0:
                    # Remove a
                    selected.discard(a)
                    changed = True
                    break
            if changed:
                break
        # Also check: for each a in selected, does any triple in selected\{a} sum to a multiple of a?
        # The above loop checks this.
    
    # Now try to add back some removed elements
    removed = set(candidate_set) - selected
    for x in sorted(removed, reverse=True):
        trial = selected | {x}
        ok = True
        for a in trial:
            others = [v for v in trial if v != a]
            for triple in combinations(others, 3):
                if sum(triple) % a == 0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected = trial
    
    if len(selected) >= len(candidate_set) - 5:
        return list(selected)
    
    # If we couldn't get close to 48, try a different range
    # Try 140..192 (53 numbers), pick best 48
    for start in range(140, 130, -1):
        if time.time() > deadline:
            break
        cand = list(range(start, 193))
        if len(cand) < 48:
            continue
        # Greedily select up to 48 from cand
        sel = []
        for x in sorted(cand, reverse=True):
            if len(sel) >= 48:
                break
            trial = sel + [x]
            ok = True
            for a in trial:
                others = [v for v in trial if v != a]
                for triple in combinations(others, 3):
                    if sum(triple) % a == 0:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                sel = trial
        if len(sel) >= len(selected):
            selected = set(sel)
    
    return list(selected)
# EVOLVE-BLOCK-END
