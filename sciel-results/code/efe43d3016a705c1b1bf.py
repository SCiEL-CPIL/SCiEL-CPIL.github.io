"""Search for maximum pairwise-noncoprime subset of [1,255] containing 255."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # safety margin below 90s limit

    n = 255
    # Universe: numbers in [1, n] that share a prime factor with n (i.e., divisible by 3, 5, or 17)
    # Any valid set containing 255 can only contain numbers that share a factor with 255 (since gcd(a,255)>1 required)
    primes_of_n = [3, 5, 17]

    def shares_factor_with_n(x):
        for p in primes_of_n:
            if x % p == 0:
                return True
        return False

    universe = [x for x in range(1, n + 1) if shares_factor_with_n(x)]
    # universe must include 255
    assert 255 in universe

    # For each number in universe, determine which of {3,5,17} divide it
    # Two numbers can be in the same pairwise-noncoprime set iff gcd(a,b) > 1
    # Since all numbers in universe are divisible by at least one of {3,5,17},
    # gcd(a,b) > 1 iff they share at least one common prime factor from {3,5,17}
    #   OR if they share some other common factor... wait, that's not quite right.
    # Actually two numbers both divisible by 3 have gcd >= 3.
    # A number divisible by 5 but not 3, and a number divisible by 3 but not 5: gcd could be 1 or more.
    # Example: 5 and 3 -> gcd 1. 5 and 6 -> gcd 1. 10 and 3 -> gcd 1.
    # But 10 and 15 -> gcd 5. 21 and 5 -> gcd 1.
    # So the condition is simply gcd(a,b) > 1 for all pairs.

    size = len(universe)
    # Build conflict graph: edge between a,b if gcd(a,b) == 1 (they CANNOT both be in the set)
    # We want maximum clique in the "compatibility" graph where edge means gcd > 1.
    # Equivalently, maximum independent set in the "conflict" graph where edge means gcd == 1.

    # For efficiency, let's think about this differently.
    # Partition universe by which primes from {3,5,17} divide each element:
    # Type determined by the subset of {3,5,17} that divides the number.
    # There are 7 non-empty types: {3}, {5}, {17}, {3,5}, {3,17}, {5,17}, {3,5,17}
    # Two numbers are compatible (gcd > 1) if their type-subsets intersect.
    # Wait - that's only true if we're considering ONLY factors from {3,5,17}.
    # A number like 25 (5^2) has type {5}. A number like 27 (3^3) has type {3}. gcd(25,27)=1. Correct.
    # A number like 9 (3^2) has type {3}. A number like 15 has type {3,5}. They share 3. Correct.
    # But consider 4 (not in universe since not div by 3,5,17) - irrelevant.
    # Consider 77 = 7*11 (not in universe).
    # What about numbers that have factors other than 3,5,17? E.g., 9 = 3^2 (type {3}), 25 = 5^2 (type {5}).
    # The key insight: if two numbers share a common prime from {3,5,17}, their gcd > 1.
    # If they do NOT share a common prime from {3,5,17}, can their gcd still be > 1?
    # E.g., a = 9 (type {3}), b = 27 (type {3}) - they share 3, covered.
    # a = 3*7=21 (type {3}), b = 5*7=35 (type {5}). gcd(21,35)=7 > 1! But they don't share a factor from {3,5,17}.
    # So my type-based reasoning is incomplete. We need to check actual gcd.

    # Let's just build the compatibility graph directly using gcd.
    # Size of universe: count of numbers in [1,255] divisible by 3, 5, or 17.
    # |A_3| = 85, |A_5| = 51, |A_17| = 15
    # |A_3 ∩ A_5| = |A_15| = 17, |A_3 ∩ A_17| = |A_51| = 5, |A_5 ∩ A_17| = |A_85| = 3
    # |A_3 ∩ A_5 ∩ A_17| = |A_255| = 1
    # |A_3 ∪ A_5 ∪ A_17| = 85 + 51 + 15 - 17 - 5 - 3 + 1 = 127

    # For maximum clique in compatibility graph with 127 nodes, we can use Bron-Kerbosch with pivot.
    # But first, let's try a simpler approach: since the baseline is 85 (all multiples of 3),
    # let's see if we can do better by a mixed strategy.

    # Approach: Use the fact that any valid set S containing 255 must have all elements sharing a factor with 255.
    # Moreover, if we group by "primary" factor (the smallest prime in {3,5,17} that divides the number),
    # we can try greedy/heuristic approaches.

    # Let's try: the set of all multiples of 3 has size 85.
    # Can we add any non-multiple of 3? A non-multiple of 3 in the universe is divisible by 5 and/or 17 but not 3.
    # For such an x to be compatible with all multiples of 3 currently in the set,
    # x must be divisible by 3 (contradiction) OR share another factor with each multiple of 3.
    # Since multiples of 3 include 3 itself, x must share a factor with 3, i.e., be divisible by 3. Contradiction.
    # So we can't just add to the full set of multiples of 3.
    # But we could remove some multiples of 3 and add multiples of 5 (not of 3) if it's net positive.

    # Strategy: For each "type" of number (which subset of {3,5,17} divides it), try to find
    # the maximum compatible set using integer programming or careful combinatorial search.

    # Let's use a simpler approach: since 127 is small enough, let's use a branch-and-bound maximum clique.
    # But we need to be fast. Let's try a greedy + local search first, then verify.

    # Actually, let me think about this more carefully with a type-based approach.
    # Define types by the exact set of primes from {3,5,17} dividing each number:
    types = {}  # type_tuple -> list of numbers
    for x in universe:
        divs = tuple(sorted(p for p in primes_of_n if x % p == 0))
        if divs not in types:
            types[divs] = []
        types[divs].append(x)

    # Two numbers of types T1 and T2 are "type-compatible" if T1 ∩ T2 != emptyset.
    # If type-compatible, they definitely have gcd > 1 (since they share a prime factor).
    # If type-incompatible (disjoint), they MIGHT still have gcd > 1 via another common factor,
    # but in most cases they won't. For safety, we should check actual gcd for type-incompatible pairs.

    # For a fast solution, let's assume type-compatible pairs are the main compatibility.
    # This gives us a complete multipartite-like structure.

    # The key observation: if we pick a set of types, and for each type we pick numbers,
    # all pairs must be compatible. Two numbers from the same type are always compatible (they share all primes in that type).
    # Two numbers from different types T1, T2 are compatible iff T1 ∩ T2 != ∅ (guaranteed) or gcd > 1 (extra).

    # Maximum set: we want to maximize total count. Since same-type numbers are mutually compatible,
    # we should include as many as possible from each chosen type. The question is which combination of types to use.

    # Types and their sizes:
    # ({3},): multiples of 3 only (not 5, not 17) = |A_3| - |A_15| - |A_51| + |A_255| = 85 - 17 - 5 + 1 = 64
    # ({5},): multiples of 5 only = 51 - 17 - 3 + 1 = 32
    # ({17},): multiples of 17 only = 15 - 5 - 3 + 1 = 8
    # ({3,5}): multiples of 15 but not 17 = 17 - 1 = 16
    # ({3,17}): multiples of 51 but not 5 = 5 - 1 = 4
    # ({5,17}): multiples of 85 but not 3 = 3 - 1 = 2
    # ({3,5,17}): multiples of 255 = 1

    # Now, which subsets of types are mutually type-compatible?
    # Two types are compatible if they share a prime.
    # ({3},) and ({5},) : disjoint -> NOT type-compatible
    # ({3},) and ({3,5}) : share 3 -> compatible
    # ({5},) and ({3,5}) : share 5 -> compatible
    # ({3},) and ({17},) : disjoint -> NOT type-compatible
    # ({5},) and ({17},) : disjoint -> NOT type-compatible
    # ({3},) and ({3,17}) : share 3 -> compatible
    # ({5},) and ({5,17}) : share 5 -> compatible
    # ({17},) and ({3,17}) : share 17 -> compatible
    # ({17},) and ({5,17}) : share 17 -> compatible
    # ({3,5}) and ({17},) : disjoint -> NOT type-compatible
    # ({3,17}) and ({5},) : disjoint -> NOT type-compatible
    # ({5,17}) and ({3},) : disjoint -> NOT type-compatible

    # So for a purely type-based selection to work, all chosen types must pairwise intersect.
    # Let's find all maximal sets of pairwise-intersecting types (from our 7 types).

    type_keys = [tuple(k) for k in types.keys()]
    type_sizes = {k: len(v) for k, v in types.items()}

    def types_compatible(t1, t2):
        s1 = set(t1)
        s2 = set(t2)
        if s1 & s2:
            return True
        return False  # Conservative: assume incompatible if no shared prime

    # Find all subsets of types where every pair is type-compatible
    n_types = len(type_keys)
    # Try all subsets (2^7 = 128, very manageable)
    best_subset = None
    best_size = 0
    for mask in range(2**n_types):
        chosen = [type_keys[i] for i in range(n_types) if mask & (1 << i)]
        if not chosen:
            continue
        # Check pairwise compatibility
        valid = True
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if not types_compatible(chosen[i], chosen[j]):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        total = sum(type_sizes[t] for t in chosen)
        # Must include 255, which is type ({3,5,17})
        if (3, 5, 17) not in chosen:
            # Can we still have 255? No, 255 has type (3,5,17), so it's in types[(3,5,17)].
            # If we don't choose that type, we exclude 255. Not allowed.
            continue
        if total > best_size:
            best_size = total
            best_subset = chosen

    # Best type-based solution
    if best_subset:
        result = []
        for t in best_subset:
            result.extend(types[t])
        result.sort()
        return result

    # Fallback: just return multiples of 3
    return list(range(3, n + 1, 3))
# EVOLVE-BLOCK-END
