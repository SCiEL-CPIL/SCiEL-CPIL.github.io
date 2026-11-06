import itertools
import random
import time

def construct():
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Precompute: for each triple, what are the K(2,2,2)s it belongs to?
    # A triple {a,b,c} is a transversal of K(2,2,2) with pairs P1,P2,P3 where
    # a in P1, b in P2, c in P3 (in some order).
    # For triple {a,b,c}, the possible pairs are: for each element, pair it with one of the other 15 points.
    # P1 contains a and some a', P2 contains b and some b', P3 contains c and some c'.
    # We need P1, P2, P3 disjoint.
    
    # For efficiency, let's precompute for each triple, the list of K(2,2,2)s (as frozensets of 8 triples)
    # that contain it.
    
    # Actually, let's think about it this way: when we add triple T, it completes a K(2,2,2) if there
    # exist 3 disjoint pairs such that all 8 transversals are in our set (including T).
    
    # Alternative: for each triple T = {a,b,c}, the K(2,2,2)s containing T are determined by choosing
    # a' != a, b' != b, c' != c such that {a,a'}, {b,b'}, {c,c'} are disjoint (and a' != b, b' != a, etc.)
    
    # Let me precompute for each triple, the set of "companion" triples that would complete a K(2,2,2)
    
    # For triple T = (a,b,c), a K(2,2,2) containing T is determined by choosing:
    # - a' from [n]\{a,b,c} (the partner of a)
    # - b' from [n]\{a,b,c,a'} (the partner of b, must be different from a')
    # - c' from [n]\{a,b,c,a',b'} (the partner of c)
    # But we also need the pairs to be disjoint: a' != b, b' != a, etc. Since a,b,c are distinct and 
    # a',b',c' are chosen from outside {a,b,c}, the only constraint is a', b', c' are distinct.
    
    # Wait, we also need: the pair {a,a'} must not contain b or c (already ensured since a' not in {a,b,c}),
    # and similarly for others. And the 6 points a,a',b,b',c,c' must be distinct.
    
    # So for T=(a,b,c), the number of K(2,2,2)s containing it is C(13,3) = 286? No:
    # Choose a' from 13 options (not a,b,c), b' from 12 (not a,b,c,a'), c' from 11 (not a,b,c,a',b').
    # But we can permute which element gets which partner. Actually, the pairs are labeled by which 
    # element of T they contain. So it's 13 * 12 * 11 = 1716? No, that's overcounting.
    
    # Actually, the K(2,2,2) is determined by the 3 pairs. If T = {a,b,c}, then a is in one pair, b in another, c in another.
    # The partner of a can be any of the 13 points not in {a,b,c}. Similarly for b and c, with distinctness.
    # So 13 * 12 * 11 = 1716 K(2,2,2)s contain a given triple. That's a lot.
    
    # Let me use a different approach. I'll use a greedy algorithm with a constraint check.
    
    # Start with the star at point 0.
    star = set()
    for a in range(1, n):
        for b in range(a+1, n):
            star.add((0, a, b))
    
    result = list(star)
    result_set = set(result)
    
    # Try to add triples not in the star, checking the constraint.
    # For efficiency, I'll check: if I add triple T, does it complete any K(2,2,2)?
    # T completes a K(2,2,2) if there exist 3 disjoint pairs P1,P2,P3 such that T is a transversal
    # and all 8 transversals of (P1,P2,P3) are in result_set ∪ {T}.
    
    # Since T is new, the other 7 transversals must already be in result_set.
    
    # For T = (a,b,c), the 7 other transversals in a given K(2,2,2) are:
    # If pairs are {a,a'}, {b,b'}, {c,c'}, the other transversals are:
    # {a',b,c}, {a,b',c}, {a,b,c'}, {a',b',c}, {a',b,c'}, {a,b',c'}, {a',b',c'}
    
    # So I need to check: for any a' not in {a,b,c}, b' not in {a,b,c,a'}, c' not in {a,b,c,a',b'},
    # are all 7 of those triples in result_set?
    
    # This is O(13*12*11) = 1716 checks per candidate triple, each check is 7 set lookups.
    # Total: ~12000 operations per candidate. With ~5000 candidates, that's ~60M operations. Might be slow.
    
    # Let me optimize: precompute for each triple, which other triples are its "7 companions" in each K(2,2,2).
    
    # Actually, let me just try it and see if it's fast enough.
    
    candidates = [t for t in all_triples if t not in result_set]
    random.shuffle(candidates)
    
    for T in candidates:
        a, b, c = T
        # Check if adding T completes any K(2,2,2)
        completes = False
        remaining = [x for x in range(n) if x not in (a,b,c)]
        for a_prime in remaining:
            rem2 = [x for x in remaining if x != a_prime]
            for b_prime in rem2:
                rem3 = [x for x in rem2 if x != b_prime]
                for c_prime in rem3:
                    # Check the 7 other transversals
                    other_triples = [
                        (a_prime, b, c),
                        (a, b_prime, c),
                        (a, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b, c_prime),
                        (a, b_prime, c_prime),
                        (a_prime, b_prime, c_prime),
                    ]
                    # Normalize to sorted tuples for set lookup
                    all_present = True
                    for ot in other_triples:
                        if tuple(sorted(ot)) not in result_set:
                            all_present = False
                            break
                    if all_present:
                        completes = True
                        break
                if completes:
                    break
            if completes:
                break
        if not completes:
            result_set.add(tuple(sorted(T)))
            result.append(list(T))
    
    return result
