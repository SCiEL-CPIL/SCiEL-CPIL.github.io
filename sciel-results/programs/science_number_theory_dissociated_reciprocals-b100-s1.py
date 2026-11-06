"""Model-and-optimize: maximize cardinality of a dissociated reciprocal subset of [1,22].

A set S is dissociated (all reciprocal subset sums distinct) iff adding each new
reciprocal never produces a subset-sum collision. We build greedily over random
orderings of [1..22] using exact Fraction arithmetic, with a wall-clock deadline
guard, and keep the largest valid set found.
"""
import time
import random
from fractions import Fraction

_DEADLINE = 75.0  # seconds; safety margin before the 90 s evaluator kill
_START = time.time()

def _deadline_hit():
    return (time.time() - _START) > _DEADLINE

def _try_greedy(order):
    """Greedily add elements in `order`, keeping a set only if no sum collision.
    Returns the final list (possibly small)."""
    sums = {Fraction(0)}          # current subset sums (normalized Fractions)
    chosen = []
    for a in order:
        r = Fraction(1, a)
        # new sums are {s + r for s in sums}; must be disjoint from `sums`
        # (if disjoint from sums, they're also distinct among themselves since
        #  sums are distinct).
        new_sums = [s + r for s in sums]
        collides = False
        for ns in new_sums:
            if ns in sums:
                collides = True
                break
        if collides:
            continue
        # commit
        sums.update(new_sums)
        chosen.append(a)
    return chosen

def _try_greedy_with_local_search(order, best_k):
    """Greedily add elements in `order`, then try local search to improve.
    Returns the final list (possibly small)."""
    sums = {Fraction(0)}          # current subset sums (normalized Fractions)
    chosen = []
    chosen_set = set()
    
    # First pass: greedy
    for a in order:
        if a in chosen_set:
            continue
        r = Fraction(1, a)
        new_sums = [s + r for s in sums]
        collides = False
        for ns in new_sums:
            if ns in sums:
                collides = True
                break
        if collides:
            continue
        sums.update(new_sums)
        chosen.append(a)
        chosen_set.add(a)
    
    if len(chosen) <= best_k:
        return chosen
    
    # Local search: try to swap out elements to make room for better ones
    # We'll try removing one element and adding another
    full_set = set(range(1, 23))
    current_set = set(chosen)
    
    for _ in range(20):  # limited iterations for local search
        if _deadline_hit():
            break
        
        # Try each possible swap
        found_improvement = False
        for remove_a in list(current_set):
            if len(current_set) - 1 + 1 <= best_k:
                continue
                
            # Compute sums without remove_a
            sums_without = {Fraction(0)}
            for b in current_set:
                if b == remove_a:
                    continue
                r_b = Fraction(1, b)
                new_sums = [s + r_b for s in sums_without]
                # Check if all new sums are distinct from existing
                valid = True
                for ns in new_sums:
                    if ns in sums_without:
                        valid = False
                        break
                if valid:
                    sums_without.update(new_sums)
            
            if len(sums_without) != 2 ** (len(current_set) - 1):
                continue  # Invalid state
            
            # Try to add an element not in current_set
            for add_a in full_set - current_set:
                r_add = Fraction(1, add_a)
                new_sums = [s + r_add for s in sums_without]
                collides = False
                for ns in new_sums:
                    if ns in sums_without:
                        collides = True
                        break
                if not collides:
                    # Found a valid swap
                    current_set.remove(remove_a)
                    current_set.add(add_a)
                    found_improvement = True
                    break
            if found_improvement:
                break
        
        if not found_improvement:
            break
    
    return list(current_set)

def construct():
    global _START
    _START = time.time()
    random.seed()
    full = list(range(1, 23))

    # Deterministic seed baseline (known valid, 9 elements).
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best = seed
    best_k = len(seed)

    # Random restarts: greedy over random permutations with local search.
    while not _deadline_hit():
        order = full[:]
        random.shuffle(order)
        cand = _try_greedy_with_local_search(order, best_k)
        if len(cand) > best_k:
            best_k = len(cand)
            best = cand
            # If we reached the theoretical-ish cap, keep going a bit more,
            # but stop early if we've found a very large set to save time.
            if best_k >= 13:
                break

    if best is None or len(best) == 0:
        best = [1]
    return list(best)
