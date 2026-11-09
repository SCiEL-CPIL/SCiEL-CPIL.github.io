"""Fast bitset-based search for reciprocal weight of 3-progression-free sets on [1,256].

Combines:
- Bitset representation + precomputed AP-pairs (fast can_add) from inspiration 1
- Multi-stage local search with 1-for-2 swaps (replace one element with two smaller)
- Multiple randomized greedy initializations for diverse starting points
"""
import time
import random
from fractions import Fraction

N = 256
TIME_LIMIT = 55.0  # safety margin before 60s kill

# Precompute reciprocal Fractions
recip_frac = [Fraction(0, 1)] * (N + 1)
for i in range(1, N + 1):
    recip_frac[i] = Fraction(1, i)

# Precompute AP pairs for each x
# ap_pairs[x] is a list of tuples (a, b) such that {x, a, b} forms a 3-AP with a,b != x
ap_pairs = [[] for _ in range(N + 1)]

def precompute_ap_pairs():
    for x in range(1, N + 1):
        pairs = set()
        # Case 1: x is first term: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            y = x + d
            z = x + 2 * d
            if y <= N and z <= N:
                pairs.add((y, z))
        # Case 2: x is middle term: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            y = x - d
            z = x + d
            if y >= 1 and z <= N:
                pairs.add((y, z))
        # Case 3: x is last term: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            y = x - 2 * d
            z = x - d
            if y >= 1 and z <= N:
                pairs.add((y, z))
        ap_pairs[x] = list(pairs)

precompute_ap_pairs()

def can_add(bitset, x):
    """Check if adding x to bitset creates a 3-AP. Returns True if safe."""
    for a, b in ap_pairs[x]:
        if (bitset & (1 << (a - 1))) and (bitset & (1 << (b - 1))):
            return False
    return True

def bitset_to_list(bitset):
    """Convert bitset to sorted list of elements."""
    return [i for i in range(1, N + 1) if bitset & (1 << (i - 1))]

def list_to_bitset(lst):
    b = 0
    for x in lst:
        b |= (1 << (x - 1))
    return b

def bitset_score(bitset):
    """Exact reciprocal sum of a bitset."""
    total = Fraction(0, 1)
    for i in range(1, N + 1):
        if bitset & (1 << (i - 1)):
            total += recip_frac[i]
    return total

def greedy_construct(order):
    """Greedy: add elements in given order if they don't create a 3-AP."""
    bitset = 0
    for x in order:
        if can_add(bitset, x):
            bitset |= (1 << (x - 1))
    return bitset

def greedy_by_reciprocal():
    """Greedy in order of decreasing 1/x (smallest x first)."""
    return greedy_construct(list(range(1, N + 1)))

def greedy_randomized(seed, trials=10):
    """Multiple randomized greedy runs, return best bitset."""
    rng = random.Random(seed)
    best_bitset = 0
    best_score = Fraction(0, 1)
    for _ in range(trials):
        candidates = list(range(1, N + 1))
        weights = [1.0 / x for x in candidates]
        noisy = [w * (0.5 + rng.random()) for w in weights]
        order = sorted(candidates, key=lambda x: noisy[x - 1], reverse=True)
        b = greedy_construct(order)
        s = bitset_score(b)
        if s > best_score:
            best_score = s
            best_bitset = b
    return best_bitset

def local_search_single_swap(bitset, deadline):
    """
    Local search: try all single-element swaps (remove one, add another)
    to find the best improvement. Repeat until no improvement.
    """
    current = bitset
    current_score = bitset_score(current)
    best = current
    best_score = current_score

    while time.time() < deadline:
        improved = False
        best_delta = Fraction(0, 1)
        best_remove = None
        best_add = None

        # Get current elements
        elements = bitset_to_list(current)

        for removed in elements:
            temp = current & ~(1 << (removed - 1))
            # Try adding each candidate not in temp
            for add_candidate in range(1, N + 1):
                if temp & (1 << (add_candidate - 1)):
                    continue
                if can_add(temp, add_candidate):
                    delta = recip_frac[add_candidate] - recip_frac[removed]
                    if delta > best_delta:
                        best_delta = delta
                        best_remove = removed
                        best_add = add_candidate

        if best_remove is not None and best_delta > 0:
            current = (current & ~(1 << (best_remove - 1))) | (1 << (best_add - 1))
            current_score += best_delta
            improved = True
            if current_score > best_score:
                best_score = current_score
                best = current
        else:
            break

    return best, best_score

def local_search_1for2_swap(bitset, deadline):
    """
    Local search with 1-for-2 swaps: try removing one element and adding two
    smaller elements (net gain in reciprocal sum). This can increase the
    reciprocal sum significantly.
    """
    current = bitset
    current_score = bitset_score(current)
    best = current
    best_score = current_score

    while time.time() < deadline:
        improved = False
        best_delta = Fraction(0, 1)
        best_remove = None
        best_add1 = None
        best_add2 = None

        elements = bitset_to_list(current)

        for removed in elements:
            temp = current & ~(1 << (removed - 1))
            # Try all pairs (a, b) with a < b, both not in temp, both can be added
            # To limit search, only consider a, b < removed (smaller = better reciprocal)
            for a in range(1, removed):
                if temp & (1 << (a - 1)):
                    continue
                if not can_add(temp, a):
                    continue
                temp2 = temp | (1 << (a - 1))
                for b in range(a + 1, removed):
                    if temp2 & (1 << (b - 1)):
                        continue
                    if not can_add(temp2, b):
                        continue
                    delta = recip_frac[a] + recip_frac[b] - recip_frac[removed]
                    if delta > best_delta:
                        best_delta = delta
                        best_remove = removed
                        best_add1 = a
                        best_add2 = b

        if best_remove is not None and best_delta > 0:
            current = (current & ~(1 << (best_remove - 1))) | (1 << (best_add1 - 1)) | (1 << (best_add2 - 1))
            current_score += best_delta
            improved = True
            if current_score > best_score:
                best_score = current_score
                best = current
        else:
            break

    return best, best_score

def try_add_all(bitset, deadline):
    """Try to add any element that can be added (greedy completion)."""
    current = bitset
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if not (current & (1 << (x - 1))):
                if can_add(current, x):
                    current |= (1 << (x - 1))
                    improved = True
    return current

def construct():
    """Main entry point: find a 3-free subset of [1,256] maximizing sum of reciprocals."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # Phase 1: Get diverse starting points via multiple greedy runs
    best_bitset = greedy_by_reciprocal()
    best_score = bitset_score(best_bitset)

    seed = 0
    while time.time() < deadline - 25:
        candidate = greedy_randomized(seed, trials=15)
        if candidate:
            val = bitset_score(candidate)
            if val > best_score:
                best_score = val
                best_bitset = candidate
        seed += 1

    # Phase 2: Try adding all possible elements
    best_bitset = try_add_all(best_bitset, deadline - 20)
    best_score = bitset_score(best_bitset)

    # Phase 3: Single-swap local search
    best_bitset, best_score = local_search_single_swap(best_bitset, deadline - 10)

    # Phase 4: 1-for-2 swap local search (the key refinement)
    best_bitset, best_score = local_search_1for2_swap(best_bitset, deadline - 3)

    # Phase 5: Final single-swap polish
    best_bitset, best_score = local_search_single_swap(best_bitset, deadline)

    # Phase 6: Try adding any remaining elements
    best_bitset = try_add_all(best_bitset, deadline)

    return bitset_to_list(best_bitset)
