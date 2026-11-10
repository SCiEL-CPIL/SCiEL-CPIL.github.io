"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START

# Precompute all forbidden triples: (a, b, c) where 1/a = 1/b + 1/c, all distinct
# We store for each pair (a, b) what c would complete the triple, and also build
# a set of all forbidden triples for fast lookup.
_FORBIDDEN_TRIPLES = set()
# Also build a dict: for each element x, which pairs (y, z) with x would form a triple
# Actually, let's precompute: for each (a,b) with a != b, compute c = ab/(a+b)
# If c is integer and c != a and c != b, then {a,b,c} is a forbidden triple.
_TRIPLE_MAP = {}  # frozenset of three -> True, but we need fast "does adding x create a triple"

def _precompute_triples():
    """Precompute all forbidden triples in [1, 180]."""
    triples = set()
    triple_count = [0] * 181  # triple_count[x] = number of triples containing x
    triple_partners = [set() for _ in range(181)]  # for each x, which pairs (y,z) form triple with x
    
    for a in range(1, 181):
        for b in range(a + 1, 181):  # b > a to avoid duplicates
            num = a * b
            den = a + b
            if num % den != 0:
                continue
            c = num // den
            if c == a or c == b:
                continue
            if c > 180:
                continue
            # We have a valid triple {a, b, c} with all distinct
            triple = frozenset((a, b, c))
            triples.add(triple)
            for x in (a, b, c):
                triple_count[x] += 1
                # Store the other two elements
                others = [y for y in (a, b, c) if y != x]
                triple_partners[x].add(tuple(sorted(others)))
    
    return triples, triple_count, triple_partners

# Precompute at module load time
_FORBIDDEN_TRIPLES, _TRIPLE_COUNT, _TRIPLE_PARTNERS = _precompute_triples()

def _has_triple_fast(s):
    """Fast check using precomputed triples: does set s contain any forbidden triple?"""
    for t in _FORBIDDEN_TRIPLES:
        if t.issubset(s):
            return True
    return False

def _would_create_triple(s, x):
    """Check if adding x to set s would create a forbidden triple."""
    # Only need to check triples that contain x
    for partner_pair in _TRIPLE_PARTNERS[x]:
        if partner_pair[0] in s and partner_pair[1] in s:
            return True
    return False

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under the 90s kill
    N = 180
    rng = random.Random(20240517)

    # Baseline seed: [91, 180]
    best = set(range(91, 181))

    # ---- Phase 1: greedy construction with multiple orderings ----
    # Start from large numbers (guaranteed safe) and try adding smaller ones.
    def greedy(order):
        s = set(range(91, 181))  # base
        for x in order:
            if x in s:
                continue
            if not _would_create_triple(s, x):
                s.add(x)
        return s

    # Candidate orderings for smaller numbers
    small = list(range(1, 91))
    for trial_iter in range(500):
        if time.time() >= deadline:
            break
        if trial_iter == 0:
            order = list(reversed(small))   # descending
        elif trial_iter == 1:
            order = small                    # ascending
        elif trial_iter == 2:
            order = sorted(small, key=lambda x: _TRIPLE_COUNT[x], reverse=True)  # most constrained first
        elif trial_iter == 3:
            order = sorted(small, key=lambda x: _TRIPLE_COUNT[x])  # least constrained first
        else:
            order = small[:]
            rng.shuffle(order)
        cand = greedy(order)
        if len(cand) > len(best):
            best = cand

    # ---- Phase 2: local search (add / swap) from the best set ----
    def try_add(s, x):
        if not _would_create_triple(s, x):
            trial = s | {x}
            return trial
        return None

    # Try adding one element at a time (greedy additions)
    for _ in range(1000):
        if time.time() >= deadline:
            break
        improved = False
        for x in range(1, 181):
            if x in best:
                continue
            cand = try_add(best, x)
            if cand is not None:
                best = cand
                improved = True
                break
        if not improved:
            break

    # Swap search: remove one element to allow another
    for _ in range(10000):
        if time.time() >= deadline:
            break
        x = rng.choice(range(1, 181))
        if x in best:
            continue
        # try removing a random element to fit x
        y = rng.choice(sorted(best))
        trial = (best - {y}) | {x}
        if not _has_triple_fast(trial) and len(trial) > len(best):
            best = trial

    # ---- Phase 3: random restarts with diverse seeds ----
    for it in range(100000):
        if time.time() >= deadline:
            break
        # Random seed set of moderate size, then greedily expand
        size = rng.randint(60, 130)
        s = set(rng.sample(range(1, 181), size))
        # Quick reject if seed already has triple
        if _has_triple_fast(s):
            continue
        # Greedy add remaining
        remaining = [x for x in range(1, 181) if x not in s]
        rng.shuffle(remaining)
        for x in remaining:
            if time.time() >= deadline:
                break
            if not _would_create_triple(s, x):
                s.add(x)
        if len(s) > len(best):
            best = s

    # Final verification
    if _has_triple_fast(best):
        # Fallback to the safe seed
        best = set(range(91, 181))

    return list(best)
# EVOLVE-BLOCK-END
