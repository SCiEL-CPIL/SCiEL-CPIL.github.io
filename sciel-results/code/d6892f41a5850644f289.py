"""Rewritten program: greedy + local search with fast feasibility checks and deadline guard."""
import time
import random

_N = 128
# Precompute all "bad triples": distinct a<b<c such that a+b, a+c, b+c are all in 1..N
# We'll precompute the list of triples to speed up validation.
_TRIPLES = []
def _build_triples():
    triples = []
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            for c in range(b + 1, _N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 <= _N and s2 <= _N and s3 <= _N:
                    triples.append((a, b, c, s1, s2, s3))
    return triples
_TRIPLES = _build_triples()

def _is_valid(set_ints):
    """Check if a set of ints (as a set) is valid."""
    s = set_ints if isinstance(set_ints, set) else set(set_ints)
    for a, b, c, s1, s2, s3 in _TRIPLES:
        if a in s and b in s and c in s:
            if s1 in s and s2 in s and s3 in s:
                return False
    return True

def _can_add(s_set, x):
    """Check if adding x to s_set keeps it valid. x not in s_set."""
    for a, b, c, s1, s2, s3 in _TRIPLES:
        # We need to check if adding x creates a new violation.
        # A violation occurs if there exist distinct p,q,r in (s_set ∪ {x}) such that
        # p+q, p+r, q+r are all in (s_set ∪ {x}).
        # Since s_set is valid, any new violation must involve x.
        # Cases:
        # 1) x is one of {a,b,c} and s1,s2,s3 are in s_set ∪ {x}
        # 2) x is one of {s1,s2,s3} and a,b,c are in s_set ∪ {x}
        # Note: x could be both a sum and a member, but since all are distinct in the triple
        # (a,b,c distinct and sums are > max(a,b,c) so sums != a,b,c), x can't be in both
        # {a,b,c} and {s1,s2,s3} for the same triple.
        
        if x in (a, b, c):
            # Check if the other two of {a,b,c} are in s_set, and all three sums are in s_set ∪ {x}
            if (a in s_set or a == x) and (b in s_set or b == x) and (c in s_set or c == x):
                # All three of a,b,c are in s_set ∪ {x}
                if (s1 in s_set or s1 == x) and (s2 in s_set or s2 == x) and (s3 in s_set or s3 == x):
                    return False
        elif x in (s1, s2, s3):
            # Check if a,b,c are in s_set ∪ {x} (x is a sum, so x not in a,b,c)
            if a in s_set and b in s_set and c in s_set:
                if (s1 in s_set or s1 == x) and (s2 in s_set or s2 == x) and (s3 in s_set or s3 == x):
                    return False
    return True

def _greedy_construct(order):
    """Greedy construction with given order of candidates."""
    s = set()
    for x in order:
        if x not in s and _can_add(s, x):
            s.add(x)
    return s

def _local_search(s_set, max_iters=20000, deadline=None):
    """Local search: try to add elements, and swap to improve."""
    s = set(s_set)
    candidates = [x for x in range(1, _N + 1) if x not in s]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        added = False
        for x in candidates:
            if _can_add(s, x):
                s.add(x)
                candidates.remove(x)
                added = True
                break
        if not added:
            # Try swaps: remove y, add x where x > y (by value) to keep size but maybe enable more
            # Actually, we want to maximize size, so we should only swap if it allows adding more later.
            # For simplicity, try single removals to see if we can then add two.
            # This is complex. Let's just try random removals and re-greedy.
            if candidates:
                # Try removing a random element and see if we can add a larger one
                y = random.choice(list(s))
                s.discard(y)
                # Try to add back y or others
                for x in candidates:
                    if _can_add(s, x):
                        s.add(x)
                        candidates.remove(x)
                        break
                # If we couldn't add anything, put y back
                if y not in s:
                    s.add(y)
                    if x in s:
                        s.discard(x)
                        candidates.append(x)
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline, 5s safety margin
    
    # Initial greedy: use the seed approach (upper half) as a starting point, then improve
    # Actually, let's try multiple strategies and keep the best.
    
    best_set = set()
    
    # Strategy 1: Greedy from 128 down to 1
    s1 = _greedy_construct(range(_N, 0, -1))
    if len(s1) > len(best_set):
        best_set = s1
    
    # Strategy 2: Greedy from 1 to 128
    s2 = _greedy_construct(range(1, _N + 1))
    if len(s2) > len(best_set):
        best_set = s2
    
    # Strategy 3: Greedy with random orders
    for _ in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, _N + 1))
        random.shuffle(order)
        s3 = _greedy_construct(order)
        if len(s3) > len(best_set):
            best_set = s3
    
    # Local search on the best
    best_set = _local_search(best_set, max_iters=50000, deadline=deadline)
    
    # Try a few more local search passes with different random seeds
    for _ in range(5):
        if time.time() > deadline:
            break
        # Perturb the best set
        perturbed = set(best_set)
        for _ in range(random.randint(1, 5)):
            if perturbed:
                perturbed.discard(random.choice(list(perturbed)))
        improved = _local_search(perturbed, max_iters=10000, deadline=deadline)
        if len(improved) > len(best_set):
            best_set = improved
    
    # Final validation
    if not _is_valid(best_set):
        # Fallback to a known valid set
        best_set = set(range(_N // 2 + 1, _N + 1))
    
    return list(best_set)
