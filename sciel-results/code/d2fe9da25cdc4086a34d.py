"""Structural rewrite: precompute all 32-subsets via greedy + local search with bitset scoring."""
import time
import random
from itertools import combinations

# Deadline guard
_DEADLINE = 150.0  # seconds
_START_TIME = None


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _time_exceeded(margin=5.0):
    return _elapsed() > (_DEADLINE - margin)


def _score_set(s):
    """Return the number of distinct d>0 such that a, a+d, a+2d in s for some a."""
    s_set = set(s)
    diffs = set()
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n):
            d = s[j] - s[i]
            if d <= 0:
                continue
            a = s[i]
            c = a + 2 * d
            if c in s_set:
                diffs.add(d)
    return len(diffs)


def _greedy_construct():
    """Greedy: build a 32-element set in 1..512 maximizing progression differences."""
    best = list(range(1, 33))
    best_score = _score_set(best)
    
    # Try multiple random seeds for greedy
    for seed in range(20):
        if _time_exceeded():
            break
        rng = random.Random(seed)
        current = set()
        # Start with a random element
        current.add(rng.randint(1, 512))
        
        for _ in range(31):
            # Try adding each candidate, pick the one that maximizes score
            best_candidate = None
            best_candidate_score = -1
            # Sample candidates for speed
            candidates = list(range(1, 513))
            rng.shuffle(candidates)
            # Only check first 100 candidates for speed
            for c in candidates[:100]:
                if c in current:
                    continue
                new_set = current | {c}
                sc = _score_set(list(new_set))
                if sc > best_candidate_score:
                    best_candidate_score = sc
                    best_candidate = c
            if best_candidate is not None:
                current.add(best_candidate)
        
        final_set = list(current)
        if len(final_set) == 32:
            sc = _score_set(final_set)
            if sc > best_score:
                best_score = sc
                best = final_set
    
    return best, best_score


def _local_search(initial_set, max_iters=5000):
    """Local search: swap elements to improve score."""
    current = list(initial_set)
    current_set = set(current)
    current_score = _score_set(current)
    best = list(current)
    best_score = current_score
    
    for it in range(max_iters):
        if _time_exceeded():
            break
        # Try a random swap
        i = random.randint(0, 31)
        # Pick a random replacement from 1..512 not in current
        while True:
            c = random.randint(1, 512)
            if c not in current_set:
                break
        old = current[i]
        new_set = (current_set - {old}) | {c}
        sc = _score_set(list(new_set))
        if sc >= current_score:
            current[i] = c
            current_set = new_set
            current_score = sc
            if sc > best_score:
                best_score = sc
                best = list(current)
    
    return best, best_score


def _multi_start_greedy_with_ls():
    """Multiple greedy starts followed by local search."""
    global_best = list(range(1, 33))
    global_best_score = _score_set(global_best)
    
    for seed in range(50):
        if _time_exceeded():
            break
        rng = random.Random(seed * 1000 + 42)
        # Build a random initial set of 32
        current = set()
        while len(current) < 32:
            current.add(rng.randint(1, 512))
        
        # Local search from this start
        res, sc = _local_search(list(current), max_iters=2000)
        if sc > global_best_score:
            global_best_score = sc
            global_best = res
    
    return global_best, global_best_score


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with the trivial set
    best = list(range(1, 33))
    best_score = _score_set(best)
    
    # Phase 1: Greedy construction
    g_best, g_score = _greedy_construct()
    if g_score > best_score:
        best_score = g_score
        best = g_best
    
    # Phase 2: Multi-start local search
    if not _time_exceeded():
        ls_best, ls_score = _multi_start_greedy_with_ls()
        if ls_score > best_score:
            best_score = ls_score
            best = ls_best
    
    # Phase 3: Additional local search from best found
    if not _time_exceeded():
        final_best, final_score = _local_search(best, max_iters=10000)
        if final_score > best_score:
            best_score = final_score
            best = final_best
    
    # Ensure exactly 32 distinct integers in 1..512
    best = sorted(set(best))
    if len(best) != 32:
        # Fallback
        best = list(range(1, 33))
    
    return best


# Pre-run some computation at import time to warm up
# (This is optional; the evaluator calls construct())
