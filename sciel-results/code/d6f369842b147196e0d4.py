"""Improved program: search for distinct multiples minimizing span."""
import time
import math
from itertools import permutations, combinations
from collections import Counter

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAXV = 1000000
N = len(DIVISORS)
DEADLINE = time.time() + 75  # safety margin vs 90s wall-clock


def _smallest_multiple_at_least(d, L):
    """Return smallest multiple of d that is >= L."""
    return ((L + d - 1) // d) * d


def _evaluate_window(L):
    """For a given lower bound L, compute smallest multiple of each divisor >= L.
    Return (span, reps) if all distinct and <= MAXV, else None."""
    reps = []
    for d in DIVISORS:
        m = _smallest_multiple_at_least(d, L)
        if m > MAXV:
            return None
        reps.append(m)
    if len(set(reps)) != len(reps):
        return None
    span = max(reps) - min(reps)
    return (span, reps)


def _resolve_conflicts(base_reps, max_kicks=3):
    """Try to resolve conflicts by incrementing some values by multiples of their divisor.
    Returns (span, reps) if successful, else None."""
    # Check if already valid
    if len(set(base_reps)) == len(base_reps):
        if all(0 < r <= MAXV for r in base_reps):
            span = max(base_reps) - min(base_reps)
            return (span, base_reps[:])
    return None


def _try_window_with_adjustments(L, max_kicks=4):
    """Try window at L, and if conflicts exist, try small adjustments."""
    # First try the basic window
    result = _evaluate_window(L)
    if result is not None:
        return result
    
    # Get base reps
    base_reps = []
    for d in DIVISORS:
        m = _smallest_multiple_at_least(d, L)
        if m > MAXV:
            return None
        base_reps.append(m)
    
    # Find conflicts
    counts = Counter(base_reps)
    conflicted_values = {v for v, c in counts.items() if c > 1}
    
    if not conflicted_values:
        return None
    
    # Try incrementing conflicting divisors by 1, 2, ... up to max_kicks * d
    conflicting_indices = [i for i in range(N) if base_reps[i] in conflicted_values]
    
    # Try all combinations of kick amounts for conflicting indices
    # For each conflicting index, try kicks 1..max_kicks
    # Limit the search space
    if len(conflicting_indices) > 4:
        # Too many conflicts, try a simpler approach
        # Just try incrementing each by 1
        reps = base_reps[:]
        for i in conflicting_indices:
            reps[i] += DIVISORS[i]
            if reps[i] > MAXV:
                return None
        if len(set(reps)) == len(reps):
            span = max(reps) - min(reps)
            return (span, reps)
        return None
    
    # Try all combinations of kick amounts (1 to max_kicks) for each conflicting index
    kick_ranges = [range(1, max_kicks + 1)] * len(conflicting_indices)
    for kicks in __import__('itertools').product(*kick_ranges):
        reps = base_reps[:]
        valid = True
        for idx, kick in zip(conflicting_indices, kicks):
            d = DIVISORS[idx]
            reps[idx] = base_reps[idx] + kick * d
            if reps[idx] > MAXV:
                valid = False
                break
        if not valid:
            continue
        if len(set(reps)) != len(reps):
            continue
        span = max(reps) - min(reps)
        return (span, reps)
    
    return None


def _local_search(reps, max_iterations=500):
    """Local search: try perturbing each representative by +/- k*d for small k."""
    best_span = max(reps) - min(reps)
    best_reps = reps[:]
    current_reps = reps[:]
    
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        improved = False
        for i in range(N):
            d = DIVISORS[i]
            for k in [-3, -2, -1, 1, 2, 3]:
                new_val = current_reps[i] + k * d
                if new_val <= 0 or new_val > MAXV:
                    continue
                new_reps = current_reps[:]
                new_reps[i] = new_val
                if len(set(new_reps)) != len(new_reps):
                    continue
                span = max(new_reps) - min(new_reps)
                if span < best_span:
                    best_span = span
                    best_reps = new_reps
                    current_reps = new_reps
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return (best_span, best_reps)


def construct():
    best_span = float('inf')
    best_reps = None
    
    # Strategy 1: Quick baseline scan
    for L in range(1, 500):
        if time.time() > DEADLINE:
            break
        result = _evaluate_window(L)
        if result is not None:
            span, reps = result
            if span < best_span:
                best_span = span
                best_reps = reps[:]
    
    # Strategy 2: Targeted search - L values that are multiples of divisors
    # The minimum representative must be a multiple of some divisor
    # Focus on ranges where the span could be small
    for d in DIVISORS:
        # Try L = m*d for m in a reasonable range
        # The span is bounded by ~max(DIVISORS) in the best case
        # So L should be such that all multiples fit in [L, L+span]
        # Try a wide range of m values
        start_mult = max(1, 100 // d)
        end_mult = min(MAXV // d, 50000)
        
        # Use adaptive step: try all multiples for small d, sampled for large d
        step = 1 if d <= 15 else 2
        
        for mult in range(start_mult, end_mult + 1, step):
            if time.time() > DEADLINE:
                break
            L = mult * d
            if L > MAXV:
                break
            result = _try_window_with_adjustments(L, max_kicks=3)
            if result is not None:
                span, reps = result
                if span < best_span:
                    best_span = span
                    best_reps = reps[:]
    
    # Strategy 3: More targeted search around the current best
    if best_reps is not None:
        # The best min value is a candidate L
        min_val = min(best_reps)
        # Try L values around min_val
        for delta in range(-50, 51):
            L = min_val + delta
            if L < 1 or L > MAXV:
                continue
            result = _try_window_with_adjustments(L, max_kicks=4)
            if result is not None:
                span, reps = result
                if span < best_span:
                    best_span = span
                    best_reps = reps[:]
    
    # Strategy 4: Local search from best solution
    if best_reps is not None:
        span, reps = _local_search(best_reps, max_iterations=1000)
        if span < best_span:
            best_span = span
            best_reps = reps
    
    # Strategy 5: Try more L values with higher kick allowance
    if time.time() < DEADLINE - 10:
        for d in DIVISORS:
            start_mult = max(1, 1000 // d)
            end_mult = min(MAXV // d, 20000)
            for mult in range(start_mult, end_mult + 1, 3):
                if time.time() > DEADLINE:
                    break
                L = mult * d
                if L > MAXV:
                    break
                result = _try_window_with_adjustments(L, max_kicks=5)
                if result is not None:
                    span, reps = result
                    if span < best_span:
                        best_span = span
                        best_reps = reps[:]
    
    # Final local search polish
    if best_reps is not None and time.time() < DEADLINE - 5:
        span, reps = _local_search(best_reps, max_iterations=2000)
        if span < best_span:
            best_span = span
            best_reps = reps
    
    if best_reps is None:
        # Fallback: return the divisors themselves (valid but not optimal)
        return DIVISORS[:]
    
    # Ensure we return Python ints
    return [int(x) for x in best_reps]
# EVOLVE-BLOCK-END
